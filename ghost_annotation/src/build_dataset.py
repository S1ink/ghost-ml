#!/usr/bin/env python3
"""
build_dataset.py

Reads the annotation ledger and original ROS2 bag files, rebuilds each
annotated scan's range image and label map using the exact grid parameters
recorded in the ledger's session header, applies a temporal train/val
split within each bag, and writes an HDF5 cache for training.

Usage:
    python3 build_dataset.py \
        --ledger range_annotations_ledger.jsonl \
        --bags /data/bag1.mcap /data/bag2.mcap \
        --output dataset.h5 \
        [--topic /multiscan/lidar_scan] \
        [--val-fraction 0.2] \
        [--temporal-gap-s 30]

HDF5 schema:
    /attrs         build metadata (date, params, split settings)
    /train/
        range_imgs  (N, H, W)  float32  raw range in metres, NaN for empty cells
        label_maps  (N, H, W)  int8     1=artifact  0=clean  -1=no point/unlabeled
        stamp_ns    (N,)       int64
        bag         (N,)       bytes    original bag basename
    /val/           same structure

Label map semantics:
    1   annotated artifact
    0   reviewed clean: point exists and was not selected as an artifact.
        Valid negative training signal.
   -1   no point in this cell (telemetry gap, or outside FOV).
        MUST be masked out of training loss -- not a known clean sample.

The three-way distinction matters. Scans absent from the ledger entirely
are unreviewed unknowns and are excluded from the cache entirely; they
must not be used as negatives.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_utils import compute_polar_grid, build_range_image

# Fallback grid params for legacy per-point records that pre-date session headers.
# These should match the constants the annotator was running with at the time.
FALLBACK_LAYER_ELEVATIONS_DEG = [
    42.2, 35.0, 29.0, 24.5, 19.0, 14.8, 10.0,
    5.5, 1.0, -4.0, -9.5, -14.0, -18.5, -22.2,
]
FALLBACK_GRID_COLS        = 360
FALLBACK_AZIMUTH_OFFSET   = 0.0

LABEL_ARTIFACT  = np.int8(1)
LABEL_CLEAN     = np.int8(0)
LABEL_UNLABELED = np.int8(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Ledger parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_ledger(path: str):
    """
    Parses the annotation ledger.

    Returns:
        sessions:    {session_id -> {layer_elevations_deg, grid_cols, azimuth_offset_deg}}
        annotations: {(bag_basename, stamp_ns) -> (session_id, frozenset of (ring, col))}
                     Last snapshot per key is authoritative (handles removals).
    """
    sessions    = {}
    annotations = {}
    snapshotted = set()   # (bag, stamp_ns) pairs with a definitive snapshot record

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Session header -- no scan_stamp_ns, carries grid metadata.
            if rec.get("type") == "session":
                sessions[rec["session"]] = {
                    "layer_elevations_deg": rec["layer_elevations_deg"],
                    "grid_cols":            rec["grid_cols"],
                    "azimuth_offset_deg":   rec.get("azimuth_offset_deg", 0.0),
                }
                continue

            bag      = rec.get("bag")
            stamp_ns = rec.get("scan_stamp_ns")
            if bag is None or stamp_ns is None:
                continue
            key = (bag, stamp_ns)

            if "selections" in rec:
                # Snapshot format: last write wins.
                sels = frozenset(
                    (int(item[0]), int(item[1]))
                    for item in rec["selections"]
                    if isinstance(item, list) and len(item) >= 2
                )
                annotations[key] = (rec.get("session", ""), sels)
                snapshotted.add(key)

            elif "ring" in rec and "azimuth_idx" in rec and key not in snapshotted:
                # Legacy per-point format. Accumulate unless a snapshot exists.
                ring, col = int(rec["ring"]), int(rec["azimuth_idx"])
                session   = rec.get("session", "")
                if key in annotations:
                    old_session, old_sels = annotations[key]
                    annotations[key] = (old_session, old_sels | frozenset([(ring, col)]))
                else:
                    annotations[key] = (session, frozenset([(ring, col)]))

    return sessions, annotations


# ─────────────────────────────────────────────────────────────────────────────
# Label map construction
# ─────────────────────────────────────────────────────────────────────────────

def build_label_map(
    selections:     frozenset,
    point_idx_img:  np.ndarray,
    grid_rows:      int,
    grid_cols:      int,
) -> np.ndarray:
    """
    (grid_rows, grid_cols) int8 label map.

    Start with -1 everywhere. For every cell that has a real point, set to 0
    (clean). Then override selected cells to 1 (artifact). This gives the
    correct three-way encoding: cells with no point stay -1 (unlabeled/masked),
    reviewed clean cells are 0, artifacts are 1.
    """
    lmap = np.full((grid_rows, grid_cols), LABEL_UNLABELED, dtype=np.int8)
    lmap[point_idx_img >= 0] = LABEL_CLEAN
    for r, c in selections:
        if 0 <= r < grid_rows and 0 <= c < grid_cols:
            lmap[r, c] = LABEL_ARTIFACT
    return lmap


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    stamp_ns_list:  list[int],
    val_fraction:   float,
    temporal_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns boolean arrays (train_mask, val_mask) over stamp_ns_list.

    Timeline layout (oldest -> newest):
        [───── train ─────][── gap ──][── val ──]

    The gap region is excluded from both sets. This ensures that even if
    consecutive annotated scans in a bag are only a few seconds apart,
    there is a hard temporal boundary between train and val.

    If the time span is too short for the gap + val window, val gets whatever
    is left at the end (may be empty -- caller will warn about this).
    """
    stamps     = np.array(stamp_ns_list, dtype=np.int64)
    t_min, t_max = stamps.min(), stamps.max()
    t_span     = t_max - t_min

    if t_span == 0:
        return np.ones(len(stamps), dtype=bool), np.zeros(len(stamps), dtype=bool)

    gap_ns      = int(temporal_gap_s * 1e9)
    val_start   = t_max - int(val_fraction * t_span)
    gap_start   = val_start - gap_ns

    train_mask  = stamps < gap_start
    val_mask    = stamps >= val_start
    return train_mask, val_mask


# ─────────────────────────────────────────────────────────────────────────────
# Bag processing
# ─────────────────────────────────────────────────────────────────────────────

def process_bag(
    bag_path:    str,
    topic:       str,
    stamp_map:   dict,   # stamp_ns -> (session_id, selections frozenset)
    sessions:    dict,
    default_params: dict,
) -> list:
    """
    Opens one bag and rebuilds the range image + label map for every
    annotated scan in stamp_map.

    Returns list of (stamp_ns, range_img, label_map).
    Scans that can't be found or have a timestamp mismatch are skipped.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs_py import point_cloud2 as pc2

    storage_opts   = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_opts = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader         = rosbag2_py.SequentialReader()
    reader.open(storage_opts, converter_opts)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        print(f"  WARNING: topic '{topic}' not found in {os.path.basename(bag_path)}")
        print(f"  Available topics: {list(type_map.keys())}")
        reader.close()
        return []

    msg_type = get_message(type_map[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    results = []
    items   = sorted(stamp_map.items())   # process in time order (matches seek direction)
    n       = len(items)

    for i, (stamp_ns, (session_id, selections)) in enumerate(items):
        print(f"  [{i+1:>4}/{n}] stamp_ns={stamp_ns} | {len(selections):>3} artifacts",
              end="", flush=True)

        reader.seek(stamp_ns)
        if not reader.has_next():
            print(" SKIP (seek past end of bag)")
            continue

        _, data, msg_t = reader.read_next()

        # Allow a small tolerance -- MCAP timestamps are occasionally off by
        # a few microseconds due to recording jitter.
        if abs(msg_t - stamp_ns) > 5_000_000:   # 5 ms
            print(f" SKIP (stamp mismatch: got {msg_t}, expected {stamp_ns})")
            continue

        msg = deserialize_message(data, msg_type)
        pts = pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
        xyz = np.column_stack([
            np.asarray(pts["x"], dtype=np.float64),
            np.asarray(pts["y"], dtype=np.float64),
            np.asarray(pts["z"], dtype=np.float64),
        ])

        # Use grid params from the session that produced this annotation.
        # Fall back to the most common session's params if this one is unknown.
        sp             = sessions.get(session_id, default_params)
        layer_elev_rad = np.radians(np.array(sp["layer_elevations_deg"], dtype=np.float64))
        grid_cols      = sp["grid_cols"]
        az_offset_rad  = math.radians(sp.get("azimuth_offset_deg", 0.0))

        grid = compute_polar_grid(
            xyz, layer_elev_rad, grid_cols, azimuth_offset_rad=az_offset_rad,
        )
        range_img, point_idx_img = build_range_image(
            grid["row_idx"], grid["col_idx"], grid["ranges"], grid["orig_indices"],
            grid["grid_rows"], grid_cols,
        )
        lmap = build_label_map(selections, point_idx_img, grid["grid_rows"], grid_cols)
        results.append((stamp_ns, range_img, lmap))
        print(f" ok  ({grid['grid_rows']}×{grid_cols})")

    reader.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 output
# ─────────────────────────────────────────────────────────────────────────────

def write_split(grp: h5py.Group, items: list):
    """
    items: list of (stamp_ns, range_img, label_map, bag_basename)
    All range images and label maps must have the same shape.
    """
    if not items:
        return

    stamps, ranges, labels, bags = zip(*items)

    grp.create_dataset("range_imgs", data=np.stack(ranges), dtype=np.float32,
                        compression="gzip", compression_opts=4)
    grp.create_dataset("label_maps", data=np.stack(labels), dtype=np.int8,
                        compression="gzip", compression_opts=4)
    grp.create_dataset("stamp_ns",   data=np.array(stamps, dtype=np.int64))

    max_len = max(len(b) for b in bags)
    grp.create_dataset("bag",
                        data=np.array([b.encode() for b in bags],
                                      dtype=f"S{max(max_len, 1)}"))

    lmaps = np.stack(labels)
    grp.attrs["n_scans"]     = len(stamps)
    grp.attrs["n_artifact"]  = int((lmaps == LABEL_ARTIFACT).sum())
    grp.attrs["n_clean"]     = int((lmaps == LABEL_CLEAN).sum())
    grp.attrs["n_unlabeled"] = int((lmaps == LABEL_UNLABELED).sum())


def write_hdf5(path: str, train_items: list, val_items: list, metadata: dict):
    with h5py.File(path, "w") as f:
        for k, v in metadata.items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)

        write_split(f.create_group("train"), train_items)
        write_split(f.create_group("val"),   val_items)

    print(f"\nDataset written to {path}")
    with h5py.File(path, "r") as f:
        for split in ("train", "val"):
            if split not in f:
                continue
            g = f[split]
            n  = g.attrs.get("n_scans", 0)
            pa = g.attrs.get("n_artifact", 0)
            nc = g.attrs.get("n_clean", 0)
            nu = g.attrs.get("n_unlabeled", 0)
            pos_rate = pa / max(pa + nc, 1) * 100
            print(f"  {split:5s}: {n:>4} scans  "
                  f"artifact={pa:>7,}  clean={nc:>9,}  "
                  f"unlabeled={nu:>8,}  pos_rate={pos_rate:.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger",           required=True,
                    help="Path to range_annotations_ledger.jsonl")
    ap.add_argument("--bags",             required=True, nargs="+",
                    help="Full paths to the .mcap bag files referenced by the ledger")
    ap.add_argument("--output",           default="dataset.h5")
    ap.add_argument("--topic",            default="/multiscan/lidar_scan")
    ap.add_argument("--val-fraction",     type=float, default=0.2,
                    help="Fraction of each bag's time span to use for validation")
    ap.add_argument("--temporal-gap-s",   type=float, default=30.0,
                    help="Minimum time gap (seconds) between train and val scans")
    args = ap.parse_args()

    # Build basename -> full path map (error on ambiguity)
    bag_map = {}
    for bp in args.bags:
        bn = os.path.basename(bp)
        if bn in bag_map:
            sys.exit(f"ERROR: duplicate bag basename '{bn}' in --bags. "
                     f"Rename one or move it to a different directory.")
        if not os.path.exists(bp):
            sys.exit(f"ERROR: bag not found: {bp}")
        bag_map[bn] = bp

    print(f"Loading ledger: {args.ledger}")
    sessions, annotations = load_ledger(args.ledger)
    print(f"  {len(sessions)} session header(s), "
          f"{len(annotations)} annotated scans total")
    if not sessions:
        print("  WARNING: no session headers found. Grid params will use "
              "fallback values for all scans. Re-run the annotator at least "
              "once to write a session header if you changed elevation angles.")

    # Group annotations by bag basename
    by_bag = defaultdict(dict)
    for (bag_bn, stamp_ns), val in annotations.items():
        by_bag[bag_bn][stamp_ns] = val

    train_items, val_items = [], []

    for bag_bn in sorted(by_bag):
        stamp_map = by_bag[bag_bn]
        print(f"\n{'─'*60}")
        print(f"Bag: {bag_bn}  ({len(stamp_map)} annotated scans)")

        if bag_bn not in bag_map:
            print(f"  SKIP: not found in --bags. Provide the full path to include it.")
            print(f"  Provided bags: {sorted(bag_map.keys())}")
            continue

        # Choose default grid params: use the most common session for this bag,
        # falling back to hard-coded defaults if no session headers exist.
        session_ids = [v[0] for v in stamp_map.values() if v[0] in sessions]
        if session_ids:
            most_common = Counter(session_ids).most_common(1)[0][0]
            default_params = sessions[most_common]
        else:
            default_params = {
                "layer_elevations_deg": FALLBACK_LAYER_ELEVATIONS_DEG,
                "grid_cols":            FALLBACK_GRID_COLS,
                "azimuth_offset_deg":   FALLBACK_AZIMUTH_OFFSET,
            }

        results = process_bag(
            bag_map[bag_bn], args.topic, stamp_map, sessions, default_params,
        )
        if not results:
            print(f"  No usable scans extracted from {bag_bn}")
            continue

        # Temporal split within this bag
        stamps     = [r[0] for r in results]
        train_mask, val_mask = temporal_split(
            stamps, args.val_fraction, args.temporal_gap_s,
        )
        n_train  = int(train_mask.sum())
        n_val    = int(val_mask.sum())
        n_gap    = len(stamps) - n_train - n_val

        # Report the time span of each set in human-readable form
        stamps_arr = np.array(stamps, dtype=np.int64)
        def fmt_span(mask):
            s = stamps_arr[mask]
            if len(s) < 2:
                return f"{len(s)} scan, span<1s"
            span_s = (s.max() - s.min()) / 1e9
            return f"{len(s)} scans, span={span_s:.1f}s"

        print(f"  train: {fmt_span(train_mask)}")
        print(f"  gap:   {n_gap} scans excluded ({args.temporal_gap_s:.0f}s buffer)")
        print(f"  val:   {fmt_span(val_mask)}")
        if n_val == 0:
            print(f"  WARNING: no val scans. Bag may be too short for a "
                  f"{args.temporal_gap_s}s gap + {args.val_fraction:.0%} val window. "
                  f"Try reducing --temporal-gap-s.")

        for i, (stamp_ns, range_img, lmap) in enumerate(results):
            item = (stamp_ns, range_img, lmap, bag_bn)
            if train_mask[i]:
                train_items.append(item)
            elif val_mask[i]:
                val_items.append(item)

    print(f"\n{'─'*60}")
    if not train_items and not val_items:
        sys.exit("ERROR: no data produced. Check --bags matches the bag names in the ledger.")

    # Use the first session's grid params for metadata (they should all match;
    # warn if they differ).
    unique_grid_configs = {
        (tuple(sp["layer_elevations_deg"]), sp["grid_cols"], sp["azimuth_offset_deg"])
        for sp in sessions.values()
    }
    if len(unique_grid_configs) > 1:
        print("WARNING: multiple distinct grid configurations found across sessions. "
              "The HDF5 will contain scans built with different grids -- ensure "
              "they're all the same shape before training, or filter by session.")

    first_sp = next(iter(sessions.values())) if sessions else default_params
    metadata = {
        "build_date":            datetime.now(timezone.utc).isoformat(),
        "val_fraction":          args.val_fraction,
        "temporal_gap_s":        args.temporal_gap_s,
        "topic":                 args.topic,
        "n_sessions":            len(sessions),
        "grid_cols":             first_sp["grid_cols"],
        "layer_elevations_deg":  str(first_sp["layer_elevations_deg"]),
        "azimuth_offset_deg":    first_sp.get("azimuth_offset_deg", 0.0),
    }

    write_hdf5(args.output, train_items, val_items, metadata)


if __name__ == "__main__":
    import rclpy
    rclpy.init(args=[])
    try:
        main()
    finally:
        rclpy.shutdown()
