#!/usr/bin/env python3
"""
classify_bag.py

Runs the trained artifact classifier over every lidar scan in a bag.

The model is picked from the bag's point fields: artifact_full.joblib when
the cloud has an `intensity` field, artifact_xyz.joblib otherwise.

Outputs, under --out-dir:

    <bag>_artifact_prob.npy   (N, rings, cols) uint8, probability * 255 on
                              the scan grid of scan_grid.py, 0 where there
                              is no classifiable return
    <bag>_summary.json        counts and per-scan statistics
    <bag>_classified/         (with --write-bag) a bag, in the storage format
                              of the input, containing
        <topic>/classified      every point, original fields plus float32
                                `artifact_prob`  (colour by it in Foxglove)
        <topic>/filtered        the scan with flagged points removed
        <topic>/artifacts       only the flagged points
        /tf, /tf_static         copied through
    <file>.jsonl              (with --proposals) flagged cells in the
                              annotator's ledger format, for review

Usage (ROS 2 environment with rosbag2_py, plus scikit-learn or XGBoost):

    python3 classify_bag.py <bag> --project <project dir> \
        [--threshold 0.3] [--write-bag] [--proposals proposals.jsonl]

Returns nearer than features.min_range_m are never classified. In
--write-bag output they carry artifact_prob = -1 and are left out of
<topic>/filtered unless --keep-near is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact_features as af
import config
import scan_grid as sg

BATCH = 128


def load_model(model_dir: str, has_intensity: bool, force: str | None):
    import joblib

    variant = force or ("full" if has_intensity else "xyz")
    path = os.path.join(model_dir, f"artifact_{variant}.joblib")
    if not os.path.exists(path) and variant == "full" and force is None:
        variant = "xyz"
        path = os.path.join(model_dir, "artifact_xyz.joblib")
    if not os.path.exists(path):
        sys.exit(f"Model not found: {path}")
    bundle = joblib.load(path)
    if bundle["use_intensity"] and not has_intensity:
        sys.exit(f"{path} needs intensity, which this bag does not have.")
    problem = config.model_problem(bundle)
    if problem:
        sys.exit(f"{path} {problem}. Retrain it.")
    return bundle, path


def classify_cache(cache: dict, bundle: dict) -> np.ndarray:
    """Probability grid (N, rings, cols) float32 for every scan of a cached bag."""
    model = bundle["model"]
    offsets = tuple(bundle["time_offsets"])
    n = cache["range_mm"].shape[0]
    prob = np.zeros((n, sg.N_RINGS, sg.N_COLS), np.float32)
    for s in range(0, n, BATCH):
        idx = np.arange(s, min(n, s + BATCH))
        rng = af.to_metres(np.asarray(cache["range_mm"][idx]))
        stack = af.gather_time_stack(cache["range_mm"], cache["hdr_t"], idx, offsets)
        inten = (np.asarray(cache["intensity"][idx])
                 if bundle["use_intensity"] else None)
        feats = af.compute_features(rng, stack, inten, offsets)
        cand = af.candidates(rng)
        if cand.any():
            block = prob[idx]
            block[cand] = model.predict_proba(feats[cand])[:, 1]
            prob[idx] = block
        if (s // BATCH) % 20 == 0:
            print(f"  classified {idx[-1] + 1}/{n} scans", flush=True)
    return prob


def summarise(prob, cache, threshold) -> dict:
    valid = np.asarray(cache["range_mm"]) >= af.MIN_RANGE_M * 1000
    flag = prob >= threshold
    per_scan = flag.sum(axis=(1, 2))
    rng = np.asarray(cache["range_mm"])[flag].astype(np.float32) / 1000.0
    return {
        "n_scans": int(prob.shape[0]),
        "threshold": threshold,
        "returns_classified": int(valid.sum()),
        "returns_flagged": int(flag.sum()),
        "flagged_fraction": float(flag.sum() / max(valid.sum(), 1)),
        "flagged_per_scan": {
            "mean": float(per_scan.mean()),
            "median": float(np.median(per_scan)),
            "p95": float(np.percentile(per_scan, 95)),
            "max": int(per_scan.max()),
        },
        "scans_with_any_flag": int((per_scan > 0).sum()),
        "flagged_range_m_percentiles": (
            {str(q): float(np.percentile(rng, q)) for q in (5, 25, 50, 75, 95)}
            if len(rng) else {}),
        "flagged_by_ring": flag.sum(axis=(0, 2)).tolist(),
    }


def write_proposals(path, bag_path, prob, cache, threshold, every):
    """Flagged cells as ledger snapshot records (one per `every`-th scan)."""
    session = "model_proposals"
    with open(path, "w") as f:
        f.write(json.dumps({
            "type": "session", "session": session,
            "bag": os.path.basename(bag_path),
            "layer_elevations_deg": sg.LAYER_ELEV_DEG.tolist(),
            "grid_cols": sg.N_COLS, "azimuth_offset_deg": 0.0,
        }) + "\n")
        n = 0
        for i in range(0, prob.shape[0], every):
            rings, cols = np.nonzero(prob[i] >= threshold)
            if len(rings) == 0:
                continue
            sel = [[int(r), int(c), round(float(cache["range_mm"][i, r, c]) / 1000.0, 3)]
                   for r, c in zip(rings, cols)]
            f.write(json.dumps({
                "bag": os.path.basename(bag_path),
                "scan_stamp_ns": int(cache["rec_t"][i]),
                "session": session, "selections": sel,
            }) + "\n")
            n += 1
    print(f"  wrote proposals for {n} scans to {path}")


def write_bag(bag_path, out_uri, topic, prob, threshold, keep_near):
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import PointCloud2, PointField

    reader = sg.open_reader(bag_path)
    # Written in the storage format of the input.
    storage = reader.get_metadata().storage_identifier or "mcap"
    topics = {t.name: t for t in reader.get_all_topics_and_types()}
    passthrough = [t for t in ("/tf", "/tf_static") if t in topics]
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic] + passthrough))

    writer = rosbag2_py.SequentialWriter()
    options = {"storage_preset_profile": "zstd_fast"} if storage == "mcap" else {}
    writer.open(rosbag2_py.StorageOptions(uri=out_uri, storage_id=storage, **options),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for t in passthrough:
        writer.create_topic(topics[t])
    out_topics = {k: f"{topic}/{k}" for k in ("classified", "filtered", "artifacts")}
    for i, name in enumerate(out_topics.values()):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=1000 + i, name=name, type="sensor_msgs/msg/PointCloud2",
            serialization_format="cdr"))

    def cloud(src, pts, fields, step):
        m = PointCloud2()
        m.header = src.header
        m.height, m.width = 1, len(pts)
        m.fields, m.is_bigendian = fields, False
        m.point_step, m.row_step = step, step * len(pts)
        m.is_dense = src.is_dense
        m.data = pts.tobytes()
        return m

    i = 0
    dt_in = dt_out = None
    while reader.has_next():
        name, data, t = reader.read_next()
        if name != topic:
            writer.write(name, data, t)
            continue
        msg = deserialize_message(data, PointCloud2)
        if dt_in is None:
            dt_in = sg.pc2_dtype(msg)
            dt_out = np.dtype({
                "names": list(dt_in.names) + ["artifact_prob"],
                "formats": [dt_in.fields[n][0] for n in dt_in.names] + ["<f4"],
                "offsets": [dt_in.fields[n][1] for n in dt_in.names] + [msg.point_step],
                "itemsize": msg.point_step + 4,
            })
            out_fields = list(msg.fields) + [PointField(
                name="artifact_prob", offset=msg.point_step,
                datatype=PointField.FLOAT32, count=1)]
        pts = np.frombuffer(bytes(msg.data), dtype=dt_in, count=msg.width * msg.height)
        ring, col, rng, ok = sg.cell_of_points(pts["x"], pts["y"], pts["z"])
        p = np.full(len(pts), -1.0, np.float32)
        # Only the nearest return of a cell was classified (multi-echo
        # recordings have several); the others stay at -1 and are kept.
        flat = ring * sg.N_COLS + col
        nearest = np.full(sg.N_RINGS * sg.N_COLS, np.inf)
        np.minimum.at(nearest, flat[ok], rng[ok])
        cls = ok & (rng >= af.MIN_RANGE_M)
        scored = cls & (rng <= nearest[flat])
        p[scored] = prob[i, ring[scored], col[scored]]
        out = np.zeros(len(pts), dt_out)
        for n in dt_in.names:
            out[n] = pts[n]
        out["artifact_prob"] = p
        flagged = p >= threshold
        keep = ~flagged & (cls | keep_near) & ok
        writer.write(out_topics["classified"],
                     serialize_message(cloud(msg, out, out_fields, dt_out.itemsize)), t)
        writer.write(out_topics["filtered"],
                     serialize_message(cloud(msg, pts[keep], list(msg.fields), msg.point_step)), t)
        writer.write(out_topics["artifacts"],
                     serialize_message(cloud(msg, pts[flagged], list(msg.fields), msg.point_step)), t)
        i += 1
        if i % 5000 == 0:
            print(f"  wrote {i} scans", flush=True)
    del writer
    print(f"  wrote {i} scans to {out_uri}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--topic", default=None,
                    help="lidar topic (default: the project's default_topic, "
                         "or the bag's only PointCloud2 topic)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="flag returns with probability >= this "
                         "(default: the model's stored threshold)")
    ap.add_argument("--variant", choices=("xyz", "full"), default=None,
                    help="force a model instead of choosing by point fields")
    ap.add_argument("--write-bag", action="store_true")
    ap.add_argument("--keep-near", action="store_true",
                    help="keep returns under features.min_range_m in "
                         "<topic>/filtered")
    ap.add_argument("--proposals", default=None, metavar="FILE.jsonl")
    ap.add_argument("--proposals-every", type=int, default=1)
    config.add_arguments(ap)
    args = ap.parse_args()
    cfg = config.from_args(args)
    args.cache_dir, args.model_dir = cfg["paths"]["cache"], cfg["paths"]["models"]
    args.out_dir = cfg["paths"]["out"]

    name = os.path.splitext(os.path.basename(args.bag))[0]
    cached = os.path.exists(sg.cache_paths(args.cache_dir, name)["range_mm"])
    if not cached or args.write_bag:
        args.topic = sg.resolve_topic(args.bag, args.topic, cfg["default_topic"])
    if not cached:
        print(f"Caching scans of {os.path.basename(args.bag)} ...", flush=True)
        sg.cache_bag(args.bag, args.cache_dir, name, args.topic)
    cache = sg.load_cache(args.cache_dir, name)

    bundle, path = load_model(args.model_dir, cache["intensity"] is not None, args.variant)
    threshold = args.threshold if args.threshold is not None else bundle["default_threshold"]
    print(f"Model: {path}  threshold: {threshold}")

    prob = classify_cache(cache, bundle)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"{name}_artifact_prob.npy"),
            np.round(prob * 255).astype(np.uint8))
    summary = summarise(prob, cache, threshold)
    summary["bag"] = os.path.basename(args.bag)
    summary["model"] = os.path.basename(path)
    with open(os.path.join(args.out_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    fps = summary["flagged_per_scan"]
    print(f"Flagged {summary['returns_flagged']} of {summary['returns_classified']} "
          f"returns ({100 * summary['flagged_fraction']:.3f}%); per scan: "
          f"mean {fps['mean']:.1f}, median {fps['median']:.0f}, "
          f"p95 {fps['p95']:.0f}, max {fps['max']}")

    if args.proposals:
        write_proposals(args.proposals, args.bag, prob, cache, threshold,
                        args.proposals_every)
    if args.write_bag:
        out_uri = os.path.join(args.out_dir, f"{name}_classified")
        if os.path.exists(out_uri):
            sys.exit(f"{out_uri} already exists; remove it or pick another --out-dir.")
        write_bag(args.bag, out_uri, args.topic, prob, threshold, args.keep_near)


if __name__ == "__main__":
    main()
