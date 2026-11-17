#!/usr/bin/env python3
"""
scan_grid.py

Converts PointCloud2 scans of a spinning multi-layer lidar into a fixed
(rings, cols) grid and caches whole bags as dense numpy arrays. Also the one
place that touches ROS: bags are read with rosbag2_py.

The grid is described by a sensor profile (see sensors/multiscan.json and
sensor_profile.py), applied once per process with configure():

    cols                   azimuth samples per revolution in one layer
    layer_elevations_deg   elevation of every layer, ascending
    layer_azimuth0_deg     azimuth of one sample of every layer (same order);
                           layers may start at different azimuths
    scan_rate_hz           scans per second
    intensity_field        point field holding intensity (optional)
    intensity_max          raw intensity value stored as full scale

Grid convention (ledger selections `[ring, col]` index this grid directly):

    row  = ring, 0 = HIGHEST elevation layer ... rings-1 = lowest
    col  = floor((azimuth_deg + 180) / step), azimuth = atan2(y, x),
           step = 360 / cols

Each return goes to the layer with the nearest elevation and to the nearest
azimuth sample of that layer. When several returns share a cell (multi-echo
recordings) the grid holds the nearest.

The profile must be the same for every bag that shares a ledger and a set of
models: labels and features are grid cells. The cache directory remembers
the profile it was built with and refuses another one.

Cached arrays, per bag, under <cache_dir>/<name>_*.npy:

    range_mm   (N, rings, cols) uint16   range in millimetres, 0 = no return
                                         (ranges beyond 65.535 m saturate)
    intensity  (N, rings, cols) uint16   only if the bag has the intensity field
    hdr_t      (N,) int64                header stamp, ns
    rec_t      (N,) int64                bag receive stamp, ns (the ledger key)

Usage:
    python3 scan_grid.py <bag> <name> [--project DIR] [--topic TOPIC]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

# Set by configure(). Read them as sg.N_RINGS etc., never `from scan_grid import`.
PROFILE: dict | None = None
N_RINGS = 0
N_COLS = 0
COL_DEG = 1.0                 # azimuth step, degrees
LAYER_ELEV_DEG = None         # ascending (ring N_RINGS-1 ... ring 0)
LAYER_AZ0_DEG = None          # same order as LAYER_ELEV_DEG
_COL0 = None
SCAN_RATE_HZ = 0.0
INTENSITY_FIELD: str | None = "intensity"
INTENSITY_MAX = 65535.0


def configure(profile: dict) -> None:
    """Applies a sensor profile to this process."""
    global PROFILE, N_RINGS, N_COLS, COL_DEG, LAYER_ELEV_DEG, LAYER_AZ0_DEG, _COL0
    global SCAN_RATE_HZ, INTENSITY_FIELD, INTENSITY_MAX
    try:
        elev = np.array(profile["layer_elevations_deg"], np.float64)
        cols = int(profile["cols"])
        rate = float(profile["scan_rate_hz"])
    except KeyError as e:
        raise ValueError(f"sensor profile has no {e}") from None
    az0 = np.array(profile.get("layer_azimuth0_deg") or [-180.0] * len(elev), np.float64)
    if len(elev) < 3 or np.any(np.diff(elev) <= 0):
        raise ValueError("layer_elevations_deg must be at least 3 ascending values")
    if len(az0) != len(elev):
        raise ValueError("layer_azimuth0_deg needs one value per layer")
    if cols < 16 or rate <= 0:
        raise ValueError("sensor profile needs cols >= 16 and scan_rate_hz > 0")
    PROFILE = dict(profile)
    N_RINGS, N_COLS, COL_DEG = len(elev), cols, 360.0 / cols
    LAYER_ELEV_DEG, LAYER_AZ0_DEG = elev, az0
    _COL0 = np.floor((az0 + 180.0) / COL_DEG).astype(np.int64)
    SCAN_RATE_HZ = rate
    INTENSITY_FIELD = profile.get("intensity_field", "intensity") or None
    INTENSITY_MAX = float(profile.get("intensity_max", 65535))


def _need() -> None:
    if PROFILE is None:
        raise RuntimeError("no sensor profile: call config.load() or scan_grid.configure()")


def fingerprint() -> str:
    """Identifies the grid layout (not the sensor's name or scan rate)."""
    _need()
    az_col0 = np.mod(cell_azimuth_deg()[:, 0] + 180.0, 360.0)
    key = json.dumps([N_COLS, np.round(LAYER_ELEV_DEG, 2).tolist(),
                      np.round(az_col0 / COL_DEG, 1).tolist()])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def cell_azimuth_deg() -> np.ndarray:
    """(rings, cols) azimuth of the beam of every grid cell."""
    _need()
    cols = np.arange(N_COLS)
    az = np.zeros((N_RINGS, N_COLS))
    for ring in range(N_RINGS):
        k = N_RINGS - 1 - ring
        az[ring] = LAYER_AZ0_DEG[k] + ((cols - _COL0[k]) % N_COLS) * COL_DEG
    return az


_PC2_DTYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def pc2_dtype(msg) -> np.dtype:
    return np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [_PC2_DTYPES[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })


def cell_of_points(x, y, z):
    """
    Returns (ring, col, range_m, ok) for arrays of points. `ok` is False for
    non-finite or zero-range points.
    """
    _need()
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    xy = np.hypot(x, y)
    rng = np.hypot(xy, z)
    ok = np.isfinite(rng) & (rng > 1e-4)
    elev = np.degrees(np.arctan2(z, xy))
    k = np.abs(elev[:, None] - LAYER_ELEV_DEG[None, :]).argmin(axis=1)
    az = np.degrees(np.arctan2(y, x))
    idx = np.round((az - LAYER_AZ0_DEG[k]) / COL_DEG).astype(np.int64) % N_COLS
    col = (idx + _COL0[k]) % N_COLS
    ring = (N_RINGS - 1) - k
    return ring, col, rng, ok


def scan_to_grid(points: np.ndarray):
    """
    points: structured array with x, y, z (and optionally intensity).
    Returns (range_mm uint16 (rings, cols), intensity uint16 (rings, cols) or
             None, cell_flat int64 (n_points,) = ring*cols+col, -1 for invalid
             points).
    Several points can share a cell (multi-echo); the grids hold the nearest.
    """
    ring, col, rng, ok = cell_of_points(points["x"], points["y"], points["z"])
    flat = np.where(ok, ring * N_COLS + col, -1)
    range_mm = np.zeros(N_RINGS * N_COLS, np.uint16)
    # Multi-echo recordings put several returns in one cell. The nearest one
    # is kept, as in the annotator, so a ledger selection means that return.
    # Written far to near so the nearest lands last.
    keep = np.nonzero(ok)[0]
    keep = keep[np.argsort(-rng[keep], kind="stable")]
    v = flat[keep]
    range_mm[v] = np.clip(np.round(rng[keep] * 1000.0), 0, 65535).astype(np.uint16)
    inten = None
    if INTENSITY_FIELD and INTENSITY_FIELD in points.dtype.names:
        raw = points[INTENSITY_FIELD][keep]
        if INTENSITY_MAX != 65535.0:
            raw = np.round(raw.astype(np.float64) * (65535.0 / INTENSITY_MAX))
        inten = np.zeros(N_RINGS * N_COLS, np.uint16)
        inten[v] = np.clip(raw, 0, 65535).astype(np.uint16)
        inten = inten.reshape(N_RINGS, N_COLS)
    return range_mm.reshape(N_RINGS, N_COLS), inten, flat


# ─────────────────────────────────────────────────────────────────────────────
# Bags (the only ROS-dependent part)
# ─────────────────────────────────────────────────────────────────────────────

STORAGE_BY_EXT = {".mcap": "mcap", ".db3": "sqlite3"}
POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"


def storage_id(bag_path: str) -> str:
    """rosbag2 storage plugin for a bag file; '' lets rosbag2 detect it
    (bag directories with a metadata.yaml)."""
    return STORAGE_BY_EXT.get(os.path.splitext(bag_path)[1].lower(), "")


def open_reader(bag_path: str, topics: list | None = None):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id(bag_path)),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def pointcloud_topics(bag_path: str) -> list:
    """[{"name", "count"}] of the PointCloud2 topics of a bag, busiest first."""
    meta = open_reader(bag_path).get_metadata()
    out = [{"name": t.topic_metadata.name, "count": t.message_count}
           for t in meta.topics_with_message_count
           if t.topic_metadata.type == POINTCLOUD2]
    return sorted(out, key=lambda t: -t["count"])


def resolve_topic(bag_path: str, topic: str | None = None,
                  preferred: str | None = None) -> str:
    """
    The lidar topic to read: `topic` when given, else `preferred` when the
    bag has it, else the bag's only PointCloud2 topic. Several candidates and
    no preference is an error that lists them.
    """
    if topic:
        return topic
    names = [t["name"] for t in pointcloud_topics(bag_path)]
    if preferred in names:
        return preferred
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(f"{os.path.basename(bag_path)} has no PointCloud2 topic")
    raise ValueError(f"{os.path.basename(bag_path)} has several PointCloud2 topics "
                     f"({', '.join(names)}); name one with --topic")


def iter_scans(bag_path: str, topic: str | None = None):
    """Yields (rec_t_ns, hdr_t_ns, msg, structured_points) for every scan."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    reader = open_reader(bag_path, [resolve_topic(bag_path, topic)])
    dt = None
    while reader.has_next():
        _, data, rec_t = reader.read_next()
        msg = deserialize_message(data, PointCloud2)
        if dt is None:
            dt = pc2_dtype(msg)
        pts = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
        hdr_t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        yield rec_t, hdr_t, msg, pts


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def check_cache_dir(cache_dir: str) -> None:
    """Ties a cache directory to the sensor profile it was built with."""
    _need()
    mark = os.path.join(cache_dir, "sensor_profile.json")
    if os.path.exists(mark):
        with open(mark) as f:
            had = json.load(f)
        if had.get("fingerprint") != fingerprint():
            raise RuntimeError(
                f"{cache_dir} was built with another sensor profile "
                f"({had.get('profile', {}).get('name', '?')}). Labels and models "
                f"made with it do not fit the current one; use a new project "
                f"directory, or restore the profile.")
        return
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{mark}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"fingerprint": fingerprint(), "profile": PROFILE}, f, indent=1)
    os.replace(tmp, mark)


def cache_paths(cache_dir: str, name: str) -> dict:
    return {k: os.path.join(cache_dir, f"{name}_{k}.npy")
            for k in ("range_mm", "intensity", "hdr_t", "rec_t")}


def cache_bag(bag_path: str, cache_dir: str, name: str,
              topic: str | None = None) -> dict:
    """Extracts every scan of a bag into the dense grid cache."""
    check_cache_dir(cache_dir)
    ranges, intens, rec, hdr = [], [], [], []
    collisions = 0
    for rec_t, hdr_t, _, pts in iter_scans(bag_path, topic):
        r, i, flat = scan_to_grid(pts)
        n_valid = int((flat >= 0).sum())
        collisions += n_valid - int((r > 0).sum())
        ranges.append(r)
        if i is not None:
            intens.append(i)
        rec.append(rec_t)
        hdr.append(hdr_t)
        if len(rec) % 5000 == 0:
            print(f"  {len(rec)} scans", flush=True)
    if not rec:
        raise ValueError(f"no scans on {topic or 'the lidar topic'} in "
                         f"{os.path.basename(bag_path)}")
    p = cache_paths(cache_dir, name)
    np.save(p["range_mm"], np.stack(ranges))
    if intens:
        np.save(p["intensity"], np.stack(intens))
    np.save(p["hdr_t"], np.array(hdr, np.int64))
    np.save(p["rec_t"], np.array(rec, np.int64))
    print(f"{name}: {len(rec)} scans, intensity={'yes' if intens else 'no'}, "
          f"cell collisions={collisions}")
    return p


def load_cache(cache_dir: str, name: str, mmap: bool = True) -> dict:
    check_cache_dir(cache_dir)
    p = cache_paths(cache_dir, name)
    mode = "r" if mmap else None
    out = {
        "range_mm": np.load(p["range_mm"], mmap_mode=mode),
        "hdr_t": np.load(p["hdr_t"]),
        "rec_t": np.load(p["rec_t"]),
        "intensity": None,
    }
    if os.path.exists(p["intensity"]):
        out["intensity"] = np.load(p["intensity"], mmap_mode=mode)
    if out["range_mm"].shape[1:] != (N_RINGS, N_COLS):
        raise RuntimeError(f"cache of {name} is {out['range_mm'].shape[1:]}, the "
                           f"sensor profile says {(N_RINGS, N_COLS)}")
    return out


if __name__ == "__main__":
    import config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--topic", default=None)
    config.add_arguments(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    cache_bag(a.bag, cfg["paths"]["cache"],
              a.name or os.path.splitext(os.path.basename(a.bag))[0],
              resolve_topic(a.bag, a.topic, cfg["default_topic"]))
