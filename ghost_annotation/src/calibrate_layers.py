#!/usr/bin/env python3
"""
calibrate_layer_elevations.py

One-off helper: run this against a RAW (pre-telemetry-reduction) bag that
still has per-point `ring` data. It computes the empirical median elevation
angle for each ring across many scans, and prints a LAYER_ELEVATIONS_DEG
list ready to paste into annotator_node.py / grid_utils usage.

This replaces guessing at "even spacing" with your sensor's actual
per-layer beam angles, which is what compute_polar_grid needs for correct
nearest-layer assignment when binning the (unordered, ring-less) reduced
data later.

Usage:
    python3 calibrate_layer_elevations.py /path/to/raw_bag.mcap /topic/name

Adjust RING_FIELD_NAME below if your driver names the field something
other than "ring" (e.g. "channel", "laser_id", "layer").
"""

import sys
import math
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2 as pc2

RING_FIELD_NAME = "ring"  # <-- adjust if your raw bag names this differently
MAX_SCANS_TO_SAMPLE = 50   # plenty for a stable median; raise if layers look noisy


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <bag_path> <topic_name>")
        sys.exit(1)
    bag_path, topic = sys.argv[1], sys.argv[2]

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        print(f"Topic '{topic}' not found. Available: {list(type_map.keys())}")
        sys.exit(1)
    msg_type = get_message(type_map[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    elevations_by_ring: dict[int, list] = {}
    n_scans = 0
    while reader.has_next() and n_scans < MAX_SCANS_TO_SAMPLE:
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, msg_type)

        field_names = [f.name for f in msg.fields]
        if RING_FIELD_NAME not in field_names:
            print(f"Field '{RING_FIELD_NAME}' not found in cloud fields: {field_names}")
            print("Update RING_FIELD_NAME at the top of this script and re-run.")
            sys.exit(1)

        pts = pc2.read_points(msg, field_names=("x", "y", "z", RING_FIELD_NAME), skip_nans=True)
        x, y, z, ring = pts["x"], pts["y"], pts["z"], pts[RING_FIELD_NAME]
        elevation_deg = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))

        for r in np.unique(ring):
            mask = ring == r
            elevations_by_ring.setdefault(int(r), []).extend(elevation_deg[mask].tolist())
        n_scans += 1

    if not elevations_by_ring:
        print("No data collected -- check topic name and field name.")
        sys.exit(1)

    print(f"Sampled {n_scans} scans.\n")
    rows = sorted(elevations_by_ring.keys())
    medians = []
    print(f"{'ring':>6} {'median_deg':>12} {'std_deg':>10} {'n_pts':>10}")
    for r in rows:
        vals = np.array(elevations_by_ring[r])
        med = float(np.median(vals))
        std = float(np.std(vals))
        medians.append(med)
        print(f"{r:>6} {med:>12.3f} {std:>10.3f} {len(vals):>10}")

    # Sort descending (top layer first) to match the row-0-is-top convention
    # used throughout grid_utils / the annotator node.
    sorted_medians = sorted(medians, reverse=True)
    print("\nPaste this into your constants (top layer first, row 0 = top):\n")
    print("LAYER_ELEVATIONS_DEG = [")
    for m in sorted_medians:
        print(f"    {m:.3f},")
    print("]")


if __name__ == "__main__":
    main()
