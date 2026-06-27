#!/usr/bin/env python3
"""
annotator_node.py

Controller-driven annotation tool for flagging lidar edge-blur ("flying
pixel") points. Reads a ROS2 bag offline (no live playback needed), builds
a fixed polar range grid per scan using grid_utils, runs the heuristic
candidate detector, republishes a colorized point cloud for viewing in
Foxglove Studio, and lets you confirm/deny each candidate via an Xbox
controller (/joy messages). Labels are appended to a JSONL ledger as you go.

Run:
    ros2 run --prefix 'python3' . annotator_node.py
or just:
    python3 annotator_node.py
(it's a plain rclpy node, no package build needed for a first version)

=====================================================================
                    ALL USER-EDITABLE CONSTANTS
=====================================================================
Fill these in / adjust to taste. Nothing below this block should need
touching for routine use.
"""

import json
import math
import os
import struct
import time
from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField, Joy
from sensor_msgs_py import point_cloud2 as pc2
import rosbag2_py

from grid_utils import (
    compute_polar_grid,
    build_range_image,
    heuristic_candidate_scores,
    get_candidate_cells,
)

# ─────────────────────────────────────────────────────────────────────────
# BAG / TOPIC CONFIG
# ─────────────────────────────────────────────────────────────────────────
BAG_PATH = "/path/to/your.mcap"                  # <-- FILL IN
INPUT_CLOUD_TOPIC = "/multiscan/lidar_scan"       # <-- confirm matches your bag
OUTPUT_CLOUD_TOPIC = "/annotator/colorized_cloud"  # what you point Foxglove at

LEDGER_PATH = "labels_ledger.jsonl"
REVIEWED_SCANS_PATH = "reviewed_scans.jsonl"
SESSION_ID = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────────────────────────────────
# SENSOR / GRID GEOMETRY  -- FILL THESE IN FOR YOUR SENSOR
# ─────────────────────────────────────────────────────────────────────────
GRID_ROWS = 14
GRID_COLS = 360
ELEVATION_MIN_DEG = -32.5    # <-- CONFIRM: actual lower bound of your FOV
ELEVATION_MAX_DEG = 32.5     # <-- CONFIRM: actual upper bound of your FOV
AZIMUTH_OFFSET_DEG = 0.0     # <-- adjust if your driver has a known azimuth zero-offset

ELEVATION_MIN_RAD = math.radians(ELEVATION_MIN_DEG)
ELEVATION_MAX_RAD = math.radians(ELEVATION_MAX_DEG)
AZIMUTH_OFFSET_RAD = math.radians(AZIMUTH_OFFSET_DEG)

# ─────────────────────────────────────────────────────────────────────────
# HEURISTIC THRESHOLD -- deliberately loose; false positives just cost a
# quick deny click, false negatives cost a missed label. Tune by watching
# how many candidates show up per scan; you said 5-20 true errors per scan,
# so if you're seeing wildly more than that, raise the threshold.
# ─────────────────────────────────────────────────────────────────────────
HEURISTIC_THRESHOLD = 1.0  # meters; see grid_utils.heuristic_candidate_scores

# ─────────────────────────────────────────────────────────────────────────
# CONTROLLER BINDINGS -- VERIFY against `ros2 topic echo /joy` for your
# specific controller/driver before relying on these. Standard Linux
# xpad + joy_node Xbox mapping is assumed below but varies by kernel/driver
# version.
# ─────────────────────────────────────────────────────────────────────────
AXIS_LEFT_TRIGGER = 2     # range: 1.0 (released) -> -1.0 (fully pressed)
AXIS_RIGHT_TRIGGER = 5    # range: 1.0 (released) -> -1.0 (fully pressed)
TRIGGER_PRESS_THRESHOLD = 0.0  # press registers when axis value drops below this

BTN_A = 0       # advance to next candidate (skip, no label)
BTN_B = 1       # finish scan now (mark reviewed, jump to next scan)
BTN_X = 2       # undo last label
BTN_Y = 3       # mark whole scan clean (deny-all remaining candidates quickly)
BTN_RB = 5      # jump back one scan (for re-review)
BTN_START = 7   # force-flush ledger to disk (auto-flushes after every write anyway)

# RT = confirm current candidate as artifact
# LT = deny current candidate (hard negative)

# ─────────────────────────────────────────────────────────────────────────
# Colors (R, G, B) 0-255, used for the republished cloud
# ─────────────────────────────────────────────────────────────────────────
COLOR_NORMAL = (160, 160, 160)
COLOR_PENDING_CANDIDATE = (220, 30, 30)     # red: not yet reviewed this pass
COLOR_CURRENT_CANDIDATE = (60, 220, 220)    # cyan: the one awaiting your input
COLOR_CONFIRMED = (255, 140, 0)             # orange flash: just confirmed
COLOR_DENIED = (90, 90, 220)                # blue flash: just denied

REPUBLISH_RATE_HZ = 8.0  # keep the cloud "live" in Foxglove between inputs


def pack_rgb_float(r: int, g: int, b: int) -> float:
    """Packs r,g,b (0-255) into the float32 bit-pattern PCL/RViz/Foxglove
    expect for a standard 'rgb' PointXYZRGB field."""
    packed = (r << 16) | (g << 8) | b
    return struct.unpack("f", struct.pack("I", packed))[0]


class AnnotatorNode(Node):
    def __init__(self):
        super().__init__("lidar_annotator")

        # ---- ledger setup ----
        self._ledger_file = open(LEDGER_PATH, "a", buffering=1)  # line-buffered
        self._reviewed_file = open(REVIEWED_SCANS_PATH, "a", buffering=1)
        self._undo_stack = []  # list of dicts we can pop to undo

        # ---- bag setup ----
        self.get_logger().info(f"Loading bag: {BAG_PATH}")
        self._scans = self._load_bag_scans(BAG_PATH, INPUT_CLOUD_TOPIC)
        self.get_logger().info(f"Loaded {len(self._scans)} scans from bag.")
        if not self._scans:
            raise RuntimeError(
                f"No messages found on topic '{INPUT_CLOUD_TOPIC}' in bag "
                f"'{BAG_PATH}'. Double check BAG_PATH / INPUT_CLOUD_TOPIC."
            )

        # ---- per-scan working state ----
        self._scan_idx = 0
        self._current_scan_state = None  # populated by _load_scan()
        self._load_scan(self._scan_idx)

        # ---- joy edge-detection state ----
        self._prev_buttons = []
        self._prev_lt_pressed = False
        self._prev_rt_pressed = False

        # ---- ROS pub/sub ----
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._cloud_pub = self.create_publisher(PointCloud2, OUTPUT_CLOUD_TOPIC, qos)
        self.create_subscription(Joy, "/joy", self._on_joy, qos)

        self.create_timer(1.0 / REPUBLISH_RATE_HZ, self._republish_current_cloud)

        self.get_logger().info(
            "Ready. RT=confirm artifact, LT=deny, A=next candidate, "
            "B=finish scan, Y=mark scan clean, X=undo, RB=prev scan."
        )

    # ─────────────────────────────────────────────────────────────────
    # Bag loading
    # ─────────────────────────────────────────────────────────────────
    def _load_bag_scans(self, bag_path: str, topic: str) -> list:
        """
        Reads every PointCloud2 message on `topic` into memory as raw XYZ
        numpy arrays. Fine for low-res scans (~5k points each); if you
        later annotate much larger bags, switch this to a streaming /
        seek-based reader instead of loading everything up front.
        """
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}
        if topic not in type_map:
            available = ", ".join(type_map.keys())
            raise RuntimeError(
                f"Topic '{topic}' not found in bag. Available topics: {available}"
            )

        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        msg_type = get_message(type_map[topic])

        storage_filter = rosbag2_py.StorageFilter(topics=[topic])
        reader.set_filter(storage_filter)

        scans = []
        while reader.has_next():
            (read_topic, data, t) = reader.read_next()
            msg = deserialize_message(data, msg_type)
            xyz = self._extract_xyz(msg)
            scans.append({"stamp_ns": t, "xyz": xyz})
        return scans

    @staticmethod
    def _extract_xyz(cloud_msg: PointCloud2) -> np.ndarray:
        """Pulls an (N, 3) float64 XYZ array out of a PointCloud2, robust to
        field order / extra fields (intensity, ring, etc. all ignored)."""
        points = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float64)
        return xyz

    # ─────────────────────────────────────────────────────────────────
    # Per-scan setup
    # ─────────────────────────────────────────────────────────────────
    def _load_scan(self, idx: int):
        idx = max(0, min(idx, len(self._scans) - 1))
        self._scan_idx = idx
        scan = self._scans[idx]
        xyz = scan["xyz"]

        grid = compute_polar_grid(
            xyz, GRID_ROWS, GRID_COLS, ELEVATION_MIN_RAD, ELEVATION_MAX_RAD,
            azimuth_offset_rad=AZIMUTH_OFFSET_RAD,
        )
        range_img, point_idx_img = build_range_image(
            grid["row_idx"], grid["col_idx"], grid["ranges"], grid["orig_indices"],
            GRID_ROWS, GRID_COLS,
        )
        scores = heuristic_candidate_scores(range_img)
        candidates = get_candidate_cells(scores, HEURISTIC_THRESHOLD)

        self._current_scan_state = {
            "stamp_ns": scan["stamp_ns"],
            "xyz": xyz,
            "range_img": range_img,
            "point_idx_img": point_idx_img,
            "scores": scores,
            "candidates": candidates,          # list of (row, col), score-sorted
            "candidate_ptr": 0,
            "decided": {},                      # (row,col) -> "artifact"/"clean"
            "flash": None,                      # ((row,col), color, expiry_time) or None
        }
        self.get_logger().info(
            f"Scan {idx+1}/{len(self._scans)}  "
            f"({len(candidates)} candidates flagged)"
        )

    # ─────────────────────────────────────────────────────────────────
    # Joy handling
    # ─────────────────────────────────────────────────────────────────
    def _on_joy(self, msg: Joy):
        buttons = msg.buttons
        axes = msg.axes
        if not self._prev_buttons:
            self._prev_buttons = list(buttons)

        def button_pressed(idx):
            return (
                idx < len(buttons)
                and idx < len(self._prev_buttons)
                and buttons[idx] == 1
                and self._prev_buttons[idx] == 0
            )

        lt_pressed_now = len(axes) > AXIS_LEFT_TRIGGER and axes[AXIS_LEFT_TRIGGER] < TRIGGER_PRESS_THRESHOLD
        rt_pressed_now = len(axes) > AXIS_RIGHT_TRIGGER and axes[AXIS_RIGHT_TRIGGER] < TRIGGER_PRESS_THRESHOLD
        lt_edge = lt_pressed_now and not self._prev_lt_pressed
        rt_edge = rt_pressed_now and not self._prev_rt_pressed
        self._prev_lt_pressed = lt_pressed_now
        self._prev_rt_pressed = rt_pressed_now

        if rt_edge:
            self._label_current_candidate("artifact")
        elif lt_edge:
            self._label_current_candidate("clean")
        elif button_pressed(BTN_A):
            self._advance_candidate()
        elif button_pressed(BTN_B):
            self._finish_scan_and_advance()
        elif button_pressed(BTN_X):
            self._undo_last()
        elif button_pressed(BTN_Y):
            self._mark_scan_clean_and_advance()
        elif button_pressed(BTN_RB):
            self._load_scan(self._scan_idx - 1)

        self._prev_buttons = list(buttons)

    # ─────────────────────────────────────────────────────────────────
    # Labeling actions
    # ─────────────────────────────────────────────────────────────────
    def _current_candidate_cell(self):
        st = self._current_scan_state
        if st["candidate_ptr"] >= len(st["candidates"]):
            return None
        return st["candidates"][st["candidate_ptr"]]

    def _label_current_candidate(self, label: str):
        cell = self._current_candidate_cell()
        if cell is None:
            return
        st = self._current_scan_state
        row, col = cell

        record = {
            "bag": os.path.basename(BAG_PATH),
            "scan_stamp_ns": st["stamp_ns"],
            "ring": int(row),
            "azimuth_idx": int(col),
            "range_m": float(st["range_img"][row, col]),
            "heuristic_score": float(st["scores"][row, col]),
            "label": label,  # "artifact" or "clean"
            "session": SESSION_ID,
        }
        self._ledger_file.write(json.dumps(record) + "\n")
        self._ledger_file.flush()

        st["decided"][cell] = label
        st["flash"] = (cell, COLOR_CONFIRMED if label == "artifact" else COLOR_DENIED,
                        time.time() + 0.25)
        self._undo_stack.append(record)

        self.get_logger().info(f"  [{label}] ring={row} az_idx={col} range={record['range_m']:.2f}m")
        self._advance_candidate()

    def _advance_candidate(self):
        st = self._current_scan_state
        st["candidate_ptr"] += 1
        if st["candidate_ptr"] >= len(st["candidates"]):
            self._finish_scan_and_advance()

    def _undo_last(self):
        if not self._undo_stack:
            return
        record = self._undo_stack.pop()
        cell = (record["ring"], record["azimuth_idx"])
        st = self._current_scan_state
        if st["stamp_ns"] == record["scan_stamp_ns"] and cell in st["decided"]:
            del st["decided"][cell]
            # Move the pointer back to this candidate if it's in the current scan
            try:
                idx = st["candidates"].index(cell)
                st["candidate_ptr"] = idx
            except ValueError:
                pass
        self.get_logger().info(f"  [undo] {record['label']} ring={record['ring']} az_idx={record['azimuth_idx']}")
        # NOTE: this does not retract the line already written to the ledger
        # file. The build-script step (which turns the ledger into training
        # tensors) should take the LAST label for any duplicate (scan, ring,
        # azimuth_idx) key as authoritative, so an undo + relabel simply
        # appends a corrected line rather than needing in-place file edits.

    def _mark_scan_clean_and_advance(self):
        st = self._current_scan_state
        for cell in st["candidates"][st["candidate_ptr"]:]:
            row, col = cell
            if cell in st["decided"]:
                continue
            record = {
                "bag": os.path.basename(BAG_PATH),
                "scan_stamp_ns": st["stamp_ns"],
                "ring": int(row),
                "azimuth_idx": int(col),
                "range_m": float(st["range_img"][row, col]),
                "heuristic_score": float(st["scores"][row, col]),
                "label": "clean",
                "session": SESSION_ID,
            }
            self._ledger_file.write(json.dumps(record) + "\n")
            st["decided"][cell] = "clean"
        self._ledger_file.flush()
        self._finish_scan_and_advance()

    def _finish_scan_and_advance(self):
        st = self._current_scan_state
        n_confirmed = sum(1 for v in st["decided"].values() if v == "artifact")
        review_record = {
            "bag": os.path.basename(BAG_PATH),
            "scan_stamp_ns": st["stamp_ns"],
            "n_candidates": len(st["candidates"]),
            "n_confirmed": n_confirmed,
            "session": SESSION_ID,
        }
        self._reviewed_file.write(json.dumps(review_record) + "\n")
        self._reviewed_file.flush()
        self._load_scan(self._scan_idx + 1)

    # ─────────────────────────────────────────────────────────────────
    # Republishing the colorized cloud
    # ─────────────────────────────────────────────────────────────────
    def _republish_current_cloud(self):
        st = self._current_scan_state
        xyz = st["xyz"]
        n = xyz.shape[0]
        colors = np.full(n, pack_rgb_float(*COLOR_NORMAL), dtype=np.float32)

        current_cell = self._current_candidate_cell()
        flash = st["flash"]
        if flash is not None and time.time() > flash[2]:
            st["flash"] = None
            flash = None

        for ptr, cell in enumerate(st["candidates"]):
            row, col = cell
            point_idx = st["point_idx_img"][row, col]
            if point_idx < 0:
                continue
            if flash is not None and cell == flash[0]:
                colors[point_idx] = pack_rgb_float(*flash[1])
            elif cell == current_cell:
                colors[point_idx] = pack_rgb_float(*COLOR_CURRENT_CANDIDATE)
            elif cell not in st["decided"]:
                colors[point_idx] = pack_rgb_float(*COLOR_PENDING_CANDIDATE)
            # already-decided, non-current, non-flashing candidates fall
            # back to COLOR_NORMAL so the scan reads cleanly as you progress

        cloud_msg = self._build_xyzrgb_cloud(xyz, colors)
        self._cloud_pub.publish(cloud_msg)

    def _build_xyzrgb_cloud(self, xyz: np.ndarray, rgb_packed: np.ndarray) -> PointCloud2:
        header_stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = np.zeros(xyz.shape[0], dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)
        ])
        cloud_data["x"] = xyz[:, 0]
        cloud_data["y"] = xyz[:, 1]
        cloud_data["z"] = xyz[:, 2]
        cloud_data["rgb"] = rgb_packed

        from std_msgs.msg import Header
        header = Header()
        header.stamp = header_stamp
        header.frame_id = "lidar"  # <-- adjust to match your TF tree if needed

        return pc2.create_cloud(header, fields, cloud_data)

    def destroy_node(self):
        self._ledger_file.close()
        self._reviewed_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = AnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
