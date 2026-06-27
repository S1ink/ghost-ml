#!/usr/bin/env python3
# NOTE: use chmod +x to make this scipt executable, otherwise ROS
# complains that it can't be found!
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
ELEVATION_MIN_DEG = -22.2
ELEVATION_MAX_DEG = 42.2
AZIMUTH_OFFSET_DEG = 0.0

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
# force-flush ledger to disk (auto-flushes after every write anyway)
BTN_START = 7

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


def prompt_bag_path(default_path=None):
    """
    Prompts the user for a bag path using a GUI dialog (zenity or tkinter) if available.
    Falls back to a CLI prompt if GUI tools are unavailable or fail.
    """
    import subprocess
    import shutil
    import os

    # 1. Try zenity (common on Linux/GTK environments)
    if shutil.which("zenity"):
        try:
            # We filter for .mcap files and metadata.yaml
            # If the user selects metadata.yaml, we'll use its directory.
            cmd = [
                "zenity",
                "--file-selection",
                "--title=Select ROS2 Bag File or Directory",
                "--file-filter=ROS2 Bags (*.mcap metadata.yaml) | *.mcap metadata.yaml",
                "--file-filter=All Files | *"
            ]
            path = subprocess.check_output(cmd, text=True).strip()
            if path:
                if os.path.basename(path) == "metadata.yaml":
                    path = os.path.dirname(path)
                return path
        except subprocess.CalledProcessError:
            # Zenity failed or user cancelled
            pass

    # 2. Try tkinter as a fallback
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select ROS2 Bag File or Directory",
            filetypes=[("ROS2 Bag", "*.mcap metadata.yaml"),
                       ("All Files", "*.*")]
        )
        if path:
            if os.path.basename(path) == "metadata.yaml":
                path = os.path.dirname(path)
            return path
    except Exception:
        pass

    # 3. Fallback to command line prompt
    print("\nCould not open file dialogue or dialogue cancelled.")
    if default_path and os.path.exists(default_path):
        prompt_str = f"Please enter the path to the ROS2 bag [default: {default_path}]: "
    else:
        prompt_str = "Please enter the path to the ROS2 bag: "

    try:
        user_input = input(prompt_str).strip()
        if not user_input and default_path:
            return default_path
        return user_input
    except (KeyboardInterrupt, EOFError):
        return default_path


class AnnotatorNode(Node):
    def __init__(self, bag_path: str = None):
        super().__init__("lidar_annotator")

        # Declare parameters with defaults from module-level constants
        self.declare_parameter('bag_path', bag_path or '')
        self.declare_parameter('input_cloud_topic', INPUT_CLOUD_TOPIC)
        self.declare_parameter('output_cloud_topic', OUTPUT_CLOUD_TOPIC)
        self.declare_parameter('ledger_path', LEDGER_PATH)
        self.declare_parameter('reviewed_scans_path', REVIEWED_SCANS_PATH)
        self.declare_parameter('grid_rows', GRID_ROWS)
        self.declare_parameter('grid_cols', GRID_COLS)
        self.declare_parameter('elevation_min_deg', ELEVATION_MIN_DEG)
        self.declare_parameter('elevation_max_deg', ELEVATION_MAX_DEG)
        self.declare_parameter('azimuth_offset_deg', AZIMUTH_OFFSET_DEG)
        self.declare_parameter('heuristic_threshold', HEURISTIC_THRESHOLD)

        # Retrieve parameter values
        self._bag_path = self.get_parameter('bag_path').get_parameter_value().string_value
        if not self._bag_path:
            self._bag_path = prompt_bag_path()
            if not self._bag_path:
                raise RuntimeError("No bag_path parameter provided and prompt returned empty.")

        # Resolve paths/topics
        self._input_cloud_topic = self.get_parameter('input_cloud_topic').get_parameter_value().string_value
        self._output_cloud_topic = self.get_parameter('output_cloud_topic').get_parameter_value().string_value
        self._ledger_path = self.get_parameter('ledger_path').get_parameter_value().string_value
        self._reviewed_scans_path = self.get_parameter('reviewed_scans_path').get_parameter_value().string_value
        
        self._grid_rows = self.get_parameter('grid_rows').get_parameter_value().integer_value
        self._grid_cols = self.get_parameter('grid_cols').get_parameter_value().integer_value
        self._elevation_min_deg = self.get_parameter('elevation_min_deg').get_parameter_value().double_value
        self._elevation_max_deg = self.get_parameter('elevation_max_deg').get_parameter_value().double_value
        self._azimuth_offset_deg = self.get_parameter('azimuth_offset_deg').get_parameter_value().double_value
        self._heuristic_threshold = self.get_parameter('heuristic_threshold').get_parameter_value().double_value

        self._elevation_min_rad = math.radians(self._elevation_min_deg)
        self._elevation_max_rad = math.radians(self._elevation_max_deg)
        self._azimuth_offset_rad = math.radians(self._azimuth_offset_deg)

        # ---- ledger setup ----
        self._ledger_file = open(self._ledger_path, "a", buffering=1)  # line-buffered
        self._reviewed_file = open(self._reviewed_scans_path, "a", buffering=1)
        self._undo_stack = []  # list of dicts we can pop to undo

        # ---- bag setup ----
        self._frame_id = "lidar"  # fallback default
        self.get_logger().info(f"Indexing bag: {self._bag_path}")
        self._init_bag_reader(self._bag_path, self._input_cloud_topic)
        self.get_logger().info(
            f"Indexed {len(self._scan_timestamps)} scans from bag.")

        # ---- cache setup ----
        self._scan_cache = {}
        self._cache_order = []
        self._cache_limit = 20

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
        self._cloud_pub = self.create_publisher(
            PointCloud2, self._output_cloud_topic, qos)
        self.create_subscription(Joy, "/joy", self._on_joy, qos)

        self.create_timer(1.0 / REPUBLISH_RATE_HZ,
                          self._republish_current_cloud)

        self.get_logger().info(
            "Ready. RT=confirm artifact, LT=deny, A=next candidate, "
            "B=finish scan, Y=mark scan clean, X=undo, RB=prev scan."
        )

    # ─────────────────────────────────────────────────────────────────
    # Bag loading and caching
    # ─────────────────────────────────────────────────────────────────
    def _init_bag_reader(self, bag_path: str, topic: str):
        """
        Opens the bag and scans the timestamps for the selected topic
        without deserializing the actual PointCloud2 payloads, keeping startup fast.
        """
        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        self._reader = rosbag2_py.SequentialReader()
        self._reader.open(storage_options, converter_options)

        topic_types = self._reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}
        if topic not in type_map:
            available = ", ".join(type_map.keys())
            raise RuntimeError(
                f"Topic '{topic}' not found in bag. Available topics: {available}"
            )

        from rosidl_runtime_py.utilities import get_message
        self._msg_type = get_message(type_map[topic])

        storage_filter = rosbag2_py.StorageFilter(topics=[topic])
        self._reader.set_filter(storage_filter)

        self._scan_timestamps = []
        while self._reader.has_next():
            (read_topic, data, t) = self._reader.read_next()
            self._scan_timestamps.append(t)

    def _read_scan_from_bag(self, idx: int) -> dict:
        """
        Seeks to the timestamp of the requested scan and deserializes it on demand.
        """
        t = self._scan_timestamps[idx]
        self._reader.seek(t)
        if self._reader.has_next():
            (read_topic, data, msg_t) = self._reader.read_next()
            from rclpy.serialization import deserialize_message
            msg = deserialize_message(data, self._msg_type)
            if hasattr(msg.header, 'frame_id'):
                self._frame_id = msg.header.frame_id
            xyz = self._extract_xyz(msg)
            return {"stamp_ns": msg_t, "xyz": xyz}
        raise RuntimeError(f"Failed to read scan at index {idx} with stamp {t}")

    def _get_scan(self, idx: int) -> dict:
        """
        Maintains a small cache of loaded scans to keep navigation snappy
        without consuming excessive memory.
        """
        if idx in self._scan_cache:
            # Move to end of cache_order (most recently used)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._scan_cache[idx]

        # Load from bag
        scan = self._read_scan_from_bag(idx)

        # Add to cache
        self._scan_cache[idx] = scan
        self._cache_order.append(idx)
        if len(self._cache_order) > self._cache_limit:
            oldest = self._cache_order.pop(0)
            self._scan_cache.pop(oldest, None)

        return scan

    @staticmethod
    def _extract_xyz(cloud_msg: PointCloud2) -> np.ndarray:
        """Pulls an (N, 3) float64 XYZ array out of a PointCloud2, robust to
        field order / extra fields (intensity, ring, etc. all ignored)."""
        points = pc2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.column_stack(
            [points["x"], points["y"], points["z"]]).astype(np.float64)
        return xyz

    # ─────────────────────────────────────────────────────────────────
    # Per-scan setup
    # ─────────────────────────────────────────────────────────────────
    def _load_scan(self, idx: int):
        idx = max(0, min(idx, len(self._scan_timestamps) - 1))
        self._scan_idx = idx
        scan = self._get_scan(idx)
        xyz = scan["xyz"]

        grid = compute_polar_grid(
            xyz, self._grid_rows, self._grid_cols, self._elevation_min_rad, self._elevation_max_rad,
            azimuth_offset_rad=self._azimuth_offset_rad,
        )
        range_img, point_idx_img = build_range_image(
            grid["row_idx"],
            grid["col_idx"],
            grid["ranges"],
            grid["orig_indices"],
            self._grid_rows, self._grid_cols,
        )
        scores = heuristic_candidate_scores(range_img)
        candidates = get_candidate_cells(scores, self._heuristic_threshold)

        self._current_scan_state = {
            "stamp_ns": scan["stamp_ns"],
            "xyz": xyz,
            "range_img": range_img,
            "point_idx_img": point_idx_img,
            "scores": scores,
            # list of (row, col), score-sorted
            "candidates": candidates,
            "candidate_ptr": 0,
            # (row,col) -> "artifact"/"clean"
            "decided": {},
            # ((row,col), color, expiry_time) or None
            "flash": None,
        }
        self.get_logger().info(
            f"Scan {idx + 1}/{len(self._scan_timestamps)}  "
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

        lt_pressed_now = (len(axes) > AXIS_LEFT_TRIGGER and
                          axes[AXIS_LEFT_TRIGGER] < TRIGGER_PRESS_THRESHOLD)
        rt_pressed_now = (len(axes) > AXIS_RIGHT_TRIGGER and
                          axes[AXIS_RIGHT_TRIGGER] < TRIGGER_PRESS_THRESHOLD)
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
            "bag": os.path.basename(self._bag_path),
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

        self.get_logger().info(
            f"  [{label}] ring={row} az_idx={col} range={record['range_m']:.2f}m")
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
        self.get_logger().info(
            f"  [undo] {record['label']} ring={record['ring']} az_idx={record['azimuth_idx']}")
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
                "bag": os.path.basename(self._bag_path),
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
            "bag": os.path.basename(self._bag_path),
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
            PointField(name="x", offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = np.zeros(xyz.shape[0], dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.float32)
        ])
        cloud_data["x"] = xyz[:, 0]
        cloud_data["y"] = xyz[:, 1]
        cloud_data["z"] = xyz[:, 2]
        cloud_data["rgb"] = rgb_packed

        from std_msgs.msg import Header
        header = Header()
        header.stamp = header_stamp
        header.frame_id = self._frame_id

        return pc2.create_cloud(header, fields, cloud_data)

    def destroy_node(self):
        self._ledger_file.close()
        self._reviewed_file.close()
        if hasattr(self, '_reader'):
            self._reader.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = AnnotatorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError, FileNotFoundError) as e:
        print(f"Annotator exiting: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
