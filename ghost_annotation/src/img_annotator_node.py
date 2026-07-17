#!/usr/bin/env python3
# NOTE: use chmod +x to make this script executable, otherwise ROS
# complains that it can't be found!
"""
img_annotator_node.py

Joystick-driven offline annotation tool that:
  1. Reads a ROS2 bag offline (no live playback needed), indexing scans.
  2. Builds a FOV-scaled range image where each lidar layer occupies exactly
     one pixel row, and spacer rows fill the gaps to match sensor FOV aspect
     ratio.
  3. Moves a pixel-space cursor via the joystick left-stick. D-pad gives
     single-pixel precision movement. Vertical cursor movement skips between
     valid layer rows (never lands on a blank spacer).
  4. Simultaneously highlights the corresponding 3-D point in the published
     XYZRGB cloud and publishes a 3D sphere marker so both views stay in sync.
  5. Pressing A selects / deselects the current cell. RB advances to the next
     scan, and LB goes back. Holding either scrubs without writing.
  6. Y explicitly marks the current frame for ledger inclusion, even if it has
     no selections (an "errorless" training sample).
  7. Selections are stored in memory and written to a JSONL ledger on
     single-press navigation, but ONLY if the scan was modified (dirty) or
     explicitly included via Y. Held-button scrubbing never writes to the
     ledger. Existing selections are reloaded on startup.

Published image is 32FC1 float (normalised range, NaN for empty cells).
Image overlays (cursor, selected-point dots) are published as a single
visualization_msgs/ImageMarker POINTS message so the raw float image
remains untouched. In Foxglove, point the Image panel's "Annotations"
setting at IMAGE_ANNOTATION_TOPIC.

Ledger format: one JSONL line per scan save.
  {"bag": "...", "scan_stamp_ns": 123, "session": "...",
   "selections": [[ring, col, range_m], ...]}
Last record per scan_stamp_ns is authoritative (handles removals correctly).
"""

from __future__ import annotations

import json
import math
import os
import struct
from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, Joy, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import ImageMarker, Marker
from geometry_msgs.msg import Point as GeoPoint
import rosbag2_py
from rclpy.serialization import deserialize_message
from grid_utils import compute_polar_grid, build_range_image
from joy_state import XboxController as Xbox, JoyState, JoyButton, JoyAxis, JoyPov

# ─────────────────────────────────────────────────────────────────────────────
# USER-EDITABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CLOUD_TOPIC  = "/multiscan/lidar_scan"
OUTPUT_CLOUD_TOPIC = "/annotator/range_cloud"
OUTPUT_IMAGE_TOPIC = "/annotator/range_image"

IMAGE_ANNOTATION_TOPIC = "/annotator/image_annotations"
CURSOR_MARKER_TOPIC    = "/annotator/cursor_marker"

# Per-layer elevation angles for nearest-angle binning. Replace placeholder
# with output from calibrate_layer_elevations.py.
LAYER_ELEVATIONS_DEG = [
    42.2, 35.0, 29.0, 24.5, 19.0, 14.8, 10.0,
    5.5, 1.0, -4.0, -9.5, -14.0, -18.5, -22.2,
]  # <-- PLACEHOLDER. Replace with calibrate_layer_elevations.py output.
GRID_COLS = 360
AZIMUTH_OFFSET_DEG = 0.0

# ── Joystick tuning ───────────────────────────────────────────────────────────
CURSOR_SPEED_DEFAULT = 50
CURSOR_SPEED_MIN     = 1
CURSOR_SPEED_MAX     = 100

STICK_DEADZONE = 0.1
BUTTON_HELD_THRESH = 0.25

# ── Visual colours ─────────────────────────────────────────────────────────────
PC_DEFAULT_COLOR  = (175, 175, 175)
PC_SELECTED_COLOR = (255, 30,  30)

IMG_SELECTED_COLOR      = (1.0, 0.1, 0.1, 1.0)
IMG_CURSOR_COLOR        = (0.1, 0.9, 0.4, 1.0)
IMG_CURSOR_SELECT_COLOR = (0.9, 0.9, 0.1, 1.0)

PC_CURSOR_MARKER_RAD_M        = 0.025
PC_CURSOR_MARKER_RAD_MIN_M    = 0.005
PC_CURSOR_MARKER_RAD_MAX_M    = 0.5
PC_CURSOR_MARKER_COLOR        = (0.1, 0.9, 0.4, 1.0)
PC_CURSOR_SELECT_MARKER_COLOR = (0.9, 0.6, 0.1, 1.0)

REPUBLISH_HZ      = 10.0
JOY_REPUBLISH_HZ  = 30.0

# Single ledger file: one snapshot record per scan save.
# REVIEWED_SCANS_PATH is no longer needed; skipped/confirmed counts are
# embedded in each ledger record and derivable by downstream scripts.
LEDGER_PATH = "range_annotations_ledger.jsonl"

SKIP_JUMP_SIZE = 100
SESSION_ID = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pack_rgb_float(r: int, g: int, b: int) -> float:
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


def build_layer_row_map(
    layer_elevations_rad: np.ndarray,
    img_width: int,
) -> tuple[np.ndarray, int]:
    """
    Places each layer at a pixel row proportional to its actual elevation
    angle within the sensor FOV. Row 0 = top of image = highest elevation.
    Image height is set by the hFOV/vFOV aspect ratio.
    """
    sorted_layers = np.sort(layer_elevations_rad)[::-1]  # descending: row 0 = top
    total_fov    = sorted_layers[0] - sorted_layers[-1]
    aspect_ratio = math.pi * 2 / total_fov
    img_height   = round(img_width / aspect_ratio)

    el_max  = sorted_layers[0]
    el_min  = sorted_layers[-1]
    el_span = el_max - el_min

    layer_to_row = np.clip(
        np.round(
            (1.0 - (sorted_layers - el_min) / el_span) * (img_height - 1)
        ).astype(np.int32),
        0, img_height - 1,
    )
    return layer_to_row, img_height


def prompt_bag_path(default_path=None):
    import subprocess, shutil
    if shutil.which("zenity"):
        try:
            cmd = [
                "zenity", "--file-selection",
                "--title=Select ROS2 Bag File or Directory",
                "--file-filter=ROS2 Bags (*.mcap metadata.yaml) | *.mcap metadata.yaml",
                "--file-filter=All Files | *",
            ]
            path = subprocess.check_output(cmd, text=True).strip()
            if path:
                if os.path.basename(path) == "metadata.yaml":
                    path = os.path.dirname(path)
                return path
        except subprocess.CalledProcessError:
            pass

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select ROS2 Bag File or Directory",
            filetypes=[("ROS2 Bag", "*.mcap metadata.yaml"), ("All Files", "*.*")],
        )
        if path:
            if os.path.basename(path) == "metadata.yaml":
                path = os.path.dirname(path)
            return path
    except Exception:
        pass

    prompt_str = (
        f"Path to ROS2 bag [default: {default_path}]: "
        if default_path and os.path.exists(default_path)
        else "Path to ROS2 bag: "
    )
    try:
        user_input = input(prompt_str).strip()
        return user_input if user_input else default_path
    except (KeyboardInterrupt, EOFError):
        return default_path


# ─────────────────────────────────────────────────────────────────────────────
# Main Node
# ─────────────────────────────────────────────────────────────────────────────

class ImageAnnotatorNode(Node):

    class JoyControls:
        def __init__(self):
            self.joy_state = JoyState()

            self.cursor_h_axis      = JoyAxis(self.joy_state, Xbox.AXIS_LEFT_X)
            self.cursor_v_axis      = JoyAxis(self.joy_state, Xbox.AXIS_LEFT_Y)

            self.dpad_h_plus  = JoyPov(self.joy_state, Xbox.AXIS_DPAD_HORIZONTAL, Xbox.DPAD_RIGHT_VAL)
            self.dpad_h_minus = JoyPov(self.joy_state, Xbox.AXIS_DPAD_HORIZONTAL, Xbox.DPAD_LEFT_VAL)
            self.dpad_v_plus  = JoyPov(self.joy_state, Xbox.AXIS_DPAD_VERTICAL,   Xbox.DPAD_UP_VAL)
            self.dpad_v_minus = JoyPov(self.joy_state, Xbox.AXIS_DPAD_VERTICAL,   Xbox.DPAD_DOWN_VAL)

            self.cursor_smaller_axis = JoyAxis(self.joy_state, Xbox.AXIS_LEFT_TRIGGER)
            self.cursor_larger_axis  = JoyAxis(self.joy_state, Xbox.AXIS_RIGHT_TRIGGER)

            self.grad_base_axis     = JoyAxis(self.joy_state, Xbox.AXIS_RIGHT_X)
            self.grad_range_axis    = JoyAxis(self.joy_state, Xbox.AXIS_RIGHT_Y)

            self.select_btn       = JoyButton(self.joy_state, Xbox.BUTTON_A)
            self.select_btn2      = JoyButton(self.joy_state, Xbox.BUTTON_LEFT_STICK)
            self.clear_btn        = JoyButton(self.joy_state, Xbox.BUTTON_X)
            self.save_btn         = JoyButton(self.joy_state, Xbox.BUTTON_B)
            self.reset_grad_btn   = JoyButton(self.joy_state, Xbox.BUTTON_RIGHT_STICK)
            self.next_scan_btn    = JoyButton(self.joy_state, Xbox.BUTTON_RIGHT_BUMPER)
            self.prev_scan_btn    = JoyButton(self.joy_state, Xbox.BUTTON_LEFT_BUMPER)

        def update(self, joy: Joy):
            self.joy_state.update(joy)


    def __init__(self, bag_path: str = None):
        super().__init__("img_annotator")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("bag_path",             bag_path or "")
        self.declare_parameter("input_cloud_topic",    INPUT_CLOUD_TOPIC)
        self.declare_parameter("output_cloud_topic",   OUTPUT_CLOUD_TOPIC)
        self.declare_parameter("output_image_topic",   OUTPUT_IMAGE_TOPIC)
        self.declare_parameter("layer_elevations_deg", LAYER_ELEVATIONS_DEG)
        self.declare_parameter("grid_cols",            GRID_COLS)
        self.declare_parameter("azimuth_offset_deg",   AZIMUTH_OFFSET_DEG)
        self.declare_parameter("cursor_speed",         CURSOR_SPEED_DEFAULT)
        self.declare_parameter("scrub_speed",          SKIP_JUMP_SIZE)
        self.declare_parameter("ledger_path",          LEDGER_PATH)

        p = self.get_parameter
        self._bag_path             = p("bag_path").value
        self._cloud_topic          = p("input_cloud_topic").value
        self._out_cloud            = p("output_cloud_topic").value
        self._out_image            = p("output_image_topic").value
        self._layer_elevations_deg = list(p("layer_elevations_deg").value)
        self._layer_elevations_rad = np.radians(
            np.array(self._layer_elevations_deg, dtype=np.float64)
        )
        self._grid_cols            = int(p("grid_cols").value)
        self._azimuth_offset_deg   = p("azimuth_offset_deg").value
        self._scrub_speed          = int(p("scrub_speed").value)
        self._ledger_path          = p("ledger_path").value
        self._grid_rows            = len(self._layer_elevations_deg)

        self._cursor_speed: int = max(
            CURSOR_SPEED_MIN,
            min(CURSOR_SPEED_MAX, int(p("cursor_speed").value))
        )

        if not self._bag_path:
            self._bag_path = prompt_bag_path()
            if not self._bag_path:
                raise RuntimeError("No bag_path provided and prompt returned empty.")

        # ── Ledger ──────────────────────────────────────────────────────────
        self._load_ledger()
        self._ledger_file = open(self._ledger_path, "a", buffering=1)

        # ── Bag reader ──────────────────────────────────────────────────────
        self._frame_id = "lidar"
        self.get_logger().info(f"Indexing bag: {self._bag_path}")
        self._init_bag_reader(self._bag_path, self._cloud_topic)
        self.get_logger().info(f"Indexed {len(self._scan_timestamps)} scans from bag.")

        # ── Scan cache ──────────────────────────────────────────────────────
        self._scan_cache  = {}
        self._cache_order = []
        self._cache_limit = 100

        # ── Per-scan state ──────────────────────────────────────────────────
        self._xyz:           np.ndarray | None = None
        self._range_grid:    np.ndarray | None = None
        self._point_idx_img: np.ndarray | None = None
        self._num_cols:      int = 0

        self._layer_to_row:  np.ndarray | None = None
        self._img_height:    int = 0
        self._img_width:     int = 0

        self._selected:      set[tuple[int, int]] = set()
        self._cursor_x:      float = 0
        self._cursor_y:      float = 0
        self._cursor_layer:  int = 0
        self._cursor_col:    int = 0
        self._pt_cursor_rad: float = PC_CURSOR_MARKER_RAD_M

        # Dirty tracking: True when selections were modified during this visit
        # or the scan was explicitly force-included via Y.
        self._scan_dirty:    bool = False
        # Snapshot of selections at load time, used to detect changes.
        self._selected_at_load: set[tuple[int, int]] = set()

        self._scan_idx = 0
        self._load_scan(self._scan_idx)

        self._controls = ImageAnnotatorNode.JoyControls()
        self._dpad_hold_times: list[float] = []
        self._grad_base:  float = 0
        self._grad_range: float = 2.5

        self._last_publish_ns:      int = 0
        self._joy_republish_min_ns: int = int(1e9 / JOY_REPUBLISH_HZ)

        # ── QoS ────────────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self._cloud_pub     = self.create_publisher(PointCloud2, self._out_cloud,         qos)
        self._image_pub     = self.create_publisher(Image,       self._out_image,         qos)
        self._img_annot_pub = self.create_publisher(ImageMarker, IMAGE_ANNOTATION_TOPIC,  qos)
        self._marker_pub    = self.create_publisher(Marker,      CURSOR_MARKER_TOPIC,     qos)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Joy, "/joy", self._on_joy, qos)

        # ── Republish timer ─────────────────────────────────────────────────
        self.create_timer(1.0 / REPUBLISH_HZ, self._republish)

        self.get_logger().info(
            f"RangeAnnotator ready. "
            f"A/LS=Select, RB=Next, LB=Prev (hold to scrub), "
            f"X=Clear, B=Force include, B+RB/LB=Include+Nav. "
            f"Total scans: {len(self._scan_timestamps)}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Bag reading and caching
    # ─────────────────────────────────────────────────────────────────────────

    def _init_bag_reader(self, bag_path: str, topic: str):
        storage_options   = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        self._reader = rosbag2_py.SequentialReader()
        self._reader.open(storage_options, converter_options)

        type_map = {t.name: t.type for t in self._reader.get_all_topics_and_types()}
        if topic not in type_map:
            raise RuntimeError(
                f"Topic '{topic}' not found. Available: {', '.join(type_map)}"
            )

        from rosidl_runtime_py.utilities import get_message
        self._msg_type = get_message(type_map[topic])
        self._reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

        self._scan_timestamps = []
        while self._reader.has_next():
            (_, _, t) = self._reader.read_next()
            self._scan_timestamps.append(t)

    def _read_scan_from_bag(self, idx: int) -> dict:
        t = self._scan_timestamps[idx]
        self._reader.seek(t)
        if self._reader.has_next():
            (_, data, msg_t) = self._reader.read_next()
            return {"stamp_ns": msg_t, "msg": deserialize_message(data, self._msg_type)}
        raise RuntimeError(f"Failed to read scan at index {idx} (stamp_ns={t})")

    def _get_scan(self, idx: int) -> dict:
        if idx in self._scan_cache:
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._scan_cache[idx]
        scan = self._read_scan_from_bag(idx)
        self._scan_cache[idx] = scan
        self._cache_order.append(idx)
        if len(self._cache_order) > self._cache_limit:
            self._scan_cache.pop(self._cache_order.pop(0), None)
        return scan

    # ─────────────────────────────────────────────────────────────────────────
    # Scan loading and cloud processing
    # ─────────────────────────────────────────────────────────────────────────

    def _load_scan(self, idx: int):
        idx = max(0, min(idx, len(self._scan_timestamps) - 1))
        self._scan_idx = idx
        scan = self._get_scan(idx)
        msg  = scan["msg"]

        if hasattr(msg.header, "frame_id"):
            self._frame_id = msg.header.frame_id

        self._process_cloud(msg)
        self._selected = set(self._all_selections.get(scan["stamp_ns"], []))
        self._selected_at_load = set(self._selected)
        self._scan_dirty = False

        self.get_logger().info(
            f"Scan {idx + 1}/{len(self._scan_timestamps)} "
            f"(stamp_ns={scan['stamp_ns']}, {len(self._selected)} existing selections)"
        )

    def _process_cloud(self, msg: PointCloud2) -> None:
        pts = pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)

        xyz = np.column_stack([
            np.asarray(pts["x"], dtype=np.float64),
            np.asarray(pts["y"], dtype=np.float64),
            np.asarray(pts["z"], dtype=np.float64),
        ])

        grid = compute_polar_grid(
            xyz,
            self._layer_elevations_rad,
            self._grid_cols,
            azimuth_offset_rad=math.radians(self._azimuth_offset_deg),
        )
        range_grid, point_idx_img = build_range_image(
            grid["row_idx"], grid["col_idx"], grid["ranges"], grid["orig_indices"],
            grid["grid_rows"], self._grid_cols,
        )

        self._xyz           = xyz.astype(np.float32)
        self._range_grid    = range_grid
        self._point_idx_img = point_idx_img
        self._num_cols      = self._grid_cols
        self._grid_rows     = grid["grid_rows"]

        if self._grid_cols != self._img_width:
            self._img_width = self._grid_cols
            self._layer_to_row, self._img_height = build_layer_row_map(
                self._layer_elevations_rad, self._grid_cols,
            )
            self.get_logger().info(
                f"Image geometry: {self._grid_cols}x{self._img_height} px"
            )

        self._cursor_layer = min(self._cursor_layer, self._grid_rows - 1)
        self._cursor_col   = min(self._cursor_col,   self._num_cols  - 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Joystick input
    # ─────────────────────────────────────────────────────────────────────────

    def _on_joy(self, msg: Joy) -> None:
        self._controls.update(msg)

        t  = self._controls.joy_state.stamp
        dt = self._controls.joy_state.dt

        if not self._dpad_hold_times:
            self._dpad_hold_times = [t] * 4

        if self._layer_to_row is not None and self._img_height:
            self._cursor_x -= (
                self._controls.cursor_h_axis.deadzone_value(STICK_DEADZONE) *
                dt * self._cursor_speed
            )
            self._cursor_y += (
                self._controls.cursor_v_axis.deadzone_value(STICK_DEADZONE) *
                dt * self._cursor_speed
            )

            ROW_INCREMENT = self._img_height / self._grid_rows
            DPAD_BUTTONS = [
                (self._controls.dpad_h_minus, (-1,  0),              (-0.5,  0)),
                (self._controls.dpad_h_plus,  (+1,  0),              (+0.5,  0)),
                (self._controls.dpad_v_plus,  ( 0, +ROW_INCREMENT),  ( 0, +0.5)),
                (self._controls.dpad_v_minus, ( 0, -ROW_INCREMENT),  ( 0, -0.5)),
            ]

            for i, D in enumerate(DPAD_BUTTONS):
                if not D[0].raw_value():
                    self._dpad_hold_times[i] = t
                if D[0].was_pressed():
                    self._cursor_x += D[1][0]
                    self._cursor_y += D[1][1]
                if (t - self._dpad_hold_times[i]) > BUTTON_HELD_THRESH:
                    self._cursor_x += D[2][0] * dt * self._cursor_speed
                    self._cursor_y += D[2][1] * dt * self._cursor_speed

            self._cursor_x     = self._cursor_x % self._img_width
            self._cursor_y     = max(0, min(self._cursor_y, self._img_height - 1))
            self._cursor_col   = math.floor(self._cursor_x)
            self._cursor_layer = math.floor(self._cursor_y / ROW_INCREMENT)

            self._pt_cursor_rad += (
                (self._controls.cursor_larger_axis.raw_value() -
                 self._controls.cursor_smaller_axis.raw_value()) *
                (0.1 * dt)
            )
            self._pt_cursor_rad = max(
                PC_CURSOR_MARKER_RAD_MIN_M,
                min(self._pt_cursor_rad, PC_CURSOR_MARKER_RAD_MAX_M),
            )

            self._grad_base -= (
                self._controls.grad_base_axis.deadzone_value(STICK_DEADZONE) * dt
            )
            self._grad_range += (
                self._controls.grad_range_axis.deadzone_value(STICK_DEADZONE) * dt
            )
            self._grad_base  = max(0, self._grad_base)
            self._grad_range = max(1e-7, self._grad_range)

        if self._controls.reset_grad_btn.was_pressed():
            self._grad_base  = 0
            self._grad_range = 2.5

        if (self._controls.select_btn.was_pressed() or
                self._controls.select_btn2.was_pressed()):
            self._toggle_selection()

        elif self._controls.clear_btn.was_pressed():
            self._clear_selections()

        # B + RB/LB combo: force-include then navigate.
        # Checked before plain RB/LB so the held B doesn't get missed.
        elif (self._controls.save_btn.raw_value() and
                self._controls.next_scan_btn.was_pressed()):
            self._scan_dirty = True
            self._navigate_scan(1)
        elif (self._controls.save_btn.raw_value() and
                self._controls.prev_scan_btn.was_pressed()):
            self._scan_dirty = True
            self._navigate_scan(-1)

        elif self._controls.next_scan_btn.was_pressed():
            self._navigate_scan(1)
        elif self._controls.prev_scan_btn.was_pressed():
            self._navigate_scan(-1)

        elif self._controls.next_scan_btn.was_held(BUTTON_HELD_THRESH):
            self._scrub_scan(math.floor(self._scrub_speed * dt))
        elif self._controls.prev_scan_btn.was_held(BUTTON_HELD_THRESH):
            self._scrub_scan(math.floor(-self._scrub_speed * dt))

        # elif self._controls.save_btn.was_pressed():
        #     self._force_include_scan()

        # Re-publish outputs immediately if the rate window allows.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_publish_ns >= self._joy_republish_min_ns:
            self._do_republish()

    def _toggle_selection(self) -> None:
        cell = (self._cursor_layer, self._cursor_col)
        if cell in self._selected:
            self._selected.discard(cell)
            self.get_logger().info(f"Deselected: ring={cell[0]} col={cell[1]}")
        else:
            self._selected.add(cell)
            self.get_logger().info(f"Selected:   ring={cell[0]} col={cell[1]}")
        self._scan_dirty = True

    def _clear_selections(self) -> None:
        if self._selected:
            # Had selections that are now being removed — mark dirty so the
            # empty state is written to the ledger on navigate (overwriting
            # the old record).
            self._scan_dirty = True
        self._selected.clear()
        self.get_logger().info("Selections cleared.")

    def _is_scan_dirty(self) -> bool:
        """True if the current scan's selections differ from what was loaded."""
        return self._scan_dirty or self._selected != self._selected_at_load

    def _navigate_scan(self, delta: int):
        """Single-press navigation: saves to ledger only if dirty."""
        self._save_current_scan_to_memory()
        if self._is_scan_dirty():
            self._write_scan_to_ledger(self._scan_idx)
        new_idx = max(0, min(self._scan_idx + delta, len(self._scan_timestamps) - 1))
        self._load_scan(new_idx)

    def _scrub_scan(self, delta: int):
        """Held-button scrubbing: navigates without writing to ledger."""
        self._save_current_scan_to_memory()
        new_idx = max(0, min(self._scan_idx + delta, len(self._scan_timestamps) - 1))
        self._load_scan(new_idx)

    def _force_include_scan(self):
        """B button: explicitly include this scan in the ledger even if it has
        no selections (marking it as an errorless training sample). Also used
        as part of B+RB / B+LB combo to force-include before navigating."""
        self._scan_dirty = True
        self._save_current_scan_to_memory()
        self._write_scan_to_ledger(self._scan_idx)
        self.get_logger().info(
            f"Force-included scan {self._scan_idx + 1} "
            f"({len(self._selected)} selections)."
        )

    def _save_current_scan_to_memory(self):
        if not self._scan_timestamps:
            return
        stamp_ns = self._get_scan(self._scan_idx)["stamp_ns"]
        self._all_selections[stamp_ns] = set(self._selected)

    # ─────────────────────────────────────────────────────────────────────────
    # Ledger I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _write_scan_to_ledger(self, idx: int):
        """
        Writes a single snapshot record for the given scan capturing its
        complete, current selection state:

            {"bag": "...", "scan_stamp_ns": 123, "session": "...",
             "selections": [[ring, col, range_m], ...]}

        The last record for any scan_stamp_ns is authoritative on load, so
        removals are handled correctly: a later save with fewer selections
        simply supersedes earlier saves that included those points. No
        tombstones, no separate reviews file.
        """
        scan     = self._get_scan(idx)
        stamp_ns = scan["stamp_ns"]
        selections = self._all_selections.get(stamp_ns, set())

        sel_list = []
        for (ring, col) in sorted(selections):
            rng = None
            if self._range_grid is not None:
                v = float(self._range_grid[ring, col])
                if not math.isnan(v):
                    rng = round(v, 3)  # mm precision is sufficient
            sel_list.append([ring, col, rng])

        record = {
            "bag":           os.path.basename(self._bag_path),
            "scan_stamp_ns": stamp_ns,
            "session":       SESSION_ID,
            "selections":    sel_list,
        }
        self._ledger_file.write(json.dumps(record) + "\n")
        self._ledger_file.flush()

    def _load_ledger(self):
        """
        Loads selections from the ledger using last-write-wins per scan.

        New format: each line is a snapshot. The last record for a given
        scan_stamp_ns is the authoritative selection state, so deselections
        made in a later session are correctly reflected even though the file
        is append-only.

        Backward compat:
          - Old snapshot with dict items: {"selections": [{"ring":..., "col":...}]}
          - Old per-point format: {"label": "artifact", "ring":..., "azimuth_idx":...}
            These are accumulated as before, but any subsequent snapshot for the
            same stamp overwrites them entirely.
        """
        self._all_selections: dict[int, set] = {}
        if not os.path.exists(self._ledger_path):
            return

        self.get_logger().info(f"Loading selections from: {self._ledger_path}")
        # Tracks which stamps have been given a definitive snapshot record.
        # Once a stamp is snapshotted, old per-point accumulation is ignored
        # for that stamp regardless of record order.
        snapshotted: set[int] = set()

        try:
            with open(self._ledger_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data     = json.loads(line)
                        stamp_ns = data.get("scan_stamp_ns")
                        if stamp_ns is None:
                            continue

                        if "selections" in data:
                            # Snapshot format (new compact or old grouped dict).
                            # Last write wins: unconditionally overwrite.
                            sels = set()
                            for item in data["selections"]:
                                if isinstance(item, list) and len(item) >= 2:
                                    # New: [ring, col, range_m?]
                                    sels.add((int(item[0]), int(item[1])))
                            self._all_selections[stamp_ns] = sels
                            snapshotted.add(stamp_ns)

                        elif "ring" in data and "azimuth_idx" in data:
                            # Old per-point format. Only accumulate if no snapshot
                            # has appeared for this stamp (snapshot is authoritative).
                            if stamp_ns not in snapshotted:
                                ring = data.get("ring")
                                col  = data.get("azimuth_idx")
                                if ring is not None and col is not None:
                                    self._all_selections.setdefault(
                                        stamp_ns, set()
                                    ).add((int(ring), int(col)))

                    except Exception as e:
                        self.get_logger().warn(f"Skipped malformed ledger line: {e}")

        except Exception as e:
            self.get_logger().error(f"Error reading ledger: {e}")

        n_pts = sum(len(v) for v in self._all_selections.values())
        self.get_logger().info(
            f"Loaded {n_pts} selections across {len(self._all_selections)} scans "
            f"({len(snapshotted)} snapshot, "
            f"{len(self._all_selections) - len(snapshotted)} legacy per-point)."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Publishing
    # ─────────────────────────────────────────────────────────────────────────

    def _do_republish(self) -> None:
        """Publish all outputs and record the timestamp. Called both by the
        heartbeat timer and directly from _on_joy (rate-limited)."""
        if self._range_grid is None:
            return
        now = self.get_clock().now()
        self._last_publish_ns = now.nanoseconds
        stamp = now.to_msg()
        self._publish_cloud(stamp)
        self._publish_image(stamp)
        self._publish_image_annotations(stamp)
        self._publish_cursor_3d_marker(stamp)

    def _republish(self) -> None:
        """Heartbeat timer callback — fires at REPUBLISH_HZ to keep displays
        fresh even when no joystick input arrives."""
        self._do_republish()

    def _publish_image(self, stamp) -> None:
        if self._layer_to_row is None:
            return

        W = self._img_width
        H = self._img_height
        img = np.full((H, W), np.nan, dtype=np.float32)

        rg = self._range_grid
        for layer in range(self._grid_rows):
            px_row = int(self._layer_to_row[layer])
            if 0 <= px_row < H:
                row_data = rg[layer, :]
                norm     = np.clip(
                    (row_data - self._grad_base) / self._grad_range, 0, 1
                )
                img[px_row, :] = np.where(np.isfinite(row_data), norm, np.nan)

        img_msg = Image()
        img_msg.header.stamp    = stamp
        img_msg.header.frame_id = self._frame_id
        img_msg.height          = H
        img_msg.width           = W
        img_msg.encoding        = "32FC1"
        img_msg.is_bigendian    = 0
        img_msg.step            = W * 4
        img_msg.data            = img.tobytes()
        self._image_pub.publish(img_msg)

    def _publish_image_annotations(self, stamp) -> None:
        if self._layer_to_row is None:
            return

        marker = ImageMarker()
        marker.header.stamp    = stamp
        marker.header.frame_id = self._frame_id
        marker.ns     = "annotator"
        marker.id     = 0
        marker.type   = ImageMarker.POINTS
        marker.action = ImageMarker.ADD
        marker.scale  = 0.25

        def add_point(col: float, row: float, color: tuple):
            r, g, b, a = color
            marker.points.append(GeoPoint(x=col + 0.25, y=row + 0.25, z=0.0))
            marker.outline_colors.append(ColorRGBA(r=r, g=g, b=b, a=a))

        for (sel_layer, sel_col) in self._selected:
            if 0 <= sel_layer < self._grid_rows:
                sel_row = self._layer_to_row[sel_layer]
                add_point(sel_col,       sel_row,       IMG_SELECTED_COLOR)
                add_point(sel_col,       sel_row + 0.5, IMG_SELECTED_COLOR)
                add_point(sel_col + 0.5, sel_row,       IMG_SELECTED_COLOR)
                add_point(sel_col + 0.5, sel_row + 0.5, IMG_SELECTED_COLOR)

        if 0 <= self._cursor_layer < self._grid_rows:
            color   = (IMG_CURSOR_SELECT_COLOR
                       if (self._cursor_layer, self._cursor_col) in self._selected
                       else IMG_CURSOR_COLOR)
            sel_row = self._layer_to_row[self._cursor_layer]
            add_point(self._cursor_col - 0.5, sel_row - 0.5, color)
            add_point(self._cursor_col - 0.5, sel_row,       color)
            add_point(self._cursor_col - 0.5, sel_row + 0.5, color)
            add_point(self._cursor_col - 0.5, sel_row + 1.0, color)
            add_point(self._cursor_col,       sel_row - 0.5, color)
            add_point(self._cursor_col + 0.5, sel_row - 0.5, color)
            add_point(self._cursor_col + 1.0, sel_row - 0.5, color)
            add_point(self._cursor_col,       sel_row + 1.0, color)
            add_point(self._cursor_col + 0.5, sel_row + 1.0, color)
            add_point(self._cursor_col + 1.0, sel_row,       color)
            add_point(self._cursor_col + 1.0, sel_row + 0.5, color)
            add_point(self._cursor_col + 1.0, sel_row + 1.0, color)

        self._img_annot_pub.publish(marker)

    def _publish_cursor_3d_marker(self, stamp) -> None:
        """
        Publishes a sphere Marker in the 3D point cloud at the position of the
        point the image cursor is currently over. Useful for correlating the
        image cursor position with the 3D scene in Foxglove.
        """
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp    = stamp
        marker.ns     = "annotator_cursor"
        marker.id     = 0
        marker.type   = Marker.SPHERE

        cur_layer = self._cursor_layer
        cur_col   = self._cursor_col

        if (
            self._point_idx_img is None
            or self._xyz is None
            or cur_layer >= self._grid_rows
            or cur_col  >= self._num_cols
        ):
            marker.action = Marker.DELETE
            self._marker_pub.publish(marker)
            return

        idx = int(self._point_idx_img[cur_layer, cur_col])
        if idx < 0:
            marker.action = Marker.DELETE
            self._marker_pub.publish(marker)
            return

        x, y, z = self._xyz[idx]
        marker.action = Marker.ADD
        marker.pose.position.x    = float(x)
        marker.pose.position.y    = float(y)
        marker.pose.position.z    = float(z)
        marker.pose.orientation.w = 1.0
        d = self._pt_cursor_rad * 2.0
        marker.scale.x = marker.scale.y = marker.scale.z = d
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
            PC_CURSOR_SELECT_MARKER_COLOR
            if (cur_layer, cur_col) in self._selected
            else PC_CURSOR_MARKER_COLOR
        )
        self._marker_pub.publish(marker)

    def _publish_cloud(self, stamp) -> None:
        xyz = self._xyz
        if xyz is None:
            return

        N      = xyz.shape[0]
        colors = np.full(N, pack_rgb_float(*PC_DEFAULT_COLOR), dtype=np.float32)

        if self._point_idx_img is not None and self._range_grid is not None:
            flat_idx = self._point_idx_img[:self._grid_rows, :self._num_cols].ravel().astype(np.int32)
            flat_rng = self._range_grid[:self._grid_rows, :self._num_cols].ravel()
            valid    = (flat_idx >= 0) & np.isfinite(flat_rng)
            if valid.any():
                v = np.clip(
                    (flat_rng[valid] - self._grad_base) / self._grad_range, 0.0, 1.0
                )
                g      = (v * 255).astype(np.uint32)
                packed = (g << 16) | (g << 8) | g
                colors[flat_idx[valid]] = packed.astype(np.uint32).view(np.float32)

        sel_color = pack_rgb_float(*PC_SELECTED_COLOR)
        for (sel_layer, sel_col) in self._selected:
            if 0 <= sel_layer < self._grid_rows:
                idx = int(self._point_idx_img[sel_layer, sel_col])
                if idx >= 0:
                    colors[idx] = sel_color

        fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        dtype = np.dtype([("x", np.float32), ("y", np.float32),
                          ("z", np.float32), ("rgb", np.float32)])
        data        = np.zeros(N, dtype=dtype)
        data["x"]   = xyz[:, 0]
        data["y"]   = xyz[:, 1]
        data["z"]   = xyz[:, 2]
        data["rgb"] = colors

        header          = Header()
        header.stamp    = stamp
        header.frame_id = self._frame_id
        self._cloud_pub.publish(pc2.create_cloud(header, fields, data))

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        handle = getattr(self, "_ledger_file", None)
        if handle:
            handle.close()
        if hasattr(self, "_reader") and self._reader:
            self._reader.close()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    node = None
    try:
        node = ImageAnnotatorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError, FileNotFoundError) as e:
        print(f"RangeAnnotator exiting... {e}")
    finally:
        if node is not None:
            node.destroy_node()
        # rclpy.shutdown()


if __name__ == "__main__":
    main()
