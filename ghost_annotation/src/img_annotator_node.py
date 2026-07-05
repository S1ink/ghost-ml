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
  5. Pressing A selects / deselects the current cell. B advances to the next
     scan, and RB goes back.
  6. Selections are stored in memory and written to a JSONL ledger on
     navigation (one line per selected point, aligned with the previous
     annotator tool format). Existing selections are reloaded on startup.

Published image is 32FC1 float (raw range in metres, NaN for empty cells).
Image overlays (cursor crosshair, selected-point dots) are published as a
single visualization_msgs/ImageMarker POINTS message so the raw float image
remains untouched. In Foxglove, point the Image panel's "Annotations" setting
at IMAGE_ANNOTATION_TOPIC.

Controller bindings (standard xpad / joy_node Xbox layout):
  Left stick H/V (axes 0, 1) -- move cursor (continuous, speed-scaled)
  D-pad H/V (axes 6, 7)      -- move cursor one pixel at a time
  A  (button 0)              -- select / deselect cell under cursor
  B  (button 1)              -- save current selections and advance to next scan
  X  (button 2)              -- clear selections for the current scan
  Y  (button 3)              -- (reserved / unused)
  LB (button 4)              -- fast-forward (skip SKIP_JUMP_SIZE scans)
  RB (button 5)              -- save current selections and go back one scan
  Start (button 7)           -- force-save current scan selections to ledger

Point cloud coloring:
  Default:  white
  Cursor:   cyan
  Selected: amber
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

# ─────────────────────────────────────────────────────────────────────────────
# USER-EDITABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CLOUD_TOPIC  = "/multiscan/lidar_scan"
OUTPUT_CLOUD_TOPIC = "/annotator/range_cloud"
OUTPUT_IMAGE_TOPIC = "/annotator/range_image"

# Image annotation overlays (cursor dot + selection dots) are published here
# as a single visualization_msgs/ImageMarker POINTS message. In Foxglove,
# point your Image panel's "Annotations" setting at this topic.
IMAGE_ANNOTATION_TOPIC = "/annotator/image_annotations"

# 3D sphere marker showing which point the image cursor is currently on.
# Add this topic to your Foxglove 3D panel as a Marker display.
CURSOR_MARKER_TOPIC = "/annotator/cursor_marker"

# Lidar geometry -- used for image aspect-ratio calculation only.
# FOV_UP_DEG + FOV_DOWN_DEG = total vertical FOV.
FOV_UP_DEG   = 41.891   # highest layer elevation (degrees above horizontal)
FOV_DOWN_DEG = 22.401   # lowest layer elevation magnitude (degrees below)

# Per-layer elevation angles for nearest-angle binning. Replace placeholder
# with output from calibrate_layer_elevations.py (run against a raw bag that
# still has ring data). Order doesn't matter; sorted internally.
LAYER_ELEVATIONS_DEG = [
    42.2, 35.0, 29.0, 24.5, 19.0, 14.8, 10.0,
    5.5, 1.0, -4.0, -9.5, -14.0, -18.5, -22.2,
]  # <-- PLACEHOLDER. Replace with calibrate_layer_elevations.py output.
GRID_COLS = 360

AZIMUTH_OFFSET_DEG = 0.0

# ── Joystick tuning ───────────────────────────────────────────────────────────
# Continuous stick movement speed. 1 = one column per joy tick at full
# deflection. Keep this low -- the d-pad handles precise single-step movement.
CURSOR_SPEED_DEFAULT = 1
CURSOR_SPEED_MIN     = 1
CURSOR_SPEED_MAX     = 20

AXIS_STICK_H  = 0   # left-stick horizontal (left = negative)
AXIS_STICK_V  = 1   # left-stick vertical   (up = positive on most drivers)
STICK_DEADZONE = 0.15

# D-pad axes (typical Linux xpad / joy_node mapping).
# Each d-pad press moves the cursor exactly one pixel.
# Axis 6 horizontal: left = +1.0, right = -1.0
# Axis 7 vertical:   up   = +1.0, down  = -1.0
AXIS_DPAD_H = 6
AXIS_DPAD_V = 7
DPAD_AXIS_THRESHOLD = 0.5   # axis magnitude needed to register a press

# Set True if the image appears vertically flipped relative to stick input.
# This inverts the vertical direction for both the stick and the d-pad.
CURSOR_INVERT_V = True

# ── Visual colours ─────────────────────────────────────────────────────────────
# 3D point cloud (R, G, B, 0-255).
COLOR_DEFAULT_CLOUD = (175, 175, 175)
COLOR_CURSOR_CLOUD  = (0,   220, 220)   # cyan
COLOR_SELECT_CLOUD  = (255, 30,  30)   # amber

# Image annotation markers (R, G, B, 0-255).
COLOR_CURSOR_IMG = (0,   220, 220)      # cyan
COLOR_SELECT_IMG = (255, 30,  30)      # amber

# 3D sphere marker for cursor point.
MARKER_RADIUS_M = 0.045
MARKER_COLOR    = (0.0, 1.0, 1.0, 0.9)  # (r, g, b, a) 0-1
MARKER_SELECTED_COLOR = (1.0, 0.1, 0.1, 0.9)

REPUBLISH_HZ = 10.0

LEDGER_PATH         = "range_annotations_ledger.jsonl"
REVIEWED_SCANS_PATH = "range_annotations_reviewed.jsonl"
SKIP_JUMP_SIZE      = 100
SESSION_ID = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")

# Controller buttons.
BTN_SELECT = 0   # A
BTN_NEXT   = 1   # B
BTN_CLEAR  = 2   # X
BTN_SKIP   = 4   # LB
BTN_PREV   = 5   # RB
BTN_SAVE   = 7   # Start


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pack_rgb_float(r: int, g: int, b: int) -> float:
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


def build_layer_row_map(
    layer_elevations_rad: np.ndarray,
    fov_up_deg: float,
    fov_down_deg: float,
    img_width: int,
) -> tuple[np.ndarray, int]:
    """
    Places each layer at a pixel row proportional to its actual elevation
    angle within the sensor FOV, rather than at uniform intervals.
    Image height is set by the hFOV/vFOV aspect ratio (360 deg / total vFOV).
    Row 0 = top of image = highest elevation.
    """
    total_fov_deg = fov_up_deg + fov_down_deg
    aspect_ratio  = 360.0 / total_fov_deg
    img_height    = round(img_width / aspect_ratio)

    el_max  = math.radians(fov_up_deg)
    el_min  = math.radians(-fov_down_deg)
    el_span = el_max - el_min

    # Sort descending so index 0 = highest elevation = row 0 (top).
    sorted_layers = np.sort(layer_elevations_rad)[::-1]

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

    def __init__(self, bag_path: str = None):
        super().__init__("img_annotator")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("bag_path",             bag_path or "")
        self.declare_parameter("input_cloud_topic",    INPUT_CLOUD_TOPIC)
        self.declare_parameter("output_cloud_topic",   OUTPUT_CLOUD_TOPIC)
        self.declare_parameter("output_image_topic",   OUTPUT_IMAGE_TOPIC)
        self.declare_parameter("layer_elevations_deg", LAYER_ELEVATIONS_DEG)
        self.declare_parameter("grid_cols",            GRID_COLS)
        self.declare_parameter("fov_up_deg",           FOV_UP_DEG)
        self.declare_parameter("fov_down_deg",         FOV_DOWN_DEG)
        self.declare_parameter("azimuth_offset_deg",   AZIMUTH_OFFSET_DEG)
        self.declare_parameter("cursor_speed",         CURSOR_SPEED_DEFAULT)
        self.declare_parameter("ledger_path",          LEDGER_PATH)
        self.declare_parameter("reviewed_scans_path",  REVIEWED_SCANS_PATH)
        self.declare_parameter("skip_jump_size",       SKIP_JUMP_SIZE)

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
        self._fov_up_deg           = p("fov_up_deg").value
        self._fov_down_deg         = p("fov_down_deg").value
        self._azimuth_offset_deg   = p("azimuth_offset_deg").value
        self._ledger_path          = p("ledger_path").value
        self._reviewed_scans_path  = p("reviewed_scans_path").value
        self._skip_jump_size       = int(p("skip_jump_size").value)
        self._grid_rows            = len(self._layer_elevations_deg)

        self._cursor_speed: int = max(
            CURSOR_SPEED_MIN, min(CURSOR_SPEED_MAX, int(p("cursor_speed").value))
        )

        if not self._bag_path:
            self._bag_path = prompt_bag_path()
            if not self._bag_path:
                raise RuntimeError("No bag_path provided and prompt returned empty.")

        # ── Ledger ──────────────────────────────────────────────────────────
        self._load_ledger()
        self._ledger_file   = open(self._ledger_path,         "a", buffering=1)
        self._reviewed_file = open(self._reviewed_scans_path, "a", buffering=1)

        # ── Bag reader ──────────────────────────────────────────────────────
        self._frame_id = "lidar"
        self.get_logger().info(f"Indexing bag: {self._bag_path}")
        self._init_bag_reader(self._bag_path, self._cloud_topic)
        self.get_logger().info(f"Indexed {len(self._scan_timestamps)} scans from bag.")

        # ── Scan cache ──────────────────────────────────────────────────────
        self._scan_cache  = {}
        self._cache_order = []
        self._cache_limit = 20

        # ── Per-scan state ──────────────────────────────────────────────────
        self._xyz:           np.ndarray | None = None
        self._ranges_flat:   np.ndarray | None = None
        self._range_grid:    np.ndarray | None = None
        self._point_idx_img: np.ndarray | None = None
        self._num_cols:      int = 0

        self._layer_to_row:  np.ndarray | None = None
        self._img_height:    int = 0
        self._img_width:     int = 0

        self._selected:     set[tuple[int, int]] = set()
        self._cursor_layer: int = 0
        self._cursor_col:   int = 0
        self._col_accum:    float = 0.0

        self._scan_idx = 0
        self._load_scan(self._scan_idx)

        # Input edge-detection state.
        self._prev_buttons: list[int]   = []
        self._prev_axes:    list[float] = []

        # ── QoS ────────────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self._cloud_pub     = self.create_publisher(PointCloud2, self._out_cloud,            qos)
        self._image_pub     = self.create_publisher(Image,       self._out_image,            qos)
        self._img_annot_pub = self.create_publisher(ImageMarker, IMAGE_ANNOTATION_TOPIC,     qos)
        self._marker_pub    = self.create_publisher(Marker,      CURSOR_MARKER_TOPIC,        qos)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Joy, "/joy", self._on_joy, qos)

        # ── Republish timer ─────────────────────────────────────────────────
        self.create_timer(1.0 / REPUBLISH_HZ, self._republish)

        self.get_logger().info(
            f"RangeAnnotator ready. "
            f"A=Select/Deselect, B=Next, RB=Prev, LB=Skip+{self._skip_jump_size}, "
            f"X=Clear, Start=Force save. Total scans: {len(self._scan_timestamps)}"
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
        self._ranges_flat   = grid["ranges"].astype(np.float32)
        self._range_grid    = range_grid
        self._point_idx_img = point_idx_img
        self._num_cols      = self._grid_cols
        self._grid_rows     = grid["grid_rows"]

        if self._grid_cols != self._img_width:
            self._img_width = self._grid_cols
            self._layer_to_row, self._img_height = build_layer_row_map(
                self._layer_elevations_rad,
                self._fov_up_deg, self._fov_down_deg,
                self._grid_cols,
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
        buttons = list(msg.buttons)
        axes    = list(msg.axes)

        if not self._prev_buttons:
            self._prev_buttons = list(buttons)
        if not self._prev_axes:
            self._prev_axes = list(axes)

        # ── Button edge detection ───────────────────────────────────────────
        def btn_pressed(idx: int) -> bool:
            return (
                idx < len(buttons)
                and idx < len(self._prev_buttons)
                and buttons[idx] == 1
                and self._prev_buttons[idx] == 0
            )

        # ── D-pad edge detection (axes that go 0 -> ±1 on press) ───────────
        def dpad_pressed(axis_idx: int, positive: bool) -> bool:
            """True on the first tick an axis crosses the threshold."""
            if axis_idx >= len(axes) or axis_idx >= len(self._prev_axes):
                return False
            cur = axes[axis_idx]
            prv = self._prev_axes[axis_idx]
            if positive:
                return cur >  DPAD_AXIS_THRESHOLD and prv <=  DPAD_AXIS_THRESHOLD
            else:
                return cur < -DPAD_AXIS_THRESHOLD and prv >= -DPAD_AXIS_THRESHOLD

        # ── Continuous stick movement ───────────────────────────────────────
        ax_h = axes[AXIS_STICK_H] if len(axes) > AXIS_STICK_H else 0.0
        ax_v = axes[AXIS_STICK_V] if len(axes) > AXIS_STICK_V else 0.0

        if abs(ax_h) < STICK_DEADZONE:
            ax_h = 0.0
        if abs(ax_v) < STICK_DEADZONE:
            ax_v = 0.0

        if ax_h != 0.0 and self._num_cols > 0:
            self._col_accum += -ax_h * self._cursor_speed
            col_delta        = int(self._col_accum)
            self._col_accum -= col_delta
            if col_delta != 0:
                self._cursor_col = (self._cursor_col + col_delta) % self._num_cols

        if ax_v != 0.0:
            # CURSOR_INVERT_V flips so stick-up moves toward the top of the
            # displayed image. Adjust if still inverted on your setup.
            v_sign = 1 if CURSOR_INVERT_V else -1
            self._step_layer(v_sign if ax_v > 0.0 else -v_sign)

        # ── D-pad: single-pixel steps ───────────────────────────────────────
        # Axis 6: left = +1, right = -1 (typical xpad). Left → col - 1 (moves
        # left in azimuth). Adjust AXIS_DPAD_H polarity if inverted on your
        # driver.
        if dpad_pressed(AXIS_DPAD_H, positive=True) and self._num_cols > 0:
            self._cursor_col = (self._cursor_col - 1) % self._num_cols
        elif dpad_pressed(AXIS_DPAD_H, positive=False) and self._num_cols > 0:
            self._cursor_col = (self._cursor_col + 1) % self._num_cols

        # Axis 7: up = +1, down = -1. Uses the same vertical inversion as the
        # stick so the d-pad and stick feel consistent.
        v_sign = 1 if CURSOR_INVERT_V else -1
        if dpad_pressed(AXIS_DPAD_V, positive=True):
            self._step_layer(v_sign)
        elif dpad_pressed(AXIS_DPAD_V, positive=False):
            self._step_layer(-v_sign)

        # ── Button actions ──────────────────────────────────────────────────
        if btn_pressed(BTN_SELECT):
            self._toggle_selection()
        elif btn_pressed(BTN_CLEAR):
            self._clear_selections()
        elif btn_pressed(BTN_NEXT):
            self._navigate_scan(1, skipped=False)
        elif btn_pressed(BTN_PREV):
            self._navigate_scan(-1, skipped=False)
        elif btn_pressed(BTN_SKIP):
            self._navigate_scan(self._skip_jump_size, skipped=True)
        elif btn_pressed(BTN_SAVE):
            self._save_current_scan_to_memory()
            self._write_scan_to_ledger(self._scan_idx, skipped=False)
            self.get_logger().info(f"Force-saved scan {self._scan_idx + 1}.")

        self._prev_buttons = buttons
        self._prev_axes    = axes

    def _step_layer(self, direction: int) -> None:
        self._cursor_layer = max(0, min(self._grid_rows - 1, self._cursor_layer + direction))

    def _toggle_selection(self) -> None:
        cell = (self._cursor_layer, self._cursor_col)
        if cell in self._selected:
            self._selected.discard(cell)
            self.get_logger().info(f"Deselected: ring={cell[0]} col={cell[1]}")
        else:
            self._selected.add(cell)
            self.get_logger().info(f"Selected:   ring={cell[0]} col={cell[1]}")

    def _clear_selections(self) -> None:
        self._selected.clear()
        self.get_logger().info("Selections cleared.")

    def _navigate_scan(self, delta: int, skipped: bool = False):
        self._save_current_scan_to_memory()
        self._write_scan_to_ledger(self._scan_idx, skipped=skipped)
        new_idx = max(0, min(self._scan_idx + delta, len(self._scan_timestamps) - 1))
        self._load_scan(new_idx)

    def _save_current_scan_to_memory(self):
        if not self._scan_timestamps:
            return
        stamp_ns = self._get_scan(self._scan_idx)["stamp_ns"]
        self._all_selections[stamp_ns] = set(self._selected)

    # ─────────────────────────────────────────────────────────────────────────
    # Ledger I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _write_scan_to_ledger(self, idx: int, skipped: bool = False):
        scan     = self._get_scan(idx)
        stamp_ns = scan["stamp_ns"]
        selections = self._all_selections.get(stamp_ns, set())

        for (ring, col) in sorted(selections):
            rng = None
            if self._range_grid is not None:
                v = float(self._range_grid[ring, col])
                if not math.isnan(v):
                    rng = v
            record = {
                "bag":                 os.path.basename(self._bag_path),
                "scan_stamp_ns":       stamp_ns,
                "ring":                ring,
                "azimuth_idx":         col,
                "range_m":             rng,
                "heuristic_mechanism": None,
                "heuristic_score":     None,
                "label":               "artifact",
                "session":             SESSION_ID,
            }
            self._ledger_file.write(json.dumps(record) + "\n")
        self._ledger_file.flush()

        review_record = {
            "bag":           os.path.basename(self._bag_path),
            "scan_stamp_ns": stamp_ns,
            "n_candidates":  0,
            "n_confirmed":   len(selections),
            "session":       SESSION_ID,
            "skipped":       skipped,
        }
        self._reviewed_file.write(json.dumps(review_record) + "\n")
        self._reviewed_file.flush()

    def _load_ledger(self):
        self._all_selections: dict[int, set] = {}
        if not os.path.exists(self._ledger_path):
            return

        self.get_logger().info(f"Loading existing selections from: {self._ledger_path}")
        loaded = 0
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
                        if "label" in data and data.get("label") == "artifact":
                            ring = data.get("ring")
                            col  = data.get("azimuth_idx")
                            if ring is not None and col is not None:
                                self._all_selections.setdefault(stamp_ns, set()).add((ring, col))
                                loaded += 1
                        elif "selections" in data:
                            for item in data["selections"]:
                                r = item.get("ring")
                                c = item.get("col") or item.get("azimuth_idx")
                                if r is not None and c is not None:
                                    self._all_selections.setdefault(stamp_ns, set()).add((r, c))
                                    loaded += 1
                    except Exception as e:
                        self.get_logger().warn(f"Skipped malformed ledger line: {e}")
            self.get_logger().info(
                f"Loaded {loaded} selections across {len(self._all_selections)} scans."
            )
        except Exception as e:
            self.get_logger().error(f"Error reading ledger: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Publishing
    # ─────────────────────────────────────────────────────────────────────────

    def _republish(self) -> None:
        if self._range_grid is None:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_cloud(stamp)
        self._publish_image(stamp)
        self._publish_image_annotations(stamp)
        self._publish_cursor_3d_marker(stamp)

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
                img[px_row, :] = np.where(np.isfinite(row_data), row_data, np.nan)

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
        """
        Publishes cursor and all selected cells as a single POINTS ImageMarker
        with per-point colors via outline_colors. Using one message with one id
        avoids the Foxglove multi-id rendering issue. Scale=1.0 gives a single
        pixel per point.
        """
        if self._layer_to_row is None:
            return

        marker = ImageMarker()
        marker.header.stamp    = stamp
        marker.header.frame_id = self._frame_id
        marker.ns     = "annotator"
        marker.id     = 0
        marker.type   = ImageMarker.POINTS
        marker.action = ImageMarker.ADD
        marker.scale  = 0.5   # single pixel

        def add_point(col: int, layer: int, color: tuple):
            r, g, b = color
            marker.points.append(GeoPoint(
                x=float(col) + 0.5,
                y=float(self._layer_to_row[layer]) + 0.5,
                z=0.0,
            ))
            marker.outline_colors.append(
                ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=0.75)
            )

        # Cursor point (drawn last so it renders on top of any selection
        # that shares the same cell).
        for (sel_layer, sel_col) in self._selected:
            if 0 <= sel_layer < self._grid_rows:
                add_point(sel_col, sel_layer, COLOR_SELECT_IMG)

        if 0 <= self._cursor_layer < self._grid_rows:
            add_point(self._cursor_col, self._cursor_layer, COLOR_CURSOR_IMG)

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
        d = MARKER_RADIUS_M * 2.0
        marker.scale.x = marker.scale.y = marker.scale.z = d
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = MARKER_SELECTED_COLOR if ((cur_layer, cur_col) in self._selected) else MARKER_COLOR
        self._marker_pub.publish(marker)

    def _publish_cloud(self, stamp) -> None:
        xyz = self._xyz
        if xyz is None:
            return

        N      = xyz.shape[0]
        white  = pack_rgb_float(*COLOR_DEFAULT_CLOUD)
        colors = np.full(N, white, dtype=np.float32)

        sel_color = pack_rgb_float(*COLOR_SELECT_CLOUD)
        for (sel_layer, sel_col) in self._selected:
            if 0 <= sel_layer < self._grid_rows:
                idx = int(self._point_idx_img[sel_layer, sel_col])
                if idx >= 0:
                    colors[idx] = sel_color

        cur_layer = self._cursor_layer
        cur_col   = self._cursor_col
        if 0 <= cur_layer < self._grid_rows and 0 <= cur_col < self._num_cols:
            cur_idx = int(self._point_idx_img[cur_layer, cur_col])
            if cur_idx >= 0:
                colors[cur_idx] = pack_rgb_float(*COLOR_CURSOR_CLOUD)

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
        for attr in ("_ledger_file", "_reviewed_file"):
            handle = getattr(self, attr, None)
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
        print(f"RangeAnnotator exiting: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
