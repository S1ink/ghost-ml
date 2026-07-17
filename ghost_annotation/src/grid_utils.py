from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Binning: raw XYZ -> polar grid using REAL per-layer elevation angles
# ─────────────────────────────────────────────────────────────────────────

def compute_polar_grid(
    xyz: np.ndarray,
    layer_elevations_rad: np.ndarray,
    grid_cols: int,
    azimuth_offset_rad: float = 0.0,
    max_elevation_diff_rad: float | None = None,
) -> dict:
    """
    Bins an unordered (N, 3) XYZ array into a polar grid using NEAREST-ANGLE
    assignment against your sensor's real per-layer elevation angles, not
    uniform spacing. This matters because uneven layer spacing means a
    formula-based row index would misassign points near layer boundaries,
    which corrupts same-row neighbor comparisons right where the artifact
    heuristic makes its decisions.

    layer_elevations_rad: array of your sensor's actual per-layer beam
        angles, in radians. This function sorts them descending internally
        (row 0 = top layer) regardless of the order you pass in -- the
        returned row indices correspond to that sorted order. See
        `sorted_layer_elevations_rad` in the returned dict if you need the
        actual per-row angle for display.

    max_elevation_diff_rad: if set, points whose elevation is farther than
        this from their nearest layer angle are dropped as invalid (out of
        FOV / nearest-layer match too poor to trust). If None, every point
        is force-assigned to its nearest layer regardless of distance.

    Returns a dict with:
        row_idx, col_idx              : (M,) int arrays, valid-point bins
        valid_mask                    : (N,) bool, which input points kept
        ranges                        : (M,) float, range per valid point
        orig_indices                  : (M,) int, index into ORIGINAL xyz
        sorted_layer_elevations_rad   : (rows,) float, layer angles in the
                                         row order actually used
        grid_rows                     : int, len(layer_elevations_rad)
    """
    layer_elevations_rad = np.asarray(layer_elevations_rad, dtype=np.float64)
    sort_order = np.argsort(-layer_elevations_rad)  # descending: row 0 = top
    sorted_layers = layer_elevations_rad[sort_order]
    grid_rows = len(sorted_layers)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xy_dist_sq = x * x + y * y
    ranges_all = np.sqrt(xy_dist_sq + z * z)

    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    range_mask = ranges_all > 1e-4

    azimuth_all = np.mod(np.arctan2(y, x) + azimuth_offset_rad + np.pi, 2 * np.pi) - np.pi
    elevation_all = np.arctan2(z, np.sqrt(xy_dist_sq))

    # Nearest-layer assignment (replaces the old uniform-bin formula).
    diffs = np.abs(elevation_all[:, None] - sorted_layers[None, :])  # (N, rows)
    row_idx_all = np.argmin(diffs, axis=1)
    min_diff = diffs[np.arange(len(diffs)), row_idx_all]

    if max_elevation_diff_rad is not None:
        elev_ok = min_diff <= max_elevation_diff_rad
    else:
        elev_ok = np.ones_like(min_diff, dtype=bool)

    valid_mask = finite_mask & range_mask & elev_ok
    orig_indices = np.nonzero(valid_mask)[0]

    row_idx = row_idx_all[valid_mask]
    ranges = ranges_all[valid_mask]
    azimuth = azimuth_all[valid_mask]

    az_span = 2 * np.pi
    u = (azimuth + np.pi) / az_span
    col_idx = np.clip((u * grid_cols).astype(np.int64), 0, grid_cols - 1)

    return {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "valid_mask": valid_mask,
        "ranges": ranges,
        "orig_indices": orig_indices,
        "sorted_layer_elevations_rad": sorted_layers,
        "grid_rows": grid_rows,
    }


def build_range_image(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    ranges: np.ndarray,
    orig_indices: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scatters points into a (grid_rows, grid_cols) range image, keeping the
    CLOSEST range on collision. Cells with no point (including gaps from
    your telemetry reduction dropping no-return points) are left as NaN --
    this is the correct default, and nothing downstream should treat a
    missing cell as range 0 or interpolate over it silently.
    """
    range_img = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    point_idx_img = np.full((grid_rows, grid_cols), -1, dtype=np.int64)

    order = np.argsort(-ranges)  # far -> near, so nearest wins on collision
    for i in order:
        r, c = row_idx[i], col_idx[i]
        range_img[r, c] = ranges[i]
        point_idx_img[r, c] = orig_indices[i]

    return range_img, point_idx_img
