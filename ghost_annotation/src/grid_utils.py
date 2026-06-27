"""
grid_utils.py

Pure-numpy logic for turning an unordered XYZ point cloud into a fixed
(rows x cols) polar range grid, plus the candidate-detection heuristic used
to pre-flag likely edge-blur ("flying pixel") points for manual review.

Kept free of any ROS/rclpy imports so it can be unit tested in isolation
with synthetic data (see test_grid_utils.py).
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Binning: raw XYZ -> fixed (row, col) polar grid
# ─────────────────────────────────────────────────────────────────────────

def compute_polar_grid(
    xyz: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    elevation_min_rad: float,
    elevation_max_rad: float,
    azimuth_offset_rad: float = 0.0,
) -> dict:
    """
    Bins an unordered (N, 3) XYZ array into a FIXED polar grid.

    Unlike the Foxglove viewer script this was adapted from, bin boundaries
    here are fixed constants (full 360 degrees of azimuth, and the sensor's
    known elevation FOV), not recomputed per-scan. This is required so that
    (row, col) is a stable physical identity across scans -- the same
    azimuth/elevation always lands in the same cell, scan to scan, which
    the annotation ledger and the training cache both depend on.

    Returns a dict with:
        row_idx, col_idx   : (M,) int arrays, valid-point bin coordinates
        valid_mask         : (N,) bool array, which input points were kept
                              (range > epsilon and within elevation FOV)
        ranges              : (M,) float array, range of each valid point
        orig_indices        : (M,) int array, index into the ORIGINAL xyz
                              array for each valid point (lets you map a
                              grid cell back to a raw point index for
                              recoloring / publishing)
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xy_dist_sq = x * x + y * y
    ranges_all = np.sqrt(xy_dist_sq + z * z)

    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    range_mask = ranges_all > 1e-4

    azimuth_all = np.mod(np.arctan2(y, x) + azimuth_offset_rad + np.pi, 2 * np.pi) - np.pi
    elevation_all = np.arctan2(z, np.sqrt(xy_dist_sq))

    elev_mask = (elevation_all >= elevation_min_rad) & (elevation_all <= elevation_max_rad)

    valid_mask = finite_mask & range_mask & elev_mask
    orig_indices = np.nonzero(valid_mask)[0]

    azimuth = azimuth_all[valid_mask]
    elevation = elevation_all[valid_mask]
    ranges = ranges_all[valid_mask]

    # Fixed bin width, full circle / full known FOV -- NOT data-dependent.
    az_span = 2 * np.pi
    el_span = elevation_max_rad - elevation_min_rad

    u = (azimuth + np.pi) / az_span          # azimuth in [-pi, pi) -> [0, 1)
    v = (elevation - elevation_min_rad) / el_span  # elevation -> [0, 1)

    col_idx = np.clip((u * grid_cols).astype(np.int64), 0, grid_cols - 1)
    # row 0 = top (max elevation), matches typical "ring 0 = top layer" convention
    row_idx = np.clip(((1.0 - v) * grid_rows).astype(np.int64), 0, grid_rows - 1)

    return {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "valid_mask": valid_mask,
        "ranges": ranges,
        "orig_indices": orig_indices,
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
    CLOSEST range on collision (multiple points landing in the same cell --
    expected given the lossy bandwidth-reduction your data went through).

    Returns:
        range_img      : (rows, cols) float32, NaN where no point landed
        point_idx_img   : (rows, cols) int64, index into the ORIGINAL point
                          array occupying that cell (-1 if empty). Lets the
                          annotation node map a grid cell back to a raw
                          point for recoloring in the republished cloud.
    """
    range_img = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    point_idx_img = np.full((grid_rows, grid_cols), -1, dtype=np.int64)

    # Process in descending range order so the LAST write per cell is the
    # closest point (simple way to get "keep nearest" without a manual loop
    # comparison -- last-write-wins, and we sort far-to-near beforehand).
    order = np.argsort(-ranges)  # far -> near
    for i in order:
        r, c = row_idx[i], col_idx[i]
        range_img[r, c] = ranges[i]
        point_idx_img[r, c] = orig_indices[i]

    return range_img, point_idx_img


# ─────────────────────────────────────────────────────────────────────────
# Heuristic candidate detection (no normals required -- works on range only)
# ─────────────────────────────────────────────────────────────────────────

def heuristic_candidate_scores(range_img: np.ndarray) -> np.ndarray:
    """
    Scores every cell by how strongly it looks like a flying-pixel / edge-
    blur point: a range value that sits BETWEEN two distinct depth clusters
    along its own ring, rather than simply being far from its neighbors.

    Works purely off range values (no normals), so it's robust at low
    vertical resolution where normal estimation breaks down.

    Returns a (rows, cols) float32 score array, NaN where there's no point.
    Higher score = more suspicious. Threshold this yourself (deliberately
    loose, since this only feeds a human review step, not a final filter).
    """
    rows, cols = range_img.shape
    scores = np.full((rows, cols), np.nan, dtype=np.float32)

    # Azimuth is circular -- wrap neighbors around the seam.
    left = np.roll(range_img, 1, axis=1)
    right = np.roll(range_img, -1, axis=1)

    valid = ~np.isnan(range_img) & ~np.isnan(left) & ~np.isnan(right)

    diff_left = np.abs(range_img - left)
    diff_right = np.abs(range_img - right)
    total_jump = np.abs(right - left)  # only meaningful where l/r differ a lot

    # KEY DISTINCTION (this is the part that actually separates blur from a
    # genuine sharp edge): a real edge point still matches ONE of its
    # neighbors closely -- it's still fully on one surface, just at the
    # boundary. A flying-pixel / blurred point matches NEITHER neighbor --
    # it's a blend that belongs fully to neither surface. So the signal is
    # "distance to the CLOSER of the two neighbors", not the raw curvature
    # (curvature/second-derivative fires just as hard on real edges, since
    # |d_right - d_left| is large for a genuine step too -- that was the
    # bug in the first version of this function).
    nearest_neighbor_diff = np.minimum(diff_left, diff_right)

    raw_score = nearest_neighbor_diff * (total_jump > 0.3)  # 0.3m: tune to your scenes

    # Same-column (cross-ring) check is weaker at 14 rows but still useful:
    # a true edge tends to affect multiple adjacent rings at the same
    # azimuth; pure sensor noise on a flat surface usually doesn't.
    up = np.roll(range_img, 1, axis=0)
    down = np.roll(range_img, -1, axis=0)
    with np.errstate(invalid="ignore"):
        cross_ring_jump = np.nanmin(
            np.stack([np.abs(range_img - up), np.abs(range_img - down)]), axis=0
        )
    cross_ring_jump = np.nan_to_num(cross_ring_jump, nan=0.0)

    raw_score = raw_score + 0.5 * cross_ring_jump

    scores[valid] = raw_score[valid]
    return scores


def get_candidate_cells(
    scores: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """
    Returns (row, col) cells exceeding `threshold`, sorted by descending
    score so the most suspicious points are reviewed first within a scan.
    Deliberately loose thresholds are fine -- false positives just cost a
    quick deny click during review.
    """
    rows, cols = np.nonzero(scores > threshold)
    cell_scores = scores[rows, cols]
    order = np.argsort(-cell_scores)
    return [(int(rows[i]), int(cols[i])) for i in order]
