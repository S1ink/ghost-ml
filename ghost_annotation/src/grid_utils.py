"""
grid_utils.py  (v2)

Changes from v1, driven by real-data testing feedback:

1. compute_polar_grid now bins by NEAREST ACTUAL LAYER ANGLE, not uniform
   spacing across the FOV. Pass in your sensor's real per-layer elevation
   angles (see calibrate_layer_elevations.py to extract these empirically
   from a raw/non-reduced bag that still has ring data).

2. heuristic candidate scoring is now range-gated: only points within
   MAX_CANDIDATE_RANGE_M are eligible to be flagged at all. This is the
   direct fix for far-range false positives -- bouncing/noisy far points
   (a low-angular-resolution artifact, not a geometry problem) are
   excluded by construction rather than being fought with a smarter score.

3. Added a second, independent detector: popout_candidate_scores, for
   points that read closer than everything around them in their own ring
   (a local-minimum spike), which is a distinct failure mode from the
   "ghost between two surfaces" blur the original detector targets.

4. Fixed a real bug in the cross-ring boost: it used np.roll on the row
   axis, which wraps row 0 to row 13 -- treating elevation as circular,
   which it isn't. Replaced with an edge-aware shift that leaves the top/
   bottom row's missing neighbor as NaN instead of wrapping.
"""

from __future__ import annotations

import warnings

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


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _shift_cols_circular(arr: np.ndarray, k: int) -> np.ndarray:
    """Azimuth wraps around -- this IS supposed to be circular."""
    return np.roll(arr, k, axis=1)


def _shift_rows_edge_aware(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Elevation does NOT wrap -- shifting by k rows must leave the missing
    edge as NaN rather than pulling in the opposite end of the FOV (the
    v1 bug: np.roll on axis=0 silently treated row 0's "up" neighbor as
    the bottom row).
    """
    shifted = np.roll(arr, k, axis=0)
    if k > 0:
        shifted[:k, :] = np.nan
    elif k < 0:
        shifted[k:, :] = np.nan
    return shifted


# ─────────────────────────────────────────────────────────────────────────
# Detector 1: "ghost" / edge-blur candidates (point matches neither
# immediate neighbor -- the original ShadowPoints-style failure mode)
# ─────────────────────────────────────────────────────────────────────────

def ghost_candidate_scores(
    range_img: np.ndarray,
    max_candidate_range_m: float,
    min_context_jump_m: float = 0.3,
) -> np.ndarray:
    """
    Flags points that match NEITHER immediate azimuthal neighbor closely,
    while a real transition exists nearby (min_context_jump_m gate) -- the
    "blended/blurred between two surfaces" signature. A genuine sharp,
    non-blurred edge does NOT get flagged here, because the boundary point
    on either side of a clean step still matches one of its neighbors
    exactly.

    Range-gated to max_candidate_range_m: the main fix for far-range false
    positives. Far, low-angular-resolution points naturally jitter between
    neighbors due to range quantization noise, not real edges, and that
    jitter satisfies this same "matches neither neighbor" signature just
    from sensor noise. Since the artifacts you actually care about all
    occur close-in, excluding far points from candidacy removes this
    failure mode by construction rather than out-clevering it with a
    sharper threshold.
    """
    left = _shift_cols_circular(range_img, 1)
    right = _shift_cols_circular(range_img, -1)

    valid = ~np.isnan(range_img) & ~np.isnan(left) & ~np.isnan(right)
    valid &= range_img <= max_candidate_range_m

    diff_left = np.abs(range_img - left)
    diff_right = np.abs(range_img - right)
    total_jump = np.abs(right - left)
    nearest_neighbor_diff = np.minimum(diff_left, diff_right)

    raw_score = nearest_neighbor_diff * (total_jump > min_context_jump_m)

    # Cross-ring boost, edge-aware now (no more row-wrap bug).
    up = _shift_rows_edge_aware(range_img, 1)
    down = _shift_rows_edge_aware(range_img, -1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cross_ring_jump = np.nanmin(
            np.stack([np.abs(range_img - up), np.abs(range_img - down)]), axis=0
        )
    cross_ring_jump = np.nan_to_num(cross_ring_jump, nan=0.0)
    raw_score = raw_score + 0.5 * cross_ring_jump

    scores = np.full_like(range_img, np.nan, dtype=np.float32)
    scores[valid] = raw_score[valid]
    return scores


# ─────────────────────────────────────────────────────────────────────────
# Detector 2: "pop-out" candidates -- a single point reading closer than
# everything around it in its own ring (distinct from ghost blur; this is
# a local-minimum spike, not a between-two-surfaces blend)
# ─────────────────────────────────────────────────────────────────────────

def popout_candidate_scores(
    range_img: np.ndarray,
    max_candidate_range_m: float,
    window_radius: int = 5,
) -> np.ndarray:
    """
    Flags points that read closer than ALL points in a same-ring window
    around them (excluding the point itself). Score = how much closer
    (meters) than the nearest of those window neighbors -- threshold this
    yourself; bigger margin = more confident it's a real pop-out, not just
    normal surface micro-texture.

    Range-gated the same way as ghost_candidate_scores, since you said
    this failure mode also only matters within the near-field range.
    """
    offsets = range(1, window_radius + 1)
    shifts = []
    for k in offsets:
        shifts.append(_shift_cols_circular(range_img, k))
        shifts.append(_shift_cols_circular(range_img, -k))
    window_stack = np.stack(shifts)

    with np.errstate(invalid="ignore"):
        local_min_neighbor = np.nanmin(window_stack, axis=0)

    valid = ~np.isnan(range_img) & ~np.isnan(local_min_neighbor)
    valid &= range_img <= max_candidate_range_m

    margin = local_min_neighbor - range_img  # positive = point closer than everything around it

    scores = np.full_like(range_img, np.nan, dtype=np.float32)
    scores[valid] = margin[valid]
    return scores


# ─────────────────────────────────────────────────────────────────────────
# Combining both detectors into one candidate list
# ─────────────────────────────────────────────────────────────────────────

def get_candidate_cells(
    ghost_scores: np.ndarray,
    ghost_threshold: float,
    popout_scores: np.ndarray,
    popout_threshold: float,
) -> list[tuple[int, int, str, float]]:
    """
    Unions both detectors into one score-sorted candidate list.

    Returns (row, col, mechanism, score) tuples, sorted by score
    descending. mechanism is "ghost" or "popout" -- worth keeping in your
    ledger; it'll tell you later which detector is earning its keep and
    which threshold needs adjusting, without re-deriving it from range
    values after the fact.

    If a cell is flagged by both detectors, it's included once, tagged
    with whichever mechanism produced the higher score (the two scores
    are both "meters of confidence" in a loose sense, so taking the larger
    to break ties is reasonable without true normalization).
    """
    ghost_rows, ghost_cols = np.nonzero(np.nan_to_num(ghost_scores, nan=-np.inf) > ghost_threshold)
    popout_rows, popout_cols = np.nonzero(np.nan_to_num(popout_scores, nan=-np.inf) > popout_threshold)

    candidates: dict[tuple[int, int], tuple[str, float]] = {}
    for r, c in zip(ghost_rows, ghost_cols):
        candidates[(int(r), int(c))] = ("ghost", float(ghost_scores[r, c]))
    for r, c in zip(popout_rows, popout_cols):
        existing = candidates.get((int(r), int(c)))
        score = float(popout_scores[r, c])
        if existing is None or score > existing[1]:
            candidates[(int(r), int(c))] = ("popout", score)

    sorted_items = sorted(candidates.items(), key=lambda kv: -kv[1][1])
    return [(r, c, mech, score) for (r, c), (mech, score) in sorted_items]
