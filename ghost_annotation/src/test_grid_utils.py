"""
Synthetic sanity checks for grid_utils.py v2 -- run with:
    python3 test_grid_utils.py
No ROS dependencies needed.

Covers the failure modes reported from real-data testing:
  - far-range bouncing/noisy points (low angular resolution) -> must NOT
    be flagged now that detection is range-gated
  - near-range ghost/blur artifact -> must be flagged
  - near-range pop-out (single point closer than everything around it)
    -> must be flagged
  - non-uniform layer spacing -> nearest-angle binning must still land
    points in the correct row
"""

import numpy as np
from grid_utils import (
    compute_polar_grid,
    build_range_image,
    ghost_candidate_scores,
    popout_candidate_scores,
    get_candidate_cells,
)

GRID_COLS = 360

# Deliberately NON-uniform layer spacing, to test that nearest-angle
# binning (not a uniform-bin formula) is actually being used.
LAYER_ELEVATIONS_DEG = np.array([
    42.2, 35.0, 29.0, 24.5, 19.0, 14.8, 10.0,
    5.5, 1.0, -4.0, -9.5, -14.0, -18.5, -22.2,
])
LAYER_ELEVATIONS_RAD = np.radians(LAYER_ELEVATIONS_DEG)
GRID_ROWS = len(LAYER_ELEVATIONS_DEG)

MAX_CANDIDATE_RANGE_M = 3.0


def make_synthetic_scan():
    """
    Builds a scan with several known regions:
      - near wall at 2.0m, columns 0-179 (everything inside candidate range)
      - far wall at 8.0m, columns 180-269 (outside candidate range -- should
        never be flagged regardless of how noisy it is)
      - far wall gets per-point quantization-style noise (+/- up to 0.4m,
        unrealistically large on purpose) to simulate the "bounces a lot at
        low angular resolution" complaint
      - a near-range doorway edge at columns 90-95 (1.0m) with REAL ghost
        artifacts injected at the boundary cells (range between 1.0 and 2.0)
      - a near-range pop-out spike injected at column 40 (single point much
        closer than its neighbors)
      - some cells are dropped entirely (NaN) to simulate telemetry-
        reduction gaps
    """
    rows, cols = GRID_ROWS, GRID_COLS
    az_width = 2 * np.pi / cols
    az = -np.pi + (np.arange(cols) + 0.5) * az_width
    el = LAYER_ELEVATIONS_RAD  # already sorted descending, row 0 = top

    range_grid = np.full((rows, cols), 2.0)  # near wall baseline

    # Far wall, columns 180-269
    range_grid[:, 180:270] = 8.0
    rng = np.random.default_rng(42)
    far_noise = rng.uniform(-0.4, 0.4, size=(rows, 90))
    range_grid[:, 180:270] += far_noise

    # Doorway edge cut into the near wall: columns 90-95 are actually 1.0m
    range_grid[:, 90:96] = 1.0

    ghost_cells = [(2, 89), (3, 89), (5, 96), (6, 96), (9, 89)]
    for r, c in ghost_cells:
        range_grid[r, c] = 1.5  # smeared value between 1.0 and 2.0

    popout_cells = [(7, 40)]
    for r, c in popout_cells:
        range_grid[r, c] = 1.2  # ~0.8m closer than the surrounding 2.0m wall

    dropped_cells = [(0, 10), (4, 200), (11, 300)]  # simulate gaps

    pts = []
    orig_row_col = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in dropped_cells:
                continue
            d = range_grid[r, c]
            x = d * np.cos(el[r]) * np.cos(az[c])
            y = d * np.cos(el[r]) * np.sin(az[c])
            z = d * np.sin(el[r])
            pts.append((x, y, z))
            orig_row_col.append((r, c))

    xyz = np.array(pts, dtype=np.float64)
    return xyz, ghost_cells, popout_cells, dropped_cells


def main():
    xyz, ghost_cells, popout_cells, dropped_cells = make_synthetic_scan()

    grid = compute_polar_grid(xyz, LAYER_ELEVATIONS_RAD, GRID_COLS)
    print(f"Valid points: {grid['valid_mask'].sum()} / {len(xyz)}")

    range_img, point_idx_img = build_range_image(
        grid["row_idx"], grid["col_idx"], grid["ranges"], grid["orig_indices"],
        grid["grid_rows"], GRID_COLS,
    )

    print(f"\n--- Binning correctness (non-uniform layer spacing) ---")
    print(f"Dropped-cell gaps preserved as NaN: "
          f"{[np.isnan(range_img[r, c]) for r, c in dropped_cells]} (expect all True)")
    print(f"Near wall sample range_img[7, 50] = {range_img[7, 50]:.2f} (expect 2.00)")
    print(f"Far wall sample (noisy) range_img[7, 220] = {range_img[7, 220]:.2f} "
          f"(expect ~8.0 +/- 0.4)")

    ghost_scores = ghost_candidate_scores(range_img, MAX_CANDIDATE_RANGE_M)
    popout_scores = popout_candidate_scores(range_img, MAX_CANDIDATE_RANGE_M)
    candidates = get_candidate_cells(ghost_scores, 0.3, popout_scores, 0.3)
    flagged_cells = {(r, c) for r, c, _, _ in candidates}

    print(f"\n--- Candidate detection ---")
    print(f"Total candidates flagged: {len(candidates)}")

    far_range_flags = [(r, c, m, s) for r, c, m, s in candidates if c >= 180]
    print(f"\nFar-range false positives (col >= 180): {len(far_range_flags)} (expect 0)")
    if far_range_flags:
        print(f"  -> {far_range_flags[:10]}")

    ghost_set = set(ghost_cells)
    ghost_recall = len(ghost_set & flagged_cells) / len(ghost_set)
    print(f"\nGhost artifact recall: {ghost_recall:.2f} "
          f"({len(ghost_set & flagged_cells)}/{len(ghost_set)} caught)")
    print(f"Missed ghost cells: {ghost_set - flagged_cells}")

    popout_set = set(popout_cells)
    popout_recall = len(popout_set & flagged_cells) / len(popout_set)
    print(f"\nPop-out artifact recall: {popout_recall:.2f} "
          f"({len(popout_set & flagged_cells)}/{len(popout_set)} caught)")
    print(f"Missed popout cells: {popout_set - flagged_cells}")

    near_field_non_artifact_flags = [
        (r, c, m, s) for r, c, m, s in candidates
        if c < 180 and (r, c) not in ghost_set and (r, c) not in popout_set
    ]
    print(f"\nNear-field false positives (flagged but not injected): "
          f"{len(near_field_non_artifact_flags)} (expect 0, or very few)")
    if near_field_non_artifact_flags:
        print(f"  -> {near_field_non_artifact_flags}")


if __name__ == "__main__":
    main()
