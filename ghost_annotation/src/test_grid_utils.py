"""
Quick synthetic sanity checks for grid_utils.py -- run with:
    python3 test_grid_utils.py
No ROS dependencies needed.
"""

import numpy as np
from grid_utils import (
    compute_polar_grid,
    build_range_image,
    heuristic_candidate_scores,
    get_candidate_cells,
)

GRID_ROWS = 14
GRID_COLS = 360
ELEV_MIN = np.radians(-32.5)
ELEV_MAX = np.radians(32.5)


def make_synthetic_scan():
    """
    Build a fake scan: 14 rings x 360 azimuth steps, all on a flat wall at
    range 5.0m, EXCEPT a few cells injected with a "flying pixel" value
    sitting between the wall (5.0m) and a doorway behind it (9.0m).
    """
    rows, cols = GRID_ROWS, GRID_COLS
    # Bin CENTERS, not edges -- placing synthetic points exactly on a bin
    # boundary makes the test vulnerable to floating-point round-trip
    # aliasing through atan2 (a point sitting exactly at a boundary can
    # land in either neighboring bin depending on sub-ULP noise). Real
    # lidar data essentially never sits exactly on a bin edge, so this is
    # a test-generator concern, not a grid_utils bug -- but worth avoiding
    # here to get a clean read on the heuristic itself.
    az_width = 2 * np.pi / cols
    el_width = (ELEV_MAX - ELEV_MIN) / rows
    az = -np.pi + (np.arange(cols) + 0.5) * az_width
    el = ELEV_MAX - (np.arange(rows) + 0.5) * el_width  # row 0 = top

    pts = []
    ranges_grid = np.full((rows, cols), 5.0)

    # Doorway: columns 100-130 are actually 9.0m (real edge, not noise)
    ranges_grid[:, 100:130] = 9.0

    # Inject flying-pixel artifacts right at the edges (cols 99, 130) on a
    # few rings -- values between 5.0 and 9.0, simulating beam straddling.
    injected_cells = [(3, 99), (4, 99), (5, 130), (6, 130), (9, 99)]
    for r, c in injected_cells:
        ranges_grid[r, c] = 7.0  # smeared value between the two surfaces

    for r in range(rows):
        for c in range(cols):
            rng = ranges_grid[r, c]
            x = rng * np.cos(el[r]) * np.cos(az[c])
            y = rng * np.cos(el[r]) * np.sin(az[c])
            z = rng * np.sin(el[r])
            pts.append((x, y, z))

    xyz = np.array(pts, dtype=np.float64)
    return xyz, injected_cells


def main():
    xyz, injected_cells = make_synthetic_scan()

    grid = compute_polar_grid(
        xyz,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        elevation_min_rad=ELEV_MIN,
        elevation_max_rad=ELEV_MAX,
    )
    print(f"Valid points: {grid['valid_mask'].sum()} / {len(xyz)}")

    range_img, point_idx_img = build_range_image(
        grid["row_idx"], grid["col_idx"], grid["ranges"], grid["orig_indices"],
        GRID_ROWS, GRID_COLS,
    )
    print(f"Range image shape: {range_img.shape}, "
          f"NaN cells: {np.isnan(range_img).sum()} (expect 0, every cell has a point)")

    # Sanity check a known flat-wall cell and a known doorway cell.
    print(f"Flat wall sample range_img[7, 50] = {range_img[7, 50]:.2f} (expect 5.00)")
    print(f"Doorway sample   range_img[7, 115] = {range_img[7, 115]:.2f} (expect 9.00)")
    for r, c in injected_cells:
        print(f"Injected artifact range_img[{r},{c}] = {range_img[r, c]:.2f} (expect 7.00)")

    scores = heuristic_candidate_scores(range_img)
    candidates = get_candidate_cells(scores, threshold=1.5)
    print(f"\nCandidates flagged: {len(candidates)}")
    print(f"Candidates: {candidates}")

    injected_set = set(injected_cells)
    flagged_set = set(candidates)
    recall = len(injected_set & flagged_set) / len(injected_set)
    print(f"\nRecall on injected artifacts: {recall:.2f} "
          f"({len(injected_set & flagged_set)}/{len(injected_set)} caught)")
    false_positives = flagged_set - injected_set
    print(f"False positives (non-injected cells flagged): {len(false_positives)}")
    if false_positives:
        print(f"  -> {sorted(false_positives)}")


if __name__ == "__main__":
    main()
