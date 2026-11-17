#!/usr/bin/env python3
"""
artifact_features.py

Per-return features for the lidar artifact classifier, computed on the
(rings, cols) scan grid from scan_grid.py. Shared by training and inference so
the two cannot drift apart.

Everything is expressed relative to the return's own range (log range
ratios), because the artifacts live at 0.2-1.5 m where differences of a few
centimetres matter and would vanish under a global range scale.

A return is described by:
  * its own range and ring (and intensity, when the bag has it)
  * the log range ratio to every cell of a 5 x 9 neighbourhood
    (rings +-2, azimuth +-4, circular in azimuth)
  * summaries of that neighbourhood: how many neighbours agree with it, how
    many are a different surface behind / in front, local roughness
  * the log range ratio to the same and adjacent cells in the neighbouring
    scans in time, and how much of the surrounding patch changed

Missing neighbours (no return, off the top/bottom ring, no adjacent scan)
are NaN; the gradient-boosted trees treat NaN as its own branch.

Returns nearer than MIN_RANGE_M are removed before anything is computed, so
recordings that already drop the sensor's blind zone and recordings that do
not are on identical footing.

The constants below are the defaults; config.load() replaces them from the
"features" section of the project configuration through configure(). A model
stores the values it was trained with (spec()) and is refused under others.
"""
from __future__ import annotations

import warnings

import numpy as np

MIN_RANGE_M = 0.2
RING_OFFS = (-2, -1, 0, 1, 2)
COL_OFFS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
CLOSE_LOG = 0.05           # |log ratio| under this = "agrees with me"
FAR_LOG = 0.4              # |log ratio| over this = a different surface
MAX_DT_NS = 80_000_000     # adjacent scan must be within this per step
CLIP = 4.0
DEFAULT_TIME_OFFSETS = (-2, -1, 1, 2)

# What models trained before the settings were recorded were made with.
LEGACY_SPEC = {"min_range_m": 0.2, "ring_offsets": [-2, -1, 0, 1, 2],
               "col_offsets": [-4, -3, -2, -1, 0, 1, 2, 3, 4], "close_log": 0.05,
               "far_log": 0.4, "clip": 4.0, "max_dt_ns": 80_000_000}


def configure(features: dict, scan_rate_hz: float) -> None:
    """Applies the "features" section of the configuration to this process."""
    global MIN_RANGE_M, RING_OFFS, COL_OFFS, CLOSE_LOG, FAR_LOG, MAX_DT_NS, CLIP
    global DEFAULT_TIME_OFFSETS
    ring = tuple(int(v) for v in features["ring_offsets"])
    col = tuple(int(v) for v in features["col_offsets"])
    if not {-1, 0, 1} <= set(ring) or not {-2, -1, 0, 1, 2} <= set(col):
        raise ValueError("features: ring_offsets must include -1..1 and "
                         "col_offsets -2..2")
    MIN_RANGE_M = float(features["min_range_m"])
    RING_OFFS, COL_OFFS = ring, col
    CLOSE_LOG, FAR_LOG = float(features["close_log"]), float(features["far_log"])
    CLIP = float(features["clip"])
    # Neighbouring scans count while they are no more than max_scan_gap scan
    # periods apart per step (1.6 tolerates jitter and rejects a dropped scan).
    MAX_DT_NS = int(round(float(features["max_scan_gap"]) / scan_rate_hz * 1e9))
    DEFAULT_TIME_OFFSETS = tuple(int(v) for v in features["time_offsets"])


def spec() -> dict:
    """The settings a model depends on, stored with it."""
    return {"min_range_m": MIN_RANGE_M, "ring_offsets": list(RING_OFFS),
            "col_offsets": list(COL_OFFS), "close_log": CLOSE_LOG,
            "far_log": FAR_LOG, "clip": CLIP, "max_dt_ns": MAX_DT_NS}


def _log(r: np.ndarray) -> np.ndarray:
    """
    float32 log, computed in double and rounded. A single precision log differs
    in the last bit between numpy and a C library, and the trees split exactly
    at values that occur in the data, so that last bit can flip a decision.
    Rounding a double log gives the same float32 everywhere (the C++ filter in
    ghost_tester does the same).
    """
    return np.log(r.astype(np.float64)).astype(np.float32)


def _shift(a: np.ndarray, dring: int, dcol: int) -> np.ndarray:
    """out[..., r, c] = a[..., r + dring, c + dcol]; NaN beyond the ring ends."""
    out = np.roll(a, -dcol, axis=-1) if dcol else a
    if dring:
        out = np.roll(out, -dring, axis=-2)
        if dring > 0:
            out[..., -dring:, :] = np.nan
        else:
            out[..., :-dring, :] = np.nan
    return out


def _box3x5(a: np.ndarray) -> np.ndarray:
    """Sum over rings +-1, cols +-2 (zero beyond the ring ends)."""
    out = np.zeros_like(a)
    for dr in (-1, 0, 1):
        for dc in (-2, -1, 0, 1, 2):
            out += np.nan_to_num(_shift(a, dr, dc), nan=0.0)
    return out


def feature_names(time_offsets=None, use_intensity=False) -> list[str]:
    if time_offsets is None:
        time_offsets = DEFAULT_TIME_OFFSETS
    names = ["range", "ring"]
    names += [f"s_r{dr:+d}_c{dc:+d}" for dr in RING_OFFS for dc in COL_OFFS
              if (dr, dc) != (0, 0)]
    names += ["n_close_3x5", "n_valid_3x5", "n_close_row", "min_abs_row1",
              "between_row", "n_behind_3x5", "n_front_3x5", "rough_3x5"]
    for dt in time_offsets:
        names += [f"t{dt:+d}_c{dc:+d}" for dc in (-1, 0, 1)]
    if time_offsets:
        names += ["n_close_time", "dev_time_median"]
        names += [f"n_unstable_3x5_t{dt:+d}" for dt in time_offsets if abs(dt) == 1]
        names += [f"n_gone_3x5_t{dt:+d}" for dt in time_offsets if abs(dt) == 1]
    if use_intensity:
        names += ["intensity", "i_c-1", "i_c+1", "i_r-1", "i_r+1"]
    return names


def compute_features(
    rng_m: np.ndarray,
    stack: dict | None = None,
    intensity: np.ndarray | None = None,
    time_offsets=None,
) -> np.ndarray:
    """
    rng_m:     (B, rings, cols) float32 range in metres, NaN = no return
               (use to_metres(), which also drops returns < MIN_RANGE_M).
    stack:     {dt: (B, rings, cols)} range of the scan dt steps away, all-NaN
               where that scan does not exist. Needed for each dt in
               time_offsets.
    intensity: (B, rings, cols) raw intensity counts, or None.

    Returns (B, rings, cols, F) float32, columns as in feature_names(). Only rows
    selected by candidates(rng_m) are meaningful.
    """
    if time_offsets is None:
        time_offsets = DEFAULT_TIME_OFFSETS
    r = rng_m.astype(np.float32)
    logr = _log(r)
    feats = [r, np.broadcast_to(
        np.arange(r.shape[1], dtype=np.float32)[None, :, None], r.shape)]

    zeros = lambda: np.zeros(r.shape, np.float32)
    close3x5, valid3x5, close_row = zeros(), zeros(), zeros()
    behind, front, s1, s2 = zeros(), zeros(), zeros(), zeros()
    row = {}
    for dr in RING_OFFS:
        for dc in COL_OFFS:
            if (dr, dc) == (0, 0):
                continue
            d = np.clip(_shift(logr, dr, dc) - logr, -CLIP, CLIP)
            feats.append(d)
            ok = np.isfinite(d)
            agree = ok & (np.abs(d) < CLOSE_LOG)
            if abs(dr) <= 1 and abs(dc) <= 2:
                close3x5 += agree
                valid3x5 += ok
                behind += ok & (d > FAR_LOG)
                front += ok & (d < -FAR_LOG)
                dz = np.where(ok, np.clip(d, -1.0, 1.0), 0.0)
                s1 += dz
                s2 += dz * dz
            if dr == 0:
                close_row += agree
                row[dc] = d
    with np.errstate(invalid="ignore"):
        l1, r1 = row[-1], row[1]
        min_abs_row1 = np.fmin(np.abs(l1), np.abs(r1))
        # > 0 when the return sits strictly between its two row neighbours
        # (one nearer, one farther): the mixed-pixel signature.
        between = -l1 * r1
    nv = np.maximum(valid3x5, 1.0)
    rough = np.sqrt(np.maximum(s2 / nv - (s1 / nv) ** 2, 0.0))
    feats += [close3x5, valid3x5, close_row, min_abs_row1, between,
              behind, front, rough]

    if time_offsets:
        close_t = zeros()
        same_cell = []
        patch_change = {}
        for dt in time_offsets:
            lt = _log(stack[dt].astype(np.float32))
            for dc in (-1, 0, 1):
                d = np.clip(_shift(lt, 0, dc) - logr, -CLIP, CLIP)
                feats.append(d)
                if dc == 0:
                    close_t += np.isfinite(d) & (np.abs(d) < CLOSE_LOG)
                    same_cell.append(d)
            if abs(dt) == 1:
                # How much of the 3x5 patch around the return changed against
                # the adjacent scan: unstable = present in both but the range
                # moved, gone = present now and absent then.
                here, there = np.isfinite(logr), np.isfinite(lt)
                with np.errstate(invalid="ignore"):
                    unstable = here & there & (np.abs(lt - logr) > CLOSE_LOG)
                patch_change[dt] = (_box3x5(unstable.astype(np.float32)),
                                    _box3x5((here & ~there).astype(np.float32)))
        feats.append(close_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.stack(same_cell + [np.zeros_like(logr)]), axis=0)
        feats.append(np.abs(med))
        feats += [patch_change[dt][0] for dt in time_offsets if abs(dt) == 1]
        feats += [patch_change[dt][1] for dt in time_offsets if abs(dt) == 1]

    if intensity is not None:
        feats.append(intensity_features(r, intensity))
        return np.concatenate(
            [np.stack(feats[:-1], axis=-1), feats[-1]], axis=-1)

    return np.stack(feats, axis=-1)


def intensity_features(rng_m: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """(B, rings, cols, 5): own intensity and differences to the 4 direct neighbours."""
    it = intensity.astype(np.float32) / 65535.0
    it = np.where(np.isfinite(rng_m), it, np.nan)
    out = [it]
    for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        out.append(_shift(it, dr, dc) - it)
    return np.stack(out, axis=-1)


def candidates(rng_m: np.ndarray) -> np.ndarray:
    """Cells that hold a classifiable return."""
    return np.isfinite(rng_m)


def to_metres(range_mm: np.ndarray) -> np.ndarray:
    """uint16 mm grid -> float32 metres, NaN for no return or < MIN_RANGE_M."""
    r = range_mm.astype(np.float32) / 1000.0
    r[range_mm < int(round(MIN_RANGE_M * 1000))] = np.nan
    return r


def gather_time_stack(range_mm, hdr_t: np.ndarray, idx: np.ndarray,
                      time_offsets=None) -> dict:
    """
    For scan indices `idx` into a bag cache, returns {dt: (len(idx), rings, cols)}
    ranges in metres of the scans dt steps away. A neighbour that is missing,
    or further in time than |dt| * MAX_DT_NS, is all-NaN.
    """
    if time_offsets is None:
        time_offsets = DEFAULT_TIME_OFFSETS
    idx = np.asarray(idx)
    n = len(hdr_t)
    out = {}
    for dt in time_offsets:
        j = idx + dt
        ok = (j >= 0) & (j < n)
        jc = np.clip(j, 0, n - 1)
        ok &= np.abs(hdr_t[jc] - hdr_t[idx]) <= abs(dt) * MAX_DT_NS
        uniq, inv = np.unique(jc, return_inverse=True)
        a = to_metres(np.asarray(range_mm[uniq]))[inv]
        a[~ok] = np.nan
        out[dt] = a
    return out
