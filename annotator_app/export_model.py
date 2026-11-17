#!/usr/bin/env python3
"""
export_model.py

Writes a trained model as one plain text file that needs no Python to use:
the trees, plus every setting needed to turn a point cloud into the features
the trees expect (sensor grid, feature settings, threshold). The C++ example
in ghost_tester (include/artifact_filter/artifact_filter.hpp) reads it.

    python3 export_model.py [--project DIR] [--variant xyz|full] [-o FILE]
    python3 export_model.py --model path/to/artifact_xyz.joblib [-o FILE]

The trainer writes this file next to every model it saves
(artifact_<variant>.model.txt), so this script is only needed for models
trained before that, or to make a check file:

    python3 export_model.py ... --check-file check.bin --bag <bag> [--topic T]
                                [--skip 200] [--scans 12]

writes some scans of a bag together with the probabilities Python computes
for them, for `artifact_filter_check` to compare its own result against.

File format (text, one item per line, numbers in decimal):

    lidar_artifact_model 1
    <key> <values...>          settings, see write_model()
    early_exit <n> <trees> <cutoff> ...   optional shortcut, see below
    trees <n_trees> <n_nodes>
    tree <n_nodes>             then one line per node:
    <feature> <threshold> <left> <right> <missing_left> <value>
    ...
    end

A node with feature -1 is a leaf and `value` is its output. Otherwise go to
node `left` when the feature value is below the threshold (`compare lt`: x <
threshold, `compare le`: x <= threshold), to `right` when not, and to `left`
or `right` as `missing_left` says when the value is NaN. The probability is
1 / (1 + exp(-(base_margin + sum of the leaf values of all trees))).

early_exit lists checkpoints: once <trees> trees are summed, a return whose
running sum (base_margin included) is below <cutoff> may be given up as
clean without evaluating the rest. See calibrate_early_exit().
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact_features as af
import scan_grid as sg

FORMAT_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Trees out of the two model types the trainer produces
# ─────────────────────────────────────────────────────────────────────────────

def _xgboost_trees(model):
    booster = model.get_booster()
    doc = json.loads(bytes(booster.save_raw(raw_format="json")))
    trees = []
    for t in doc["learner"]["gradient_booster"]["model"]["trees"]:
        left = np.array(t["left_children"], np.int32)
        leaf = left < 0
        # XGBoost holds these as float32 and compares in float32. They must
        # be exactly those values: real inputs often sit right on a split.
        cond = np.array(t["split_conditions"], np.float32).astype(np.float64)
        trees.append({
            "feature": np.where(leaf, -1, np.array(t["split_indices"], np.int32)),
            "threshold": np.where(leaf, 0.0, cond),
            "left": left,
            "right": np.array(t["right_children"], np.int32),
            "missing_left": np.array(t["default_left"], np.int32) != 0,
            "value": np.where(leaf, cond, 0.0),
        })
    return trees, "lt"


def _sklearn_trees(model):
    trees = []
    for stage in model._predictors:
        n = stage[0].nodes
        leaf = n["is_leaf"].astype(bool)
        if np.any(n["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        trees.append({
            "feature": np.where(leaf, -1, n["feature_idx"].astype(np.int32)),
            "threshold": np.where(leaf, 0.0, n["num_threshold"].astype(np.float64)),
            "left": np.where(leaf, -1, n["left"].astype(np.int32)),
            "right": np.where(leaf, -1, n["right"].astype(np.int32)),
            "missing_left": n["missing_go_to_left"].astype(bool),
            "value": np.where(leaf, n["value"].astype(np.float64), 0.0),
        })
    return trees, "le"


def extract_trees(model):
    """(trees, compare, base_margin) of a fitted XGBClassifier or
    HistGradientBoostingClassifier."""
    kind = type(model).__name__
    if kind == "XGBClassifier":
        trees, compare = _xgboost_trees(model)
    elif kind == "HistGradientBoostingClassifier":
        trees, compare = _sklearn_trees(model)
    else:
        raise ValueError(f"cannot export a {kind}")
    # The constant term, measured rather than read from the library's own
    # (version dependent) fields: what the model says for a row of missing
    # values minus what the trees add up to for it.
    n_feat = int(model.n_features_in_)
    probe = np.full((1, n_feat), np.nan, np.float32)
    p = float(model.predict_proba(probe)[0, 1])
    base = float(np.log(p / (1.0 - p)) - tree_sum(trees, compare, probe)[0])
    return trees, compare, base


def tree_sum(trees, compare, X) -> np.ndarray:
    """Reference evaluation of exported trees (the same rule the C++ uses)."""
    X = np.asarray(X, np.float64)
    total = np.zeros(len(X))
    rows = np.arange(len(X))
    for t in trees:
        node = np.zeros(len(X), np.int64)
        while True:
            f = t["feature"][node]
            live = f >= 0
            if not live.any():
                break
            x = X[rows, np.where(live, f, 0)]
            thr = t["threshold"][node]
            with np.errstate(invalid="ignore"):
                below = (x < thr) if compare == "lt" else (x <= thr)
            go_left = np.where(np.isnan(x), t["missing_left"][node], below)
            nxt = np.where(go_left, t["left"][node], t["right"][node])
            node = np.where(live, nxt, node)
        total += t["value"][node]
    return total


def predict(trees, compare, base, X) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(base + tree_sum(trees, compare, X))))


# ─────────────────────────────────────────────────────────────────────────────
# Early exit: stop evaluating trees for returns that are clearly clean
# ─────────────────────────────────────────────────────────────────────────────

EXIT_MIN_PROB = 0.05     # returns the full model puts at or above this are protected
EXIT_SAFETY = 1.0        # extra room under the lowest protected return, in log-odds
EXIT_FRACTIONS = (1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3)


def staged_margins(model, X, counts) -> dict:
    """{k: base + sum of the first k trees} for every row of X."""
    if type(model).__name__ == "XGBClassifier":
        import xgboost as xgb
        booster, data = model.get_booster(), xgb.DMatrix(X)
        return {k: booster.predict(data, output_margin=True, iteration_range=(0, k))
                for k in counts}
    trees, compare, base = extract_trees(model)
    return {k: base + tree_sum(trees[:k], compare, X) for k in counts}


def calibrate_early_exit(model, X, max_rows: int = 400_000) -> list:
    """
    Checkpoints [(trees evaluated, give up below this running sum), ...] for
    the C++ filter's early exit, measured on the rows X (features of real
    returns; no labels needed, the reference is the full model itself).

    At each checkpoint the cutoff is the lowest running sum of any return the
    full model ends up putting at EXIT_MIN_PROB or more, minus EXIT_SAFETY.
    So on X, every return the full model would flag at a threshold of
    EXIT_MIN_PROB or higher survives every checkpoint. On other data that is
    very likely, not guaranteed.
    """
    n_trees = (model.get_booster().num_boosted_rounds()
               if type(model).__name__ == "XGBClassifier" else len(model._predictors))
    if len(X) > max_rows:
        X = X[np.random.RandomState(0).choice(len(X), max_rows, replace=False)]
    elif type(model).__name__ != "XGBClassifier" and len(X) > 60_000:
        X = X[np.random.RandomState(0).choice(len(X), 60_000, replace=False)]
    counts = sorted({int(round(f * n_trees)) for f in EXIT_FRACTIONS} - {0, n_trees})
    stages = staged_margins(model, X, counts + [n_trees])
    protect = stages[n_trees] >= np.log(EXIT_MIN_PROB / (1 - EXIT_MIN_PROB))
    if protect.sum() < 50:
        return []                       # too few to say where the limit is
    out, stopped = [], 0.0
    for k in counts:
        cut = float(stages[k][protect].min() - EXIT_SAFETY)
        share = float((stages[k] < cut).mean())
        if share > stopped + 0.02:      # only checkpoints that stop something more
            out.append((k, cut))
            stopped = share
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def _nums(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def write_model(bundle: dict, path: str, variant: str = "") -> dict:
    """
    Writes `bundle` (as saved by train_artifact_gbdt.py) to `path`. The sensor
    profile and feature settings currently configured must be the ones the
    model was trained with (config.model_problem() checks that).
    Returns {"trees", "nodes", "max_abs_error"} where the error is the largest
    difference between the model and the exported trees on random inputs.
    """
    model = bundle["model"]
    trees, compare, base = extract_trees(model)
    n_feat = len(bundle["feature_names"])

    # Self-check on random rows with missing values, so a library change that
    # breaks the export is caught here and not in the field.
    rs = np.random.RandomState(0)
    X = rs.normal(0, 1.5, (4000, n_feat)).astype(np.float32)
    X[rs.rand(*X.shape) < 0.2] = np.nan
    X[:, 0] = np.abs(X[:, 0]) * 3
    # ... and rows whose values sit exactly on split values, where a wrong
    # comparison rule or a rounded threshold shows.
    by_feature = {}
    for t in trees:
        for f, thr in zip(t["feature"], t["threshold"]):
            if f >= 0 and np.isfinite(thr):
                by_feature.setdefault(int(f), []).append(thr)
    for f, values in by_feature.items():
        rows = rs.rand(len(X)) < 0.5
        X[rows, f] = rs.choice(np.array(values), int(rows.sum())).astype(np.float32)
    err = float(np.abs(predict(trees, compare, base, X)
                       - model.predict_proba(X)[:, 1]).max())
    if err > 1e-4:
        raise RuntimeError(f"exported trees disagree with the model (max difference "
                           f"{err:.3g}); the export does not understand this "
                           f"version of {type(model).__name__}")

    spec = bundle.get("feature_spec", af.LEGACY_SPEC)
    lines = [
        f"lidar_artifact_model {FORMAT_VERSION}",
        f"variant {variant or ('full' if bundle['use_intensity'] else 'xyz')}",
        f"trained {bundle.get('trained', '-')}",
        f"sensor_fingerprint {sg.fingerprint()}",
        f"threshold {float(bundle.get('default_threshold', 0.5))!r}",
        f"cols {sg.N_COLS}",
        f"layer_elevations_deg {sg.N_RINGS} {_nums(sg.LAYER_ELEV_DEG)}",
        f"layer_azimuth0_deg {sg.N_RINGS} {_nums(sg.LAYER_AZ0_DEG)}",
        f"scan_rate_hz {sg.SCAN_RATE_HZ!r}",
        f"use_intensity {int(bool(bundle['use_intensity']))}",
        f"intensity_field {sg.INTENSITY_FIELD or '-'}",
        f"intensity_max {sg.INTENSITY_MAX!r}",
        f"min_range_m {float(spec['min_range_m'])!r}",
        f"ring_offsets {len(spec['ring_offsets'])} {' '.join(map(str, spec['ring_offsets']))}",
        f"col_offsets {len(spec['col_offsets'])} {' '.join(map(str, spec['col_offsets']))}",
        f"close_log {float(spec['close_log'])!r}",
        f"far_log {float(spec['far_log'])!r}",
        f"clip {float(spec['clip'])!r}",
        f"max_dt_ns {int(spec['max_dt_ns'])}",
        f"time_offsets {len(bundle['time_offsets'])} "
        f"{' '.join(str(int(d)) for d in bundle['time_offsets'])}".rstrip(),
        f"n_features {n_feat}",
        f"compare {compare}",
        f"base_margin {base!r}",
        "early_exit " + " ".join([str(len(bundle.get("early_exit") or []))] + [
            f"{int(k)} {float(c)!r}" for k, c in (bundle.get("early_exit") or [])]),
        f"trees {len(trees)} {sum(len(t['feature']) for t in trees)}",
    ]
    for t in trees:
        lines.append(f"tree {len(t['feature'])}")
        for i in range(len(t["feature"])):
            lines.append(f"{int(t['feature'][i])} {float(t['threshold'][i])!r} "
                         f"{int(t['left'][i])} {int(t['right'][i])} "
                         f"{int(t['missing_left'][i])} {float(t['value'][i])!r}")
    lines.append("end")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return {"trees": len(trees), "nodes": sum(len(t["feature"]) for t in trees),
            "max_abs_error": err}


# ─────────────────────────────────────────────────────────────────────────────
# Check file: scans and the probabilities Python computes for them
# ─────────────────────────────────────────────────────────────────────────────

def write_check_file(bundle: dict, bag: str, topic: str | None, path: str,
                     skip: int, n_scans: int) -> None:
    """
    Binary, little endian:
        "LACHECK1"  int32 n_scans  int32 rings  int32 cols  int32 has_intensity
        per scan:   int64 stamp_ns  int32 n_points
                    float32 xyz[n_points * 3]
                    float32 intensity[n_points]      (if has_intensity)
                    float32 prob[rings * cols]       Python's answer per grid cell
                    float32 point_prob[n_points]     and per point (-1 = not classified)
        at the end: int32 n_features
                    float32 features[rings * cols * n_features]   the classifier
                    inputs of the last scan, NaN rows for cells without a return
    The scans are consecutive, so a filter fed with them in order sees the
    same earlier scans Python did.
    """
    offsets = tuple(bundle["time_offsets"])
    if any(d > 0 for d in offsets):
        raise ValueError("the check file is for realtime models (no later scans); "
                         "this one uses time offsets " + str(list(offsets)))
    scans = []
    for k, (_, hdr_t, _, pts) in enumerate(sg.iter_scans(bag, topic)):
        if k < skip:
            continue
        scans.append((hdr_t, np.array(pts)))
        if len(scans) >= n_scans:
            break
    if not scans:
        raise ValueError("no scans read")
    grids = [sg.scan_to_grid(p) for _, p in scans]
    has_int = bundle["use_intensity"]
    if has_int and grids[0][1] is None:
        raise ValueError("the model needs intensity and this bag has none")
    range_mm = np.stack([g[0] for g in grids])
    hdr = np.array([t for t, _ in scans], np.int64)
    idx = np.arange(len(scans))
    rng = af.to_metres(range_mm)
    stack = af.gather_time_stack(range_mm, hdr, idx, offsets)
    inten = np.stack([g[1] for g in grids]) if has_int else None
    feats = af.compute_features(rng, stack, inten, offsets)
    cand = af.candidates(rng)
    prob = np.zeros(rng.shape, np.float32)
    prob[cand] = bundle["model"].predict_proba(feats[cand])[:, 1]

    with open(path, "wb") as f:
        f.write(b"LACHECK1")
        f.write(struct.pack("<4i", len(scans), sg.N_RINGS, sg.N_COLS, int(has_int)))
        for i, (hdr_t, p) in enumerate(scans):
            ring, col, r, ok = sg.cell_of_points(p["x"], p["y"], p["z"])
            flat = ring * sg.N_COLS + col
            nearest = np.full(sg.N_RINGS * sg.N_COLS, np.inf)
            np.minimum.at(nearest, flat[ok], r[ok])
            scored = ok & (r >= af.MIN_RANGE_M) & (r <= nearest[flat])
            pp = np.full(len(p), -1.0, np.float32)
            pp[scored] = prob[i, ring[scored], col[scored]]
            f.write(struct.pack("<qi", int(hdr_t), len(p)))
            f.write(np.stack([p["x"], p["y"], p["z"]], axis=1).astype("<f4").tobytes())
            if has_int:
                f.write(p[sg.INTENSITY_FIELD].astype("<f4").tobytes())
            f.write(prob[i].astype("<f4").tobytes())
            f.write(pp.astype("<f4").tobytes())
        last = feats[-1].astype("<f4").copy()
        last[~cand[-1]] = np.nan
        f.write(struct.pack("<i", last.shape[-1]))
        f.write(last.tobytes())
    print(f"check file: {len(scans)} scans, {int(cand.sum())} classified returns, "
          f"{int((prob >= bundle.get('default_threshold', 0.5)).sum())} flagged -> {path}")


def main():
    import joblib
    import config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("xyz", "full"), default="xyz",
                    help="which installed model (xyz = range only, full = with intensity)")
    ap.add_argument("--model", default=None, help="a .joblib file instead of the installed one")
    ap.add_argument("-o", "--output", default=None,
                    help="default: next to the model, <name>.model.txt")
    ap.add_argument("--check-file", default=None, metavar="FILE")
    ap.add_argument("--bag", default=None, help="recording for --check-file")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--skip", type=int, default=200, help="scans to skip at the start of the bag")
    ap.add_argument("--scans", type=int, default=12)
    config.add_arguments(ap)
    args = ap.parse_args()
    cfg = config.from_args(args)
    src = args.model or os.path.join(cfg["paths"]["models"], f"artifact_{args.variant}.joblib")
    if not os.path.exists(src):
        sys.exit(f"Model not found: {src}")
    bundle = joblib.load(src)
    problem = config.model_problem(bundle)
    if problem:
        sys.exit(f"{src} {problem}.")
    out = args.output or src[:-len(".joblib")] + ".model.txt"
    info = write_model(bundle, out)
    kind = ("uses earlier scans only (realtime)" if all(d < 0 for d in bundle["time_offsets"])
            else "NEEDS LATER SCANS: not usable on a live sensor, train a realtime model")
    print(f"{out}: {info['trees']} trees, {info['nodes']} nodes, "
          f"{len(bundle['feature_names'])} features, {kind}; "
          f"export matches the model to {info['max_abs_error']:.1e}")
    if args.check_file:
        if not args.bag:
            sys.exit("--check-file needs --bag")
        try:
            write_check_file(bundle, args.bag,
                             sg.resolve_topic(args.bag, args.topic, cfg["default_topic"]),
                             args.check_file, args.skip, args.scans)
        except ValueError as e:
            sys.exit(str(e))


if __name__ == "__main__":
    main()
