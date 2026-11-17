#!/usr/bin/env python3
"""
train_artifact_gbdt.py

Trains the per-return lidar artifact classifier (gradient-boosted trees on
the features in artifact_features.py) from the annotation ledger.

Two models are produced:

    artifact_xyz.joblib    range-only features. Works on any bag, including
                           the telemetry-reduced ones (x, y, z only).
    artifact_full.joblib   the same plus intensity. Needs a bag with an
                           intensity field. Only trained if --intensity-bags
                           supplies the raw recordings that overlap the
                           annotated (reduced) bags; scans are paired by
                           header stamp.

By default a return is judged with the two scans before and the two after
it. A model for a live sensor cannot wait for later scans: train it with
--time-offsets -2 -1 (the app's "realtime" choice) and it uses earlier scans
only. The offsets are stored in the model, and everything that applies it
(classify_bag.py, the app) follows them.

Each model is cross-validated first (5 folds, grouped into 10 s chunks of
bag time so near-identical neighbouring scans never straddle a fold), then
refit on everything. The cross-validation report is written next to the
model as <model>.report.json and printed.

Usage (ROS 2 environment with rosbag2_py, plus scikit-learn or XGBoost):

    python3 train_artifact_gbdt.py --project <project dir> \
        --bags <annotated bag> ... \
        [--intensity-bags <overlapping raw bag> ...] \
        [--exclude-sessions <session id> ...] \
        [--exclude-scans <bag basename>:<scan_stamp_ns> ...]

Paths, the sensor profile and the defaults of the training options come from
the project settings (config.py); the flags here override them.

Label semantics: a scan in the ledger is reviewed
(selected cells are artifacts, every other return is clean), a scan absent
from the ledger is unknown and never used.

The reviewed scans were not all labelled exhaustively: many carry only the
one to three most obvious points. Unclicked returns in those scans are not
reliably clean, and using them as negatives measurably hurts. By default the models are therefore trained only on scans with
at least --min-labels selections, plus scans deliberately saved with none.
The remaining, partly labelled scans are scored afterwards as a check: the
model has never seen them, so the share of their labels it recovers is an
honest recall figure. Pass --min-labels 1 to train on every reviewed scan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact_features as af
import config
import scan_grid as sg

GROUP_NS = 10_000_000_000      # CV group, set from training.cv_group_s
SCAN_KEY_PER_BAG = 10_000_000  # scan_key = scan index + this * bag index


def cache_name(bag_path: str) -> str:
    return os.path.splitext(os.path.basename(bag_path))[0]


def ensure_cache(bag_path: str, cache_dir: str, topic) -> dict:
    """topic: (explicit topic or None, preferred topic or None)."""
    name = cache_name(bag_path)
    if not os.path.exists(sg.cache_paths(cache_dir, name)["range_mm"]):
        print(f"Caching scans of {os.path.basename(bag_path)} ...", flush=True)
        sg.cache_bag(bag_path, cache_dir, name, sg.resolve_topic(bag_path, *topic))
    return sg.load_cache(cache_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_ledger(path: str):
    """
    Returns {(bag_basename, stamp_ns): set of (ring, col)} with last snapshot
    authoritative (same rules as the annotator), and
    {(bag_basename, stamp_ns): set of session ids that wrote to that scan},
    and {(bag_basename, stamp_ns): bool} for scans whose last snapshot says
    explicitly whether every error in it was labelled ("complete", written
    by annotator_app.py; the Foxglove annotator does not write it).
    """
    ann, snapshotted, touched, complete = {}, set(), {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bag, stamp = rec.get("bag"), rec.get("scan_stamp_ns")
            if bag is None or stamp is None:
                continue
            key = (bag, stamp)
            touched.setdefault(key, set()).add(rec.get("session", ""))
            if "selections" in rec:
                ann[key] = {(int(s[0]), int(s[1])) for s in rec["selections"]
                            if isinstance(s, list) and len(s) >= 2}
                snapshotted.add(key)
                if "complete" in rec:
                    complete[key] = bool(rec["complete"])
                else:
                    complete.pop(key, None)
            elif "ring" in rec and "azimuth_idx" in rec and key not in snapshotted:
                ann.setdefault(key, set()).add((int(rec["ring"]), int(rec["azimuth_idx"])))
    return ann, touched, complete


def build_bag_dataset(bag_path, ann, touched, complete, cache_dir, topic, time_offsets,
                      intensity_caches, exclude_sessions, exclude_scans,
                      mirror=False):
    """
    Features and labels for every annotated scan of one reduced bag. With
    mirror, every scan is added a second time flipped left-right (rows marked
    in "aug"), so an error learned on one side of the robot is also known on
    the other.
    """
    base = os.path.basename(bag_path)
    cache = ensure_cache(bag_path, cache_dir, topic)
    rec2i = {t: i for i, t in enumerate(cache["rec_t"].tolist())}

    idx, labels, explicit, n_empty, n_excl, n_missing = [], [], [], 0, 0, 0
    for (bag, stamp), sels in sorted(ann.items()):
        if bag != base:
            continue
        if (touched[(bag, stamp)] & exclude_sessions) or (bag, stamp) in exclude_scans:
            n_excl += 1
            continue
        i = rec2i.get(stamp)
        if i is None:
            n_missing += 1
            continue
        lab = np.zeros((sg.N_RINGS, sg.N_COLS), bool)
        for ring, col in sels:
            if cache["range_mm"][i, ring, col] >= af.MIN_RANGE_M * 1000:
                lab[ring, col] = True
            else:
                n_empty += 1
        idx.append(i)
        labels.append(lab)
        explicit.append(int(complete[(bag, stamp)]) if (bag, stamp) in complete else -1)
    if not idx:
        return None
    idx = np.array(idx)
    lab = np.stack(labels)
    explicit = np.array(explicit, np.int8)

    rng = af.to_metres(np.asarray(cache["range_mm"][idx]))
    stack = af.gather_time_stack(cache["range_mm"], cache["hdr_t"], idx, time_offsets)

    # Intensity from an overlapping raw recording, paired by header stamp.
    inten = np.full(rng.shape, np.nan, np.float32)
    has_int = np.zeros(len(idx), bool)
    hdr = cache["hdr_t"][idx]
    for raw in intensity_caches:
        j = np.searchsorted(raw["hdr_t"], hdr).clip(0, len(raw["hdr_t"]) - 1)
        match = (raw["hdr_t"][j] == hdr) & ~has_int
        if not match.any():
            continue
        jm = j[match]
        raw_rng = af.to_metres(np.asarray(raw["range_mm"][jm]))
        agree = np.mean(np.isfinite(raw_rng) == np.isfinite(rng[match]))
        if agree < 0.999:
            print(f"  WARNING: raw/reduced grids disagree ({agree:.4f}); "
                  f"not using this intensity source for {base}")
            continue
        inten[match] = np.asarray(raw["intensity"][jm])
        has_int |= match

    t = cache["hdr_t"][idx]
    parts = []
    for flip in ((False, True) if mirror else (False,)):
        f = (lambda a: a[..., ::-1].copy()) if flip else (lambda a: a)
        r = f(rng)
        cand = af.candidates(r)
        s, _, _ = np.nonzero(cand)
        parts.append({
            "X": af.compute_features(
                r, {dt: f(a) for dt, a in stack.items()}, None, time_offsets)[cand],
            "Xi": af.intensity_features(r, f(inten))[cand],
            "y": f(lab)[cand].astype(np.int8),
            "has_int": has_int[s], "scan": idx[s], "explicit": explicit[s],
            "group": (t[s] - t.min()) // GROUP_NS,
            "stamp": cache["rec_t"][idx][s].astype(np.int64),
            "aug": np.full(len(s), flip),
        })
    cand = af.candidates(rng)
    print(f"  {base}: {len(idx)} scans ({int(has_int.sum())} with intensity), "
          f"{int(lab[cand].sum())} artifact / {int(cand.sum())} returns"
          f"{f', {n_excl} scans excluded' if n_excl else ''}"
          f"{f', {n_empty} selections on empty cells ignored' if n_empty else ''}"
          f"{f', {n_missing} ledger stamps not in bag' if n_missing else ''}")
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cpu"                 # set by --device; "cuda" trains with XGBoost
PROGRESS = False               # set by --progress; see emit()
N_ROUNDS = 600                 # set from training.rounds
CURVE_EVERY = 20


def emit(**event):
    """One machine-readable line per event for annotator_app.py to follow."""
    if PROGRESS:
        print("PROGRESS " + json.dumps(event), flush=True)


def pick_device(requested: str) -> str:
    """'cuda' when XGBoost with CUDA and a GPU are both there, else 'cpu'."""
    if requested == "cpu":
        return "cpu"
    try:
        import xgboost as xgb
        ok = xgb.build_info().get("USE_CUDA", False)
        if ok:
            xgb.XGBClassifier(n_estimators=1, device="cuda").fit(
                np.zeros((4, 1), np.float32), np.array([0, 1, 0, 1]))
    except Exception as e:      # no xgboost, no driver, no GPU
        ok = False
        why = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
    else:
        why = "XGBoost was built without CUDA"
    if not ok and requested == "cuda":
        sys.exit(f"--device cuda is not usable here ({why}).")
    return "cuda" if ok else "cpu"


def fit_model(X, y, seed=0, neg_per_pos=60, min_neg=200_000, eval_set=None,
              on_round=None):
    """
    Negatives are subsampled (they outnumber positives ~700:1 and are highly
    redundant) and up-weighted by the same factor, so predicted probabilities
    stay calibrated to the true base rate.

    On a GPU the trees are grown by XGBoost with matching settings; on CPU by
    scikit-learn. Both take NaN as "missing" and expose predict_proba.

    eval_set = (X, y) held out from this fit; on_round(round, value) is then
    called every CURVE_EVERY trees with the area under the precision-recall
    curve on it (value is None without an eval_set). On CPU scikit-learn has
    no per-tree hook, so the curve is replayed once the fit is finished.
    """

    rng = np.random.default_rng(seed)
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    k = min(len(neg), max(neg_per_pos * len(pos), min_neg))
    neg_s = rng.choice(neg, k, replace=False)
    sel = np.concatenate([pos, neg_s])
    w = np.concatenate([np.ones(len(pos)), np.full(k, len(neg) / k)])
    if DEVICE == "cuda":
        import xgboost as xgb
        class Report(xgb.callback.TrainingCallback):
            def after_iteration(self, booster, epoch, evals_log):
                if on_round and ((epoch + 1) % CURVE_EVERY == 0 or epoch + 1 == N_ROUNDS):
                    vals = [v for d in evals_log.values() for v in d.values()]
                    on_round(epoch + 1, float(vals[0][-1]) if vals else None)
                return False

        model = xgb.XGBClassifier(
            n_estimators=N_ROUNDS, learning_rate=0.03, max_leaves=31, max_depth=0,
            grow_policy="lossguide", tree_method="hist", max_bin=256,
            reg_lambda=1.0, random_state=seed, device="cuda",
            eval_metric="aucpr" if eval_set else None, callbacks=[Report()])
        model.fit(X[sel], y[sel], sample_weight=w,
                  eval_set=[eval_set] if eval_set else None, verbose=False)
        model.set_params(callbacks=None)    # not part of the saved model
        model.set_params(device="cpu")      # saved model predicts anywhere
        return model
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=N_ROUNDS, learning_rate=0.03, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=seed)
    model.fit(X[sel], y[sel], sample_weight=w)
    if on_round:
        if eval_set and eval_set[1].any():
            from sklearn.metrics import average_precision_score
            for k, pk in enumerate(model.staged_predict_proba(eval_set[0]), 1):
                if k % (CURVE_EVERY * 5) == 0 or k == N_ROUNDS:
                    on_round(k, float(average_precision_score(eval_set[1], pk[:, 1])))
        else:
            on_round(N_ROUNDS, None)
    return model


def cross_validate(X, y, groups, aug, name=""):
    """
    Out-of-fold predictions; mirrored rows train but are not predicted. Also
    returns one learning curve per fold: [[trees, PR-AUC on the fold], ...].
    """
    from sklearn.model_selection import GroupKFold

    p = np.zeros(len(y), np.float32)
    n_splits = min(5, len(np.unique(groups)))
    curves = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, groups), 1):
        te = te[~aug[te]]
        curve = []

        def on_round(r, v, k=k, curve=curve):
            if v is not None:
                curve.append([r, round(v, 4)])
            emit(ev="fit", variant=name, fold=k, folds=n_splits, round=r,
                 rounds=N_ROUNDS, value=v)

        model = fit_model(X[tr], y[tr], on_round=on_round,
                          eval_set=(X[te], y[te]) if y[te].any() else None)
        p[te] = model.predict_proba(X[te])[:, 1]
        curves.append(curve)
    return p, curves


def summarise(y, p, scan_key):
    from sklearn.metrics import average_precision_score

    n_scans = len(np.unique(scan_key))
    out = {
        "n_returns": int(len(y)), "n_artifact": int(y.sum()), "n_scans": n_scans,
        "base_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "thresholds": [],
    }
    for th in (0.2, 0.3, 0.5, 0.7, 0.9):
        flag = p >= th
        tp = int((flag & (y == 1)).sum())
        out["thresholds"].append({
            "threshold": th,
            "precision_vs_labels": tp / max(int(flag.sum()), 1),
            "recall": tp / max(int(y.sum()), 1),
            "flagged_per_scan": float(flag.sum() / n_scans),
            "flagged_unlabelled": int((flag & (y == 0)).sum()),
        })
    # Precision and recall at every 0.02 of threshold, for plotting.
    edges = np.linspace(0, 1, 51)
    above = lambda v: np.cumsum(np.histogram(v, edges)[0][::-1])[::-1]
    n_flag, n_tp = above(p), above(p[y == 1])
    out["pr_curve"] = [[round(float(edges[k]), 2), n_tp[k] / max(n_flag[k], 1),
                        n_tp[k] / max(int(y.sum()), 1)] for k in range(1, 50)]
    return out


def summarise_per_bag(y, p, scan_key, bag_names):
    """summarise() for each bag that has labelled artifacts in these rows."""
    out = {}
    bag = scan_key // SCAN_KEY_PER_BAG
    for b in np.unique(bag):
        m = bag == b
        if y[m].any():
            out[bag_names[int(b)]] = summarise(y[m], p[m], scan_key[m])
    return out


def print_summary(title, s):
    print(f"\n{title}")
    print(f"  {s['n_scans']} scans, {s['n_artifact']} labelled artifacts of "
          f"{s['n_returns']} returns (base rate {s['base_rate']:.5f})")
    print(f"  average precision: {s['average_precision']:.3f}")
    print("  threshold  precision*  recall  flagged/scan  flagged-but-unlabelled")
    for t in s["thresholds"]:
        print(f"    {t['threshold']:.1f}       {t['precision_vs_labels']:.3f}      "
              f"{t['recall']:.3f}     {t['flagged_per_scan']:6.2f}        "
              f"{t['flagged_unlabelled']}")
    print("  * against the ledger as written. Artifacts that were never clicked "
          "count as false positives here.")


def train_variant(name, X, y, groups, scan_key, aug, explicit, stamp, names, args, meta):
    print(f"\n{'=' * 70}\nModel '{name}': {X.shape[1]} features")
    # A mirrored scan has the same label count as its original.
    _, inv = np.unique(scan_key * 2 + aug, return_inverse=True)
    n_lab = np.bincount(inv, weights=y)[inv]
    use = (n_lab >= args.min_labels) | (n_lab == 0)
    # An explicit complete / not complete mark from the app overrides the guess.
    use = np.where(explicit >= 0, explicit == 1, use)
    if y[use].sum() == 0:
        sys.exit(f"No scan has >= {args.min_labels} labels; lower --min-labels.")

    p, curves = cross_validate(X[use], y[use], groups[use], aug[use], name)
    real = ~aug[use]
    yr, pr, kr = y[use][real], p[real], scan_key[use][real]
    report = {"min_labels": args.min_labels, "mirror": bool(aug.any()),
              "cross_validation": summarise(yr, pr, kr),
              "learning_curves": curves,
              "learning_curve_metric": "area under the precision-recall curve "
                                       "on the held-out fold"}
    report["cross_validation"]["per_bag"] = summarise_per_bag(yr, pr, kr, meta["bags"])

    def stamps_by_bag(mask):
        out = {}
        for b in np.unique(scan_key[mask] // SCAN_KEY_PER_BAG):
            m = mask & (scan_key // SCAN_KEY_PER_BAG == b)
            out[meta["bags"][int(b)]] = np.unique(stamp[m]).tolist()
        return out
    # Which scans the model has seen, so later reviews can be told apart.
    report["trained_scans"] = stamps_by_bag(use)
    cv = report["cross_validation"]
    at = min(cv["thresholds"], key=lambda t: abs(t["threshold"] - args.default_threshold))
    emit(ev="cv", variant=name, ap=cv["average_precision"], threshold=at["threshold"],
         precision=at["precision_vs_labels"], recall=at["recall"],
         scans=cv["n_scans"], labels=cv["n_artifact"])
    print_summary(f"Cross-validation on the training scans "
                  f"(marked complete, or >= {args.min_labels} labels / saved "
                  f"with none where unmarked)",
                  report["cross_validation"])

    model = fit_model(X[use], y[use], on_round=lambda r, v: emit(
        ev="fit", variant=name, fold=0, round=r, rounds=N_ROUNDS, value=None))

    held = ~use & ~aug
    if held.any():
        # Partly labelled scans: never trained on. Their labels measure
        # recall; "flagged-but-unlabelled" there is mostly unclicked points.
        ph = model.predict_proba(X[held])[:, 1]
        report["partly_labelled_scans"] = summarise(y[held], ph, scan_key[held])
        report["partly_labelled_scans"]["per_bag"] = summarise_per_bag(
            y[held], ph, scan_key[held], meta["bags"])
        print_summary("Partly labelled scans, not trained on (recall is "
                      "meaningful, precision is not)", report["partly_labelled_scans"])

    import joblib
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, f"artifact_{name}.joblib")
    joblib.dump({
        "model": model,
        "feature_names": names,
        "time_offsets": list(args.time_offsets),
        "use_intensity": name == "full",
        "min_range_m": af.MIN_RANGE_M,
        "feature_spec": af.spec(),
        "sensor_fingerprint": sg.fingerprint(),
        "sensor_name": sg.PROFILE.get("name", ""),
        "default_threshold": args.default_threshold,
        "trained_on": DEVICE,
        "trained": datetime.now(timezone.utc).isoformat(),
        **meta,
    }, path)
    with open(path.replace(".joblib", ".report.json"), "w") as f:
        json.dump({"model": name, "features": names, **meta, **report}, f, indent=1)
    print(f"\n  saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", required=True, nargs="+",
                    help="annotated bags, basenames must match the ledger")
    ap.add_argument("--intensity-bags", nargs="*", default=[],
                    help="raw recordings overlapping the annotated bags")
    ap.add_argument("--topic", default=None,
                    help="lidar topic, for bags that are not cached yet "
                         "(default: the project's default_topic, or the bag's "
                         "only PointCloud2 topic)")
    ap.add_argument("--time-offsets", type=int, nargs="*", default=None,
                    help="neighbouring scans used as context. Use '-2 -1' for a "
                         "causal model, or none for a single-scan model "
                         "(default: features.time_offsets)")
    ap.add_argument("--exclude-sessions", nargs="*", default=[],
                    help="drop every scan any of these annotator sessions wrote to")
    ap.add_argument("--exclude-scans", nargs="*", default=[],
                    help="<bag basename>:<scan_stamp_ns> to drop")
    ap.add_argument("--min-labels", type=int, default=None,
                    help="train only on scans with at least this many labels "
                         "(or saved with none); 1 = every reviewed scan "
                         "(default: training.min_labels)")
    ap.add_argument("--default-threshold", type=float, default=0.5)
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                    help="also train on every scan flipped left-right "
                         "(default: training.mirror)")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None,
                    help="auto = GPU (XGBoost) when available, else CPU "
                         "(scikit-learn) (default: training.device)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="boosting rounds (default: training.rounds)")
    ap.add_argument("--progress", action="store_true",
                    help="print machine-readable PROGRESS lines (for the app)")
    config.add_arguments(ap)
    args = ap.parse_args()
    cfg = config.from_args(args)
    tr = cfg["training"]
    args.ledger, args.cache_dir = cfg["paths"]["ledger"], cfg["paths"]["cache"]
    args.model_dir = cfg["paths"]["models"]
    args.topic = (args.topic, cfg["default_topic"])
    args.time_offsets = tuple(af.DEFAULT_TIME_OFFSETS if args.time_offsets is None
                              else args.time_offsets)
    for k in ("min_labels", "mirror", "device"):
        if getattr(args, k) is None:
            setattr(args, k, tr[k])
    global DEVICE, PROGRESS, N_ROUNDS, GROUP_NS
    N_ROUNDS = int(args.rounds or tr["rounds"])
    GROUP_NS = int(float(tr["cv_group_s"]) * 1e9)
    PROGRESS = args.progress
    DEVICE = pick_device(args.device)
    emit(ev="start", device=DEVICE, rounds=N_ROUNDS)
    print(f"Training on: {DEVICE} "
          f"({'XGBoost' if DEVICE == 'cuda' else 'scikit-learn'})")

    ann, touched, complete = load_ledger(args.ledger)
    excl_sessions = set(args.exclude_sessions)
    excl_scans = set()
    for item in args.exclude_scans:
        bag, stamp = item.rsplit(":", 1)
        excl_scans.add((bag, int(stamp)))

    int_caches = [ensure_cache(b, args.cache_dir, args.topic) for b in args.intensity_bags]
    int_caches = [c for c in int_caches if c["intensity"] is not None]

    given = {os.path.basename(b) for b in args.bags}
    for bag in sorted({k[0] for k in ann} - given):
        print(f"NOTE: ledger has annotations for '{bag}', which was not passed "
              f"in --bags; those scans are not used.")

    print("Building dataset:")
    parts = []
    for bi, bag in enumerate(args.bags):
        emit(ev="dataset", bag=os.path.basename(bag), k=bi + 1, of=len(args.bags))
        d = build_bag_dataset(bag, ann, touched, complete, args.cache_dir, args.topic,
                              args.time_offsets, int_caches, excl_sessions, excl_scans,
                              args.mirror)
        if d is None:
            print(f"  {os.path.basename(bag)}: no annotated scans")
            continue
        d["group"] = d["group"] + bi * 1_000_000
        d["scan_key"] = d["scan"].astype(np.int64) + bi * SCAN_KEY_PER_BAG
        parts.append(d)
    if not parts:
        sys.exit("No annotated scans found for the given bags.")
    cat = lambda k: np.concatenate([d[k] for d in parts])
    X, Xi, y = cat("X"), cat("Xi"), cat("y")
    groups, scan_key, has_int = cat("group"), cat("scan_key"), cat("has_int")
    aug, explicit, stamp = cat("aug"), cat("explicit"), cat("stamp")
    emit(ev="variants", names=["xyz", "full"] if has_int.any() else ["xyz"])

    meta = {
        "bags": [os.path.basename(b) for b in args.bags],
        "excluded_sessions": sorted(excl_sessions),
        "excluded_scans": sorted(f"{b}:{s}" for b, s in excl_scans),
        "sensor": sg.PROFILE.get("name", ""),
        "rounds": N_ROUNDS,
        "time_offsets": list(args.time_offsets),
    }
    names = af.feature_names(args.time_offsets, False)
    train_variant("xyz", X, y, groups, scan_key, aug, explicit, stamp, names, args, meta)

    if has_int.any():
        m = has_int
        names_full = af.feature_names(args.time_offsets, True)
        train_variant("full", np.concatenate([X[m], Xi[m]], axis=1), y[m],
                      groups[m], scan_key[m], aug[m], explicit[m], stamp[m], names_full, args, meta)
    else:
        print("\nNo intensity source matched the annotated scans; "
              "only the xyz model was trained.")


if __name__ == "__main__":
    main()
