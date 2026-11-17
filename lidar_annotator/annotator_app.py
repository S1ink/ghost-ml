#!/usr/bin/env python3
"""
annotator_app.py

One local web application for the lidar artifact work: register bags, review
and correct the model's proposals scan by scan (range image and 3D view),
mark scans complete, retrain, and export classified bags.

Run with run.sh next to this file, or directly in a ROS 2 Python environment
(rosbag2_py), then open http://localhost:8765

    python3 annotator_app.py --project <project dir> [--bags-root <folder>]
        [--sensor multiscan | --sensor-from-bag <bag>] [--port 8765]

The project directory holds the ledger, the bag registry, the scan cache,
the models and the exports, and optionally annotator.json with the settings
(see config.py and README.md). Nothing here is specific to one machine,
robot or lidar: the grid geometry is a sensor profile (scan_grid.py).

Besides ROS for reading bags, serving needs only the standard library, numpy
and the modules next to this file; training needs scikit-learn or XGBoost.

Ledger: append-only JSONL, one snapshot per saved scan,

    {"bag": ..., "scan_stamp_ns": ..., "session": "app_...",
     "selections": [[ring, col, range_m], ...], "complete": true|false,
     "saved_unix": 1794850000}

"complete" says every error in the scan is selected, so the rest of the scan
can be trained on as clean. train_artifact_gbdt.py trains on complete scans
and uses the others only as a recall check. Snapshots without the key (from
other tools) count as reviewed with completeness unknown.

Multi-echo bags: the grid holds the nearest return of each cell (see
scan_grid.py), and that is the return shown and labelled here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import artifact_features as af          # noqa: E402
import config                           # noqa: E402
import scan_grid as sg                  # noqa: E402

SESSION_ID = datetime.now().strftime("app_%Y%m%d_%H%M%S")


def cache_name(bag_path: str) -> str:
    return os.path.splitext(os.path.basename(bag_path))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Jobs: caching, queue scoring, training and export run in the background
# ─────────────────────────────────────────────────────────────────────────────

class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[int, dict] = {}
        self.next_id = 1

    def start(self, kind: str, label: str, fn) -> int:
        with self.lock:
            jid = self.next_id
            self.next_id += 1
            job = {"id": jid, "kind": kind, "label": label, "status": "running",
                   "log": [], "started": time.time(), "ended": None, "progress": None}
            self.jobs[jid] = job

        def log(line: str):
            job["log"].append(line.rstrip("\n"))
            del job["log"][:-400]

        def run():
            try:
                fn(log, job)
                job["status"] = "done"
            except Exception as e:      # shown in the UI
                log(traceback.format_exc())
                log(f"FAILED: {e}")
                job["status"] = "failed"
            job["ended"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        return jid

    def running(self, kind: str, label: str | None = None) -> dict | None:
        for j in self.jobs.values():
            if j["status"] == "running" and j["kind"] == kind and label in (None, j["label"]):
                return j
        return None

    def listing(self) -> list:
        out = []
        for j in sorted(self.jobs.values(), key=lambda j: -j["id"])[:12]:
            out.append({**{k: j[k] for k in ("id", "kind", "label", "status")},
                        "seconds": round((j["ended"] or time.time()) - j["started"]),
                        "log": j["log"][-40:], "progress": j["progress"]})
        return out


def run_process(cmd: list, log, on_event=None) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in proc.stdout:
        if "Warning" in line or line.strip().startswith(("warnings.warn", "return func")):
            continue
        if line.startswith("PROGRESS "):
            if on_event:
                try:
                    on_event(json.loads(line[9:]))
                except Exception:
                    traceback.print_exc()
            continue
        log(line)
    if proc.wait() != 0:
        raise RuntimeError(f"exit code {proc.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        p = cfg["paths"]
        self.args = argparse.Namespace(
            ledger=p["ledger"], registry=p["registry"], cache_dir=p["cache"],
            model_dir=p["models"], out_dir=p["out"], bags_root=cfg["bags_root"],
            topic=cfg["default_topic"])
        self.exts = tuple(cfg["bag_extensions"])
        self.jobs = Jobs()
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.caches: dict[str, dict] = {}
        self.models: dict[str, tuple] = {}      # variant -> (mtime, bundle)
        self.model_problems: dict[str, tuple] = {}
        self.reports: dict[str, tuple] = {}     # variant -> (mtime, report)
        self.stats_cache: dict[tuple, dict] = {}
        self.pred_cache: dict[tuple, np.ndarray] = {}
        self.ledger_version = 0
        self._load_registry()
        self._load_ledger()

    # ── registry ────────────────────────────────────────────────────────────
    def _load_registry(self):
        a = self.args
        if os.path.exists(a.registry):
            with open(a.registry) as f:
                self.reg = json.load(f)
        else:
            # exclude_sessions / exclude_scans: ledger sessions and
            # "<bag name>:<scan_stamp_ns>" scans that training skips.
            self.reg = {"bags": [], "exclude_sessions": [], "exclude_scans": []}
        for b in self.reg["bags"]:
            b["name"] = os.path.basename(b["path"])

    def _save_registry(self):
        with open(self.args.registry + ".tmp", "w") as f:
            json.dump(self.reg, f, indent=1)
        os.replace(self.args.registry + ".tmp", self.args.registry)

    def _find_bags(self) -> list:
        out = []
        own = {os.path.realpath(self.cfg["paths"][k]) for k in ("cache", "models", "out")}
        for root, dirs, files in os.walk(self.args.bags_root):
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and os.path.realpath(os.path.join(root, d)) not in own]
            out += [os.path.join(root, f) for f in files if f.lower().endswith(self.exts)]
        return sorted(out)

    def _is_bag(self, path: str) -> bool:
        return path.lower().endswith(self.exts) and os.path.isfile(path)

    def bag(self, name: str) -> dict:
        for b in self.reg["bags"]:
            if b["name"] == name:
                return b
        raise KeyError(f"bag '{name}' is not registered")

    def cache(self, name: str) -> dict | None:
        """Grid cache of a registered bag, or None while it is being built."""
        key = cache_name(name)
        if key not in self.caches:
            if not os.path.exists(sg.cache_paths(self.args.cache_dir, key)["rec_t"]):
                return None
            c = sg.load_cache(self.args.cache_dir, key)
            c["rec2i"] = {t: i for i, t in enumerate(c["rec_t"].tolist())}
            self.caches[key] = c
        return self.caches[key]

    def browse(self) -> list:
        have = {b["path"] for b in self.reg["bags"]}
        root = self.args.bags_root
        return [{"path": p, "rel": os.path.relpath(p, root),
                 "mb": round(os.path.getsize(p) / 1e6), "date": name_date(p)}
                for p in self._find_bags() if p not in have]

    def add_bag(self, path: str, topic: str | None) -> int:
        path = os.path.realpath(os.path.expanduser(path.strip()))
        if not self._is_bag(path):
            raise ValueError(f"not a bag file ({', '.join(self.exts)}): {path}")
        name = os.path.basename(path)
        if any(b["name"] == name for b in self.reg["bags"]):
            raise ValueError(f"a bag named {name} is already registered")
        if not topic:
            # The preferred topic when the bag has it, else its busiest one.
            names = [t["name"] for t in sg.pointcloud_topics(path)]
            if not names:
                raise ValueError(f"{name} has no PointCloud2 topic")
            topic = self.args.topic if self.args.topic in names else names[0]
        with self.lock:
            if any(b["name"] == name for b in self.reg["bags"]):
                raise ValueError(f"a bag named {name} is already registered")
            entry = {"path": path, "topic": topic, "name": name}
            self.reg["bags"].append(entry)
            self._save_registry()

        def work(log, job):
            key = cache_name(path)
            if os.path.exists(sg.cache_paths(self.args.cache_dir, key)["rec_t"]):
                log("already cached")
                return
            log(f"reading {path} ...")
            sg.cache_bag(path, self.args.cache_dir, key, entry["topic"])
            log("cached")
        return self.jobs.start("cache", name, work)

    def set_label(self, name: str, label: str):
        with self.lock:
            self.bag(name)["label"] = str(label).strip()[:60]
            self._save_registry()

    def topics(self, path: str) -> list:
        """PointCloud2 topics of a bag, with message counts."""
        path = os.path.realpath(os.path.expanduser(path.strip()))
        if not self._is_bag(path):
            raise ValueError(f"not a bag file ({', '.join(self.exts)}): {path}")
        return sg.pointcloud_topics(path)

    def set_topic(self, name: str, topic: str) -> dict:
        """Read a registered bag again from another topic (rebuilds its cache)."""
        b = self.bag(name)
        if topic == b["topic"]:
            return {"changed": False}
        if topic not in [t["name"] for t in self.topics(b["path"])]:
            raise ValueError(f"{topic} is not a PointCloud2 topic of this bag")
        if any(j["status"] == "running" and (j["kind"] in ("train", "export")
                                             or j["label"] == name)
               for j in self.jobs.jobs.values()):
            raise ValueError("wait for the running jobs to finish first")
        key = cache_name(name)
        with self.lock:
            b["topic"] = topic
            self._save_registry()
            self.caches.pop(key, None)
            self.stats_cache.clear()
            self.pred_cache.clear()
            for f in list(sg.cache_paths(self.args.cache_dir, key).values()) + [
                    os.path.join(self.args.cache_dir, f"{key}_queue.npz")]:
                if os.path.exists(f):
                    os.remove(f)

        def work(log, job):
            log(f"reading {b['path']} ({topic}) ...")
            sg.cache_bag(b["path"], self.args.cache_dir, key, topic)
            log("cached")
        return {"changed": True, "job": self.jobs.start("cache", name, work)}

    def remove_bag(self, name: str):
        with self.lock:
            self.reg["bags"] = [b for b in self.reg["bags"] if b["name"] != name]
            self._save_registry()

    def rotation(self, b: dict) -> list:
        """Lidar frame -> parent frame rotation (row-major 3x3) from /tf_static;
        identity when the bag has none."""
        if "rot" not in b:
            rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            try:
                rot = read_lidar_rotation(b["path"], b["topic"],
                                          self.cfg["lidar_frame"]) or rot
            except Exception as e:
                print(f"tf_static of {b['name']} not readable: {e}")
            with self.lock:
                b["rot"] = rot
                self._save_registry()
        return b["rot"]

    # ── ledger ──────────────────────────────────────────────────────────────
    def _load_ledger(self):
        self.ann: dict[tuple, set] = {}
        self.complete: dict[tuple, bool] = {}
        snapshotted = set()
        if not os.path.exists(self.args.ledger):
            return
        with open(self.args.ledger) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bag, stamp = rec.get("bag"), rec.get("scan_stamp_ns")
                if bag is None or stamp is None:
                    continue
                key = (bag, stamp)
                if "selections" in rec:
                    self.ann[key] = {(int(s[0]), int(s[1])) for s in rec["selections"]
                                     if isinstance(s, list) and len(s) >= 2}
                    snapshotted.add(key)
                    if "complete" in rec:
                        self.complete[key] = bool(rec["complete"])
                    else:
                        self.complete.pop(key, None)
                elif "ring" in rec and "azimuth_idx" in rec and key not in snapshotted:
                    self.ann.setdefault(key, set()).add(
                        (int(rec["ring"]), int(rec["azimuth_idx"])))

    def save_scan(self, name: str, i: int, selections: list, complete: bool) -> dict:
        c = self.cache(name)
        stamp = int(c["rec_t"][i])
        rng = np.asarray(c["range_mm"][i])
        sels = sorted({(int(r), int(col)) for r, col in selections
                       if 0 <= r < sg.N_RINGS and 0 <= col < sg.N_COLS and rng[r, col] > 0})
        rec = {"bag": name, "scan_stamp_ns": stamp, "session": SESSION_ID,
               "selections": [[r, col, round(float(rng[r, col]) / 1000.0, 3)]
                              for r, col in sels],
               "complete": bool(complete), "saved_unix": int(time.time())}
        with self.lock:
            self.ledger_version += 1
            with open(self.args.ledger, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.ann[(name, stamp)] = set(sels)
            self.complete[(name, stamp)] = bool(complete)
        return {"saved": len(sels), "complete": bool(complete)}

    def bag_stats(self, name: str) -> dict:
        scans = [k for k in self.ann if k[0] == name]
        return {"reviewed": len(scans),
                "complete": sum(1 for k in scans if self.complete.get(k)),
                "unmarked": sum(1 for k in scans if k not in self.complete),
                "labels": sum(len(self.ann[k]) for k in scans)}

    def reviewed_indices(self, name: str) -> dict:
        c = self.cache(name)
        out = {}
        if c is None:
            return out
        for (bag, stamp), sels in self.ann.items():
            if bag == name and stamp in c["rec2i"]:
                st = self.complete.get((bag, stamp))
                out[c["rec2i"][stamp]] = [len(sels), -1 if st is None else int(st)]
        return out

    # ── model ───────────────────────────────────────────────────────────────
    def model(self, variant: str):
        import joblib
        path = os.path.join(self.args.model_dir, f"artifact_{variant}.joblib")
        if not os.path.exists(path):
            return None
        mt = os.path.getmtime(path)
        if variant not in self.models or self.models[variant][0] != mt:
            bundle = joblib.load(path)
            problem = config.model_problem(bundle)
            if problem:
                if self.model_problems.get(variant) != (mt, problem):
                    print(f"{path} {problem}; ignored", flush=True)
                self.model_problems[variant] = (mt, problem)
                return None
            bundle["tag"] = f"{variant}_{int(mt)}"
            self.models[variant] = (mt, bundle)
        return self.models[variant][1]

    def model_for(self, c: dict):
        b = self.model("full") if c["intensity"] is not None else None
        return b or self.model("xyz")

    def predict(self, c: dict, idx: np.ndarray):
        """(len(idx), rings, cols) probabilities, or None with no usable model."""
        bundle = self.model_for(c)
        if bundle is None:
            return None, None
        off = tuple(bundle["time_offsets"])
        rng = af.to_metres(np.asarray(c["range_mm"][idx]))
        stack = af.gather_time_stack(c["range_mm"], c["hdr_t"], idx, off)
        inten = np.asarray(c["intensity"][idx]) if bundle["use_intensity"] else None
        feats = af.compute_features(rng, stack, inten, off)
        cand = af.candidates(rng)
        p = np.zeros(rng.shape, np.float32)
        if cand.any():
            with self.model_lock:
                p[cand] = bundle["model"].predict_proba(feats[cand])[:, 1]
        return p, bundle

    def model_info(self) -> dict:
        out = {}
        for v in ("xyz", "full"):
            rp = os.path.join(self.args.model_dir, f"artifact_{v}.report.json")
            b = self.model(v)
            if b is None or not os.path.exists(rp):
                continue
            with open(rp) as f:
                rep = json.load(f)
            out[v] = {"trained": b.get("trained"), "threshold": b.get("default_threshold", 0.5),
                      "summary": summarise_report(rep)}
        return out

    def report(self, variant: str) -> dict | None:
        path = os.path.join(self.args.model_dir, f"artifact_{variant}.report.json")
        if not os.path.exists(path):
            return None
        mt = os.path.getmtime(path)
        if variant not in self.reports or self.reports[variant][0] != mt:
            with open(path) as f:
                self.reports[variant] = (mt, json.load(f))
        return self.reports[variant][1]

    def reports_for_page(self) -> dict:
        """Curves and per-bag numbers of the installed models, for the charts."""
        out = {}
        for v in ("xyz", "full"):
            r = self.report(v)
            if r is None:
                continue
            out[v] = {k: r.get(k) for k in ("bags", "mirror", "cross_validation",
                                            "partly_labelled_scans", "learning_curves")}
            b = self.model(v)
            out[v]["trained"] = b.get("trained") if b else None
            out[v]["threshold"] = b.get("default_threshold", 0.5) if b else None
        return out

    # ── accuracy on one bag ─────────────────────────────────────────────────
    def bag_accuracy(self, name: str) -> dict:
        """
        How the installed model does on one bag:
          cv / partial  from the training report: predictions for this bag's
                        scans made by models that had not seen them
          fresh         scans reviewed after the model was trained, scored now;
                        a real test, the model has never seen them
          azimuth       where the labels are, in 10 degree steps
        """
        c = self.cache(name)
        if c is None:
            raise KeyError(f"{name} is still being cached")
        bundle = self.model_for(c)
        variant = "full" if bundle and bundle["use_intensity"] else "xyz"
        key = (name, bundle["tag"] if bundle else None, self.ledger_version)
        if key in self.stats_cache:
            return self.stats_cache[key]
        rep = self.report(variant) if bundle else None
        out = {"bag": name, "variant": variant, "model": bundle["tag"] if bundle else None,
               "cv": None, "partial": None, "fresh": None}
        if rep:
            strip = lambda d: d and {k: v for k, v in d.items() if k != "per_bag"}
            out["cv"] = strip(rep["cross_validation"].get("per_bag", {}).get(name))
            out["partial"] = strip(rep.get("partly_labelled_scans", {})
                                   .get("per_bag", {}).get(name))
            out["has_per_bag"] = "per_bag" in rep["cross_validation"]

        # 10 degree steps of grid azimuth, whatever the column count.
        hist = lambda keys: np.bincount(
            [min(35, int(col * sg.COL_DEG) // 10) for k in keys for _, col in self.ann[k]],
            minlength=36).tolist()
        mine = [k for k in self.ann if k[0] == name]
        out["azimuth"] = {"bag": hist(mine), "all": hist(list(self.ann))}

        if rep and "trained_scans" in rep:
            seen = set(rep["trained_scans"].get(name, []))
            fresh = [(c["rec2i"][k[1]], k) for k in mine
                     if k[1] not in seen and k[1] in c["rec2i"]
                     and self.complete.get(k) is not None][:600]
            ths = [round(0.05 * k, 2) for k in range(1, 20)]
            rows = {True: np.zeros((len(ths), 3), int), False: np.zeros((len(ths), 3), int)}
            n = {True: 0, False: 0}
            for s in range(0, len(fresh), 64):
                part = fresh[s:s + 64]
                need = [i for i, _ in part if (key[1], name, i) not in self.pred_cache]
                if need:
                    p, _ = self.predict(c, np.array(need))
                    if len(self.pred_cache) > 4000:
                        self.pred_cache.clear()
                    for i, pk in zip(need, p):
                        self.pred_cache[(key[1], name, i)] = pk
                for i, k in part:
                    pk = self.pred_cache[(key[1], name, i)]
                    lab = np.zeros(pk.shape, bool)
                    for ring, col in self.ann[k]:
                        lab[ring, col] = c["range_mm"][i, ring, col] >= af.MIN_RANGE_M * 1000
                    done = self.complete[k]
                    n[done] += 1
                    for t, th in enumerate(ths):
                        flag = pk >= th
                        rows[done][t] += (int((flag & lab).sum()), int(flag.sum()),
                                          int(lab.sum()))
            out["fresh"] = {"thresholds": ths,
                            "complete": {"scans": n[True], "rows": rows[True].tolist()},
                            "partial": {"scans": n[False], "rows": rows[False].tolist()}}
        self.stats_cache[key] = out
        return out

    # ── files: models and exports ───────────────────────────────────────────
    def files(self) -> dict:
        a = self.args
        models, exports = [], []

        def model_entry(rel):
            path = os.path.join(a.model_dir, rel)
            e = {"id": rel, "name": rel or "installed", "bytes": tree_size(path),
                 "mtime": os.path.getmtime(path), "current": rel == ""}
            for v in ("xyz", "full"):
                rp = os.path.join(path, f"artifact_{v}.report.json")
                if os.path.exists(rp):
                    try:
                        with open(rp) as f:
                            e[v] = summarise_report(json.load(f))
                        e["mtime"] = os.path.getmtime(rp)
                    except Exception:
                        pass
            e["usable"] = os.path.exists(os.path.join(path, "artifact_xyz.joblib"))
            return e

        if os.path.isdir(a.model_dir):
            cur = model_entry("")
            cur["bytes"] = sum(os.path.getsize(os.path.join(a.model_dir, f))
                               for f in os.listdir(a.model_dir)
                               if os.path.isfile(os.path.join(a.model_dir, f))
                               and f.startswith("artifact_"))
            models.append(cur)
            for d in sorted(os.listdir(a.model_dir)):
                full = os.path.join(a.model_dir, d)
                if not os.path.isdir(full) or d == "_staging":
                    continue
                if d == "history":
                    models += [model_entry(os.path.join(d, h)) for h in sorted(os.listdir(full))
                               if os.path.isdir(os.path.join(full, h))]
                else:
                    models.append(model_entry(d))
            models[1:] = sorted(models[1:], key=lambda e: -e["mtime"])

        if os.path.isdir(a.out_dir):
            running = self.jobs.running("export") is not None
            for d in os.listdir(a.out_dir):
                full = os.path.join(a.out_dir, d)
                e = {"id": d, "name": d, "bytes": tree_size(full),
                     "mtime": os.path.getmtime(full), "dir": os.path.isdir(full),
                     "runs": [], "busy": running and os.path.isdir(full)
                     and time.time() - os.path.getmtime(full) < 30}
                sums = ([os.path.join(full, f) for f in os.listdir(full)
                         if f.endswith("_summary.json")] if e["dir"]
                        else [full] if d.endswith("_summary.json") else [])
                for sp in sorted(sums):
                    try:
                        with open(sp) as f:
                            sm = json.load(f)
                        bagname = sm.get("bag") or os.path.basename(sp)[:-13] + ".mcap"
                        e["runs"].append({
                            "bag": bagname, "threshold": sm.get("threshold"),
                            "model": sm.get("model"), "scans": sm.get("n_scans"),
                            "flagged_fraction": sm.get("flagged_fraction"),
                            "has_bag": os.path.isdir(os.path.join(
                                os.path.dirname(sp), cache_name(bagname) + "_classified"))})
                        e["mtime"] = max(e["mtime"], os.path.getmtime(sp))
                    except Exception:
                        pass
                exports.append(e)
            exports.sort(key=lambda e: -e["mtime"])
        return {"models": models, "exports": exports,
                "model_dir": os.path.abspath(a.model_dir),
                "out_dir": os.path.abspath(a.out_dir)}

    def _listed(self, kind: str, ident: str) -> str:
        """Full path of an entry files() lists; anything else is refused."""
        listing = self.files()["models" if kind == "model" else "exports"]
        if not ident or not any(e["id"] == ident for e in listing):
            raise ValueError("no such entry")
        return os.path.join(self.args.model_dir if kind == "model" else self.args.out_dir,
                            ident)

    def delete_file(self, kind: str, ident: str) -> dict:
        path = self._listed(kind, ident)
        if self.jobs.running("train") and kind == "model":
            raise ValueError("training is running")
        if self.jobs.running("export") and kind == "export":
            raise ValueError("an export is running")
        freed = tree_size(path)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        return {"freed": freed}

    def use_model(self, ident: str) -> dict:
        """Install an older model again; the current one moves to history."""
        src = self._listed("model", ident)
        a = self.args
        if self.jobs.running("train"):
            raise ValueError("training is running")
        if not os.path.exists(os.path.join(src, "artifact_xyz.joblib")):
            raise ValueError("that folder has no artifact_xyz.joblib")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        keep = os.path.join(a.model_dir, "history", stamp + "_replaced")
        os.makedirs(keep)
        entry = {"time": stamp, "restored_from": ident, "models": {}}
        with self.model_lock:
            for v in ("xyz", "full"):
                for ext in (".joblib", ".report.json"):
                    cur = os.path.join(a.model_dir, f"artifact_{v}{ext}")
                    if os.path.exists(cur):
                        shutil.move(cur, os.path.join(keep, os.path.basename(cur)))
                    new = os.path.join(src, f"artifact_{v}{ext}")
                    if os.path.exists(new):
                        shutil.copyfile(new, cur)
                rp = os.path.join(a.model_dir, f"artifact_{v}.report.json")
                if os.path.exists(rp):
                    with open(rp) as f:
                        entry["models"][v] = summarise_report(json.load(f))
        with open(os.path.join(a.model_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.stats_cache.clear()
        return {"ok": True}

    # ── scan payload ────────────────────────────────────────────────────────
    def scan(self, name: str, i: int, ctx: int) -> dict:
        c = self.cache(name)
        if c is None:
            raise KeyError(f"{name} is still being cached")
        n = len(c["rec_t"])
        i = max(0, min(n - 1, i))
        stamp = int(c["rec_t"][i])
        key = (name, stamp)
        p, bundle = self.predict(c, np.array([i]))
        out = {
            "bag": name, "i": i, "n": n, "stamp_ns": str(stamp),
            "t": (stamp - int(c["rec_t"][0])) / 1e9,
            "range": np.asarray(c["range_mm"][i]).ravel().tolist(),
            "intensity": (np.asarray(c["intensity"][i]).ravel().tolist()
                          if c["intensity"] is not None else None),
            "prob": (np.round(p[0] * 255).astype(int).ravel().tolist()
                     if p is not None else None),
            "model": bundle["tag"] if bundle else None,
            "threshold": bundle.get("default_threshold", 0.5) if bundle else None,
            "selections": sorted(self.ann.get(key, [])),
            "reviewed": key in self.ann,
            "complete": self.complete.get(key),
            "rot": self.rotation(self.bag(name)),
            "ctx": {},
        }
        for d in range(-ctx, ctx + 1):
            j = i + d
            if d and 0 <= j < n and abs(int(c["hdr_t"][j]) - int(c["hdr_t"][i])) \
                    <= abs(d) * af.MAX_DT_NS:
                out["ctx"][str(d)] = np.asarray(c["range_mm"][j]).ravel().tolist()
        return out

    # ── review queue ────────────────────────────────────────────────────────
    def queue(self, name: str) -> dict:
        c = self.cache(name)
        if c is None:
            return {"status": "caching"}
        bundle = self.model_for(c)
        if bundle is None:
            return {"status": "no model"}
        path = os.path.join(self.args.cache_dir, f"{cache_name(name)}_queue.npz")
        tag = bundle["tag"]
        if os.path.exists(path):
            z = np.load(path)
            if str(z["tag"]) == tag:
                return {"status": "ready", "items": self._pick(name, z), "model": tag}
        job = self.jobs.running("queue", name)
        if job is None:
            def work(log, job):
                idx = np.arange(0, len(c["rec_t"]), max(1, int(self.cfg["queue"]["stride"])))
                unc = np.zeros(len(idx), np.int32)
                flg = np.zeros(len(idx), np.int32)
                thr = bundle.get("default_threshold", 0.5)
                for s in range(0, len(idx), 128):
                    p, _ = self.predict(c, idx[s:s + 128])
                    p = p.reshape(len(p), -1)
                    unc[s:s + 128] = ((p >= 0.1) & (p <= 0.9)).sum(1)
                    flg[s:s + 128] = (p >= thr).sum(1)
                    if (s // 128) % 8 == 0:
                        log(f"scored {min(s + 128, len(idx))}/{len(idx)} sampled scans")
                np.savez(path, tag=tag, idx=idx, unc=unc, flg=flg)
                log("done")
            self.jobs.start("queue", name, work)
        return {"status": "computing"}

    def _pick(self, name: str, z) -> list:
        """Most uncertain scans first, spread out, away from reviewed ones."""
        taken = sorted(self.reviewed_indices(name))
        idx, unc, flg = z["idx"], z["unc"], z["flg"]
        out = []
        gap = float(self.cfg["queue"]["min_gap_s"]) * sg.SCAN_RATE_HZ
        for k in np.argsort(-(unc * 1000 + flg), kind="stable"):
            i = int(idx[k])
            if unc[k] == 0 and flg[k] == 0:
                break
            if any(abs(i - t) < gap for t in taken):
                continue
            taken.append(i)
            out.append({"i": i, "uncertain": int(unc[k]), "flagged": int(flg[k])})
            if len(out) >= int(self.cfg["queue"]["length"]):
                break
        return out

    # ── training and export ─────────────────────────────────────────────────
    def train(self, threshold: float) -> int:
        if self.jobs.running("train"):
            raise ValueError("training is already running")
        a = self.args
        labelled = {k[0] for k in self.ann}
        bags = [b["path"] for b in self.reg["bags"]
                if b["name"] in labelled and self.cache(b["name"]) is not None]
        raws = [b["path"] for b in self.reg["bags"]
                if self.cache(b["name"]) is not None
                and self.cache(b["name"])["intensity"] is not None]
        if not bags:
            raise ValueError("no registered bag has annotations")
        staging = os.path.join(a.model_dir, "_staging")
        cmd = [sys.executable, "-u", os.path.join(HERE, "train_artifact_gbdt.py"),
               *config.passthrough(self.cfg), "--bags", *bags,
               "--model-dir", staging, "--progress",
               "--default-threshold", str(threshold)]
        if raws:
            cmd += ["--intensity-bags", *raws]
        if self.reg.get("exclude_sessions"):
            cmd += ["--exclude-sessions", *self.reg["exclude_sessions"]]
        if self.reg.get("exclude_scans"):
            cmd += ["--exclude-scans", *self.reg["exclude_scans"]]

        def work(log, job):
            shutil.rmtree(staging, ignore_errors=True)
            job["progress"] = prog = {"stage": "starting", "frac": 0.0, "device": None,
                                      "variants": ["xyz", "full"], "curves": {},
                                      "results": {},
                                      "rounds": int(self.cfg["training"]["rounds"])}
            folds = {}

            def on_event(ev):
                kind = ev.get("ev")
                if kind == "start":
                    prog["device"], prog["rounds"] = ev["device"], ev["rounds"]
                elif kind == "dataset":
                    prog["stage"] = f"reading labelled scans, bag {ev['k']} of {ev['of']}"
                    prog["frac"] = 0.1 * (ev["k"] - 1) / ev["of"]
                elif kind == "variants":
                    prog["variants"] = ev["names"]
                elif kind == "fit":
                    v, k = ev["variant"], ev["fold"]
                    if k:
                        folds[v] = ev["folds"]
                        if ev["value"] is not None:
                            prog["curves"].setdefault(v, {}).setdefault(str(k), []).append(
                                [ev["round"], round(ev["value"], 4)])
                    nf = folds.get(v, 5)
                    who = "range-only model" if v == "xyz" else "model with intensity"
                    prog["stage"] = (f"{who}: held-out fold {k} of {nf}" if k
                                     else f"{who}: final fit on every complete scan")
                    vs = prog["variants"]
                    unit = (vs.index(v) if v in vs else 0) * (nf + 1) + (k - 1 if k else nf)
                    prog["frac"] = 0.1 + 0.9 * (unit + ev["round"] / ev["rounds"]) \
                        / (len(vs) * (nf + 1))
                elif kind == "cv":
                    prog["results"][ev["variant"]] = ev

            run_process(cmd, log, on_event)
            prog["stage"], prog["frac"] = "installing the new models", 1.0
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            keep = os.path.join(a.model_dir, "history", stamp + "_replaced")
            entry = {"time": stamp, "threshold": threshold, "models": {}}
            for v in ("xyz", "full"):
                for ext in (".joblib", ".report.json"):
                    cur = os.path.join(a.model_dir, f"artifact_{v}{ext}")
                    new = os.path.join(staging, f"artifact_{v}{ext}")
                    if os.path.exists(new):
                        if os.path.exists(cur):
                            os.makedirs(keep, exist_ok=True)
                            shutil.move(cur, os.path.join(keep, os.path.basename(cur)))
                        shutil.move(new, cur)
                rp = os.path.join(a.model_dir, f"artifact_{v}.report.json")
                if os.path.exists(rp):
                    with open(rp) as f:
                        entry["models"][v] = summarise_report(json.load(f))
            shutil.rmtree(staging, ignore_errors=True)
            with open(os.path.join(a.model_dir, "history.jsonl"), "a") as f:
                f.write(json.dumps(entry) + "\n")
            prog["stage"] = "done"
            self.stats_cache.clear()
            log(f"models installed; the replaced ones are in {keep}")
        return self.jobs.start("train", "train", work)

    def history(self) -> list:
        path = os.path.join(self.args.model_dir, "history.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()][-60:]

    def export(self, name: str, threshold: float | None) -> int:
        a = self.args
        b = self.bag(name)
        out_dir = os.path.join(a.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        cmd = [sys.executable, "-u", os.path.join(HERE, "classify_bag.py"), b["path"],
               *config.passthrough(self.cfg), "--model-dir", a.model_dir,
               "--out-dir", out_dir, "--topic", b["topic"], "--write-bag"]
        if threshold is not None:
            cmd += ["--threshold", str(threshold)]

        def work(log, job):
            run_process(cmd, log)
            log(f"classified bag written under {os.path.abspath(out_dir)}")
        return self.jobs.start("export", name, work)

    # ── overall state ───────────────────────────────────────────────────────
    def state(self) -> dict:
        bags = []
        for b in self.reg["bags"]:
            c = self.cache(b["name"])
            try:
                mb = round(os.path.getsize(b["path"]) / 1e6)
            except OSError:
                mb = None                   # drive not mounted
            bags.append({"name": b["name"], "path": b["path"],
                         "label": b.get("label", ""), "mb": mb, "topic": b["topic"],
                         "folder": folder_of(b["path"], self.args.bags_root),
                         "start": float(c["rec_t"][0]) / 1e9 if c else None,
                         "name_date": name_date(b["path"]),
                         "cached": c is not None,
                         "n": len(c["rec_t"]) if c else 0,
                         "intensity": bool(c and c["intensity"] is not None),
                         "seconds": (float(c["rec_t"][-1] - c["rec_t"][0]) / 1e9) if c else 0,
                         **self.bag_stats(b["name"])})
        return {"bags": bags, "models": self.model_info(), "jobs": self.jobs.listing(),
                "history": self.history(), "session": SESSION_ID,
                "config": {"project": self.cfg["project"], "bags_root": self.args.bags_root,
                           "default_topic": self.args.topic,
                           "bag_extensions": list(self.exts),
                           "sensor": sg.PROFILE.get("name", ""),
                           "scan_rate_hz": sg.SCAN_RATE_HZ},
                "unregistered": sorted({k[0] for k in self.ann}
                                       - {b["name"] for b in self.reg["bags"]})}


def folder_of(path: str, root: str) -> str:
    """Folder of a bag, relative to the bags root when it is inside it."""
    d = os.path.dirname(path)
    rel = os.path.relpath(d, root)
    return d if rel.startswith("..") else rel


def name_date(path: str) -> str | None:
    """'2026-05-17 13:14:14' from a ..._2026_05_17-13_14_14_... file name."""
    m = re.search(r"(\d{4})_(\d\d)_(\d\d)-(\d\d)_(\d\d)_(\d\d)", os.path.basename(path)) \
        or re.search(r"(\d{4})_(\d\d)_(\d\d)-(\d\d)_(\d\d)_(\d\d)", path)
    return "{}-{}-{} {}:{}:{}".format(*m.groups()) if m else None


def tree_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def summarise_report(rep: dict) -> dict:
    def at(block, th):
        for t in block.get("thresholds", []):
            if abs(t["threshold"] - th) < 1e-6:
                return t
        return {}
    cv = rep.get("cross_validation", {})
    pl = rep.get("partly_labelled_scans", {})
    return {
        "train_scans": cv.get("n_scans"), "train_labels": cv.get("n_artifact"),
        "cv_ap": cv.get("average_precision"),
        "cv_precision": at(cv, 0.3).get("precision_vs_labels"),
        "cv_recall": at(cv, 0.3).get("recall"),
        "partial_scans": pl.get("n_scans"), "partial_labels": pl.get("n_artifact"),
        "partial_recall": at(pl, 0.3).get("recall"),
        "partial_extra": at(pl, 0.3).get("flagged_unlabelled"),
    }


def read_lidar_rotation(bag_path: str, topic: str, child: str | None = None):
    """
    Row-major rotation taking vectors of the lidar frame into its parent
    frame, from /tf_static. The lidar frame is `child`, or the frame_id of the
    first message on `topic`. None when the bag has no such transform.
    """
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage

    if not child:
        for _, _, msg, _ in sg.iter_scans(bag_path, topic):
            child = msg.header.frame_id
            break
    reader = sg.open_reader(bag_path, ["/tf_static"])
    while reader.has_next():
        _, data, _ = reader.read_next()
        for tr in deserialize_message(data, TFMessage).transforms:
            if tr.child_frame_id.lstrip("/") == (child or "").lstrip("/"):
                q = tr.transform.rotation
                x, y, z, w = q.x, q.y, q.z, q.w
                return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    return None


def geometry() -> dict:
    """Beam direction of every grid cell, so the browser can rebuild xyz."""
    elev = sg.LAYER_ELEV_DEG[::-1]          # ring 0 = highest layer
    return {"rings": sg.N_RINGS, "cols": sg.N_COLS, "col_deg": sg.COL_DEG,
            "elev_deg": elev.tolist(),
            "az_deg": np.round(sg.cell_azimuth_deg(), 3).ravel().tolist(),
            "min_range_m": af.MIN_RANGE_M}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _run(self, fn):
            try:
                self._send(200, fn())
            except (KeyError, ValueError) as e:
                self._send(400, {"error": str(e).strip("'\"")})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "annotator_app.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            routes = {
                "/api/state": lambda: app.state(),
                "/api/geometry": geometry,
                "/api/browse": lambda: app.browse(),
                "/api/scan": lambda: app.scan(q["bag"], int(q.get("i", 0)),
                                              min(int(q.get("ctx", 0)), 3)),
                "/api/reviewed": lambda: app.reviewed_indices(q["bag"]),
                "/api/queue": lambda: app.queue(q["bag"]),
                "/api/reports": lambda: app.reports_for_page(),
                "/api/accuracy": lambda: app.bag_accuracy(q["bag"]),
                "/api/files": lambda: app.files(),
                "/api/topics": lambda: app.topics(q["path"]),
            }
            if u.path in routes:
                return self._run(routes[u.path])
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            routes = {
                "/api/save": lambda: app.save_scan(body["bag"], int(body["i"]),
                                                   body["selections"], body["complete"]),
                "/api/bags/add": lambda: {"job": app.add_bag(body["path"], body.get("topic"))},
                "/api/bags/remove": lambda: app.remove_bag(body["bag"]) or {"ok": True},
                "/api/bags/label": lambda: app.set_label(body["bag"], body["label"]) or {"ok": True},
                "/api/bags/topic": lambda: app.set_topic(body["bag"], body["topic"]),
                "/api/files/delete": lambda: app.delete_file(body["kind"], body["id"]),
                "/api/models/use": lambda: app.use_model(body["id"]),
                "/api/train": lambda: {"job": app.train(float(body.get("threshold", 0.3)))},
                "/api/export": lambda: {"job": app.export(body["bag"], body.get("threshold"))},
            }
            if u.path in routes:
                return self._run(routes[u.path])
            self._send(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_arguments(ap)
    ap.add_argument("--registry", default=None, help="bag registry file")
    ap.add_argument("--bags-root", default=None,
                    help="folder the Add dialog lists bags from "
                         "(default: the project directory)")
    ap.add_argument("--topic", default=None,
                    help="lidar topic preferred when a bag has several point clouds")
    ap.add_argument("--sensor-from-bag", default=None, metavar="BAG",
                    help="derive the sensor profile from this recording, save it "
                         "as <project>/sensor.json, and start")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    over = {k: v for k, v in (("host", args.host), ("port", args.port),
                              ("default_topic", args.topic)) if v is not None}
    if args.bags_root:
        over["bags_root"] = os.path.abspath(os.path.expanduser(args.bags_root))
    if args.registry:
        over["paths"] = {"registry": os.path.abspath(args.registry)}
    if args.sensor_from_bag:
        import sensor_profile
        pre = config.from_args(args, over, need_sensor=False)
        out = os.path.join(pre["project"], "sensor.json")
        if os.path.exists(out):
            sys.exit(f"{out} already exists; start without --sensor-from-bag to use it.")
        bag = os.path.abspath(os.path.expanduser(args.sensor_from_bag))
        names = [t["name"] for t in sg.pointcloud_topics(bag)]     # busiest first
        if not names:
            sys.exit(f"{bag} has no PointCloud2 topic")
        topic = pre["default_topic"] if pre["default_topic"] in names else names[0]
        try:
            profile = sensor_profile.derive(bag, topic)
        except ValueError as e:
            sys.exit(f"Could not derive a sensor profile: {e}")
        if profile["_derived"]["problems"]:
            sys.exit("The recording does not fit a fixed scan grid (see above); "
                     "no profile written. With several point cloud topics, name "
                     "the lidar's with --topic.")
        os.makedirs(pre["project"], exist_ok=True)
        with open(out, "w") as f:
            json.dump(profile, f, indent=1)
        print(f"Sensor profile written to {out}", flush=True)
    cfg = config.from_args(args, over)
    os.makedirs(cfg["project"], exist_ok=True)
    try:
        sg.check_cache_dir(cfg["paths"]["cache"])
    except RuntimeError as e:
        sys.exit(str(e))

    app = App(cfg)
    server = ThreadingHTTPServer((cfg["host"], cfg["port"]), make_handler(app))
    server.daemon_threads = True
    print(f"Annotator: http://{cfg['host']}:{cfg['port']}   project {cfg['project']}\n"
          f"  sensor: {sg.PROFILE.get('name', '?')} ({sg.N_RINGS} x {sg.N_COLS})   "
          f"{len(app.reg['bags'])} bags, {len(app.ann)} reviewed scans, "
          f"session {SESSION_ID}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
