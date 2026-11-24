#!/usr/bin/env python3
"""
config.py

Settings shared by the annotator app, the trainer and the classifier.

A *project* is a directory holding the annotation ledger, the bag registry,
the scan cache, the models and the exports. Its settings come from, in order
of precedence:

    1. command line flags
    2. <project>/annotator.json (or the file given with --config)
    3. the defaults below

annotator.json holds only what differs from the defaults; see
annotator.example.json. Unknown keys are refused, so a typo is not silently
ignored. Relative paths are relative to the project directory.

The sensor profile (grid geometry, see scan_grid.py) is found as:

    --sensor / "sensor"   a profile shipped in sensors/ by name ("multiscan"),
                          a path to a profile file, or (in the file) the
                          profile itself
    otherwise             <project>/sensor.json, which
                          `sensor_profile.py <bag> -o <project>/sensor.json`
                          or `annotator_app.py --sensor-from-bag <bag>` writes
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import artifact_features as af
import scan_grid as sg

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    # Sensor profile: name of a file in sensors/, a path, or the profile itself.
    # None = <project>/sensor.json.
    "sensor": None,

    # Folder the "Add..." dialog lists bags from (searched recursively).
    # None = the project directory.
    "bags_root": None,
    "bag_extensions": [".mcap", ".db3"],
    # Lidar topic used when a bag has several PointCloud2 topics. None = a bag
    # with one PointCloud2 topic just works, others ask.
    "default_topic": None,
    # Frame whose /tf_static rotation levels the 3D view ("robot frame").
    # None = the frame_id of the lidar messages.
    "lidar_frame": None,

    "host": "127.0.0.1",
    "port": 8765,

    # The multiscan-driver repository trained models are published to (its
    # models/ folder holds the driver's default models). None = the
    # submodule next to this folder (src/multiscan_driver).
    "driver": None,

    "paths": {
        "ledger": "annotations.jsonl",
        "registry": "datasets.json",
        "cache": "cache",
        "models": "models",
        "out": "out",
    },

    # Review queue: which scans the app suggests looking at.
    "queue": {
        "stride": 5,            # every n-th scan is scored
        "min_gap_s": 2.0,       # suggestions are at least this far apart
        "length": 40,
    },

    # What the classifier sees. A model is tied to these values; changing
    # them means retraining. Offsets are in grid cells.
    "features": {
        "min_range_m": 0.2,     # nearer returns are ignored everywhere
        "ring_offsets": [-2, -1, 0, 1, 2],
        "col_offsets": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        "close_log": 0.05,      # |log range ratio| under this = same surface
        "far_log": 0.4,         # over this = a different surface
        "clip": 4.0,
        "time_offsets": [-2, -1, 1, 2],   # neighbouring scans used as context
        "max_scan_gap": 1.6,    # in scan periods, per step, before a
                                # neighbouring scan counts as missing
    },

    "training": {
        "rounds": 600,          # boosting rounds
        "cv_group_s": 10.0,     # scans this close in time share a CV fold
        "min_labels": 6,        # unmarked scans with fewer labels are not trained on
        "mirror": True,         # also train on every scan flipped left-right
        "device": "auto",       # auto | cuda | cpu
    },
}


def _merge(base: dict, over: dict, where: str) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):               # "_comment" and the like
            continue
        if k not in base:
            raise ValueError(f"unknown setting '{where}{k}' "
                             f"(known: {', '.join(sorted(base))})")
        if isinstance(base[k], dict) and k != "sensor":
            if not isinstance(v, dict):
                raise ValueError(f"'{where}{k}' must be an object")
            out[k] = _merge(base[k], v, f"{where}{k}.")
        else:
            out[k] = v
    return out


def load_sensor(spec, project: str) -> dict:
    """Resolves the "sensor" setting to a profile dict."""
    if isinstance(spec, dict):
        return spec
    if spec is None:
        path = os.path.join(project, "sensor.json")
        if not os.path.exists(path):
            raise SystemExit(
                "No sensor profile. Either name one (--sensor multiscan, or "
                "\"sensor\" in annotator.json; shipped: "
                f"{', '.join(builtin_sensors()) or 'none'}), or derive one from a "
                "recording of your lidar:\n"
                f"    python3 {os.path.join(HERE, 'sensor_profile.py')} <bag> "
                f"-o {path}")
    else:
        cands = [os.path.join(project, os.path.expanduser(spec)),
                 os.path.join(HERE, "sensors", spec + ".json"),
                 os.path.join(HERE, "sensors", spec)]
        path = next((c for c in cands if os.path.isfile(c)), None)
        if path is None:
            raise SystemExit(f"Sensor profile '{spec}' not found (shipped: "
                             f"{', '.join(builtin_sensors())})")
    with open(path) as f:
        return json.load(f)


def builtin_sensors() -> list:
    d = os.path.join(HERE, "sensors")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json")) \
        if os.path.isdir(d) else []


def load(project: str = ".", config: str | None = None, sensor: str | None = None,
         overrides: dict | None = None, need_sensor: bool = True) -> dict:
    """
    Reads the settings of a project and applies the sensor profile and the
    feature settings to scan_grid and artifact_features. Returns the settings
    with "project" and absolute "paths" / "bags_root".
    """
    project = os.path.abspath(os.path.expanduser(project))
    path = config or os.path.join(project, "annotator.json")
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            try:
                cfg = _merge(cfg, json.load(f), "")
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}: {e}") from None
            except ValueError as e:
                raise SystemExit(f"{path}: {e}") from None
    elif config:
        raise SystemExit(f"{config} does not exist")
    if overrides:
        try:
            cfg = _merge(cfg, overrides, "")
        except ValueError as e:
            raise SystemExit(str(e)) from None
    if sensor:
        cfg["sensor"] = sensor

    absolute = lambda p: os.path.normpath(os.path.join(project, os.path.expanduser(p)))
    cfg["project"] = project
    cfg["config_file"] = path if os.path.exists(path) else None
    cfg["paths"] = {k: absolute(v) for k, v in cfg["paths"].items()}
    cfg["bags_root"] = absolute(cfg["bags_root"]) if cfg["bags_root"] else project
    cfg["driver"] = (absolute(cfg["driver"]) if cfg["driver"]
                     else os.path.join(os.path.dirname(HERE), "multiscan_driver"))
    cfg["bag_extensions"] = [e.lower() for e in cfg["bag_extensions"]]

    if need_sensor:
        try:
            sg.configure(load_sensor(cfg["sensor"], project))
            af.configure(cfg["features"], sg.SCAN_RATE_HZ)
        except ValueError as e:
            raise SystemExit(f"configuration: {e}") from None
    return cfg


def add_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("project")
    g.add_argument("--project", default=".",
                   help="project directory (default: the current directory)")
    g.add_argument("--config", default=None,
                   help="settings file (default: <project>/annotator.json)")
    g.add_argument("--sensor", default=None,
                   help="sensor profile: a shipped name or a file "
                        "(default: the settings file, then <project>/sensor.json)")
    g.add_argument("--ledger", default=None, help="annotation ledger")
    g.add_argument("--cache-dir", default=None)
    g.add_argument("--model-dir", default=None)
    g.add_argument("--out-dir", default=None)


def from_args(args, overrides: dict | None = None, need_sensor: bool = True) -> dict:
    over = copy.deepcopy(overrides or {})
    paths = {k: os.path.abspath(v) for k, v in (
        ("ledger", args.ledger), ("cache", args.cache_dir),
        ("models", args.model_dir), ("out", args.out_dir)) if v}
    if paths:
        over.setdefault("paths", {}).update(paths)
    return load(args.project, args.config, args.sensor, over, need_sensor)


def passthrough(cfg: dict) -> list:
    """Flags that make a child process (trainer, classifier) load the same
    settings as this one."""
    out = ["--project", cfg["project"], "--ledger", cfg["paths"]["ledger"],
           "--cache-dir", cfg["paths"]["cache"]]
    if cfg["config_file"]:
        out += ["--config", cfg["config_file"]]
    if isinstance(cfg["sensor"], str):
        out += ["--sensor", cfg["sensor"]]
    return out


def model_problem(bundle: dict) -> str | None:
    """Why a saved model cannot be used under the current settings, or None."""
    want = bundle.get("feature_spec", af.LEGACY_SPEC)
    have = af.spec()
    diff = [k for k in have if want.get(k) != have[k]]
    if diff:
        return ("was trained with other feature settings ("
                + ", ".join(f"{k}: {want.get(k)} then, {have[k]} now" for k in diff) + ")")
    fp = bundle.get("sensor_fingerprint")
    if fp and fp != sg.fingerprint():
        return f"was trained for another sensor profile ({bundle.get('sensor_name', '?')})"
    names = af.feature_names(tuple(bundle["time_offsets"]), bundle["use_intensity"])
    if names != bundle["feature_names"]:
        return "was trained with a different feature set than artifact_features.py produces"
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prints the settings in effect.")
    add_arguments(ap)
    c = from_args(ap.parse_args(), need_sensor=False)
    print(json.dumps(c, indent=1))
