#!/usr/bin/env python3
"""
publish_model.py

Makes the project's trained models the multiScan driver's defaults: copies
models/artifact_<variant>.model.txt and its .report.json into the driver's
models/ folder, which it installs and loads by name
(ghost_filter.model_file: artifact_full, fallback_model_file: artifact_xyz).

    python3 publish_model.py [--driver DIR] [--variant full xyz] [--project DIR]

--driver is the multiscan-driver repository; by default the submodule next
to this one (<workspace>/src/multiscan_driver). Only realtime models with
early exit checkpoints are published: the driver filters live scans on one
core.

Afterwards, in the driver: rebuild, run its tests (test_ghost_filter checks
that the default models load and fit the sensor), and commit the models
there.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from export_model import FORMAT_VERSION

MODEL_HEADER = f"lidar_artifact_model {FORMAT_VERSION}"


def read_settings(path: str) -> dict:
    """The settings lines of a model file, up to its trees."""
    settings = {}
    with open(path) as f:
        if f.readline().strip() != MODEL_HEADER:
            raise ValueError(f"{path} is not a '{MODEL_HEADER}' file")
        for line in f:
            key, _, values = line.strip().partition(" ")
            if key == "trees":
                break
            settings[key] = values
    return settings


def check_publishable(path: str, variant: str) -> dict:
    settings = read_settings(path)
    if settings.get("variant") != variant:
        raise ValueError(f"{path} is the '{settings.get('variant')}' model, not '{variant}'")
    offsets = [int(v) for v in settings.get("time_offsets", "0").split()[1:]]
    if any(d >= 0 for d in offsets):
        raise ValueError(f"{path} looks at later scans: train a realtime model "
                         "(\"before only (realtime)\" in the app)")
    if "early_exit" not in settings:
        raise ValueError(f"{path} has no early exit checkpoints: retrain it "
                         "(the driver needs them to keep up on one core)")
    return settings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", default=None,
                    help="the multiscan-driver repository (default: <workspace>/src/multiscan_driver)")
    ap.add_argument("--variant", nargs="+", choices=("full", "xyz"), default=["full", "xyz"])
    config.add_arguments(ap)
    args = ap.parse_args()
    cfg = config.from_args(args, need_sensor=False)

    here = os.path.dirname(os.path.abspath(__file__))
    driver = args.driver or os.path.join(os.path.dirname(here), "multiscan_driver")
    target = os.path.join(driver, "models")
    if not os.path.isfile(os.path.join(driver, "package.xml")) or not os.path.isdir(target):
        sys.exit(f"{driver} is not the multiscan-driver repository (no models/ folder); "
                 "pass --driver")

    for variant in args.variant:
        name = f"artifact_{variant}"
        source = os.path.join(cfg["paths"]["models"], name + ".model.txt")
        if not os.path.exists(source):
            sys.exit(f"{source} not found: train the {variant} model first")
        try:
            new = check_publishable(source, variant)
        except ValueError as e:
            sys.exit(str(e))
        destination = os.path.join(target, name + ".model.txt")
        old = read_settings(destination) if os.path.exists(destination) else {}
        shutil.copyfile(source, destination)
        report = os.path.join(cfg["paths"]["models"], name + ".report.json")
        if os.path.exists(report):
            shutil.copyfile(report, os.path.join(target, name + ".report.json"))
        print(f"{name}: trained {old.get('trained', '(none)')} -> {new.get('trained')}")

    print(f"\nPublished to {target}. Next, in the driver:\n"
          "  colcon build --packages-select multiscan_driver\n"
          "  colcon test --packages-select multiscan_driver\n"
          "and commit the models there (and the submodule pointer here).")


if __name__ == "__main__":
    main()
