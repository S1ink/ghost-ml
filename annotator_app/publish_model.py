#!/usr/bin/env python3
"""
publish_model.py

Makes the project's trained models the multiScan driver's defaults: copies
models/artifact_<variant>.model.txt and its .report.json into the driver's
models/ folder, which it installs and loads by name
(ghost_filter.model_file: artifact_full, fallback_model_file: artifact_xyz).
The app does the same from Files -> "Publish to driver".

    python3 publish_model.py [--driver DIR] [--variant full xyz] [--project DIR]

--driver is the multiscan-driver repository. By default it is the project's
"driver" setting, which defaults to the submodule (src/multiscan_driver).
Only realtime models with early exit checkpoints are published: the driver
filters live scans on one core.

Afterwards, in the driver: rebuild, run its tests (test_ghost_filter checks
that the default models load and fit the sensor), and commit the models
there, then the submodule pointer here.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from export_model import FORMAT_VERSION

MODEL_HEADER = f"lidar_artifact_model {FORMAT_VERSION}"
VARIANTS = ("full", "xyz")

NEXT_STEPS = ("Next, in the driver:\n"
              "  colcon build --packages-select multiscan_driver\n"
              "  colcon test --packages-select multiscan_driver\n"
              "and commit the models there (and the submodule pointer here).")


# ── model files ──────────────────────────────────────────────────────────────
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
    """The model's settings; ValueError when the driver can't use it."""
    settings = read_settings(path)
    if settings.get("variant") != variant:
        raise ValueError(f"{path} is the '{settings.get('variant')}' model, not '{variant}'")
    offsets = [int(offset) for offset in settings.get("time_offsets", "0").split()[1:]]
    if any(offset >= 0 for offset in offsets):
        raise ValueError(f"{path} looks at later scans: train a realtime model "
                         "(\"before only (realtime)\" in the app)")
    if "early_exit" not in settings:
        raise ValueError(f"{path} has no early exit checkpoints: retrain it "
                         "(the driver needs them to keep up on one core)")
    return settings


def model_paths(folder: str, variant: str) -> tuple[str, str]:
    name = f"artifact_{variant}"
    return (os.path.join(folder, name + ".model.txt"),
            os.path.join(folder, name + ".report.json"))


# ── the driver ───────────────────────────────────────────────────────────────
def driver_models_dir(driver: str) -> str:
    """The driver's models/ folder; ValueError when driver isn't the driver."""
    target = os.path.join(driver, "models")
    if not os.path.isfile(os.path.join(driver, "package.xml")) or not os.path.isdir(target):
        raise ValueError(f"{driver} is not the multiscan-driver repository (no models/ "
                         "folder): set \"driver\" in annotator.json or pass --driver")
    return target


def uncommitted_models(driver: str) -> list[str]:
    """Files under the driver's models/ that git reports as changed."""
    try:
        result = subprocess.run(["git", "-C", driver, "status", "--porcelain", "--", "models"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def status(driver: str, model_dir: str) -> dict:
    """
    What publishing would do: per variant, the trained date of the project's
    model and of the driver's default, whether they are the same file, and
    why the project's model can't be published (if it can't).
    """
    out = {"driver": os.path.abspath(driver), "found": False, "problem": None,
           "variants": {}, "uncommitted": []}
    try:
        target = driver_models_dir(driver)
    except ValueError as e:
        out["problem"] = str(e)
        return out
    out["found"] = True
    out["uncommitted"] = uncommitted_models(driver)
    for variant in VARIANTS:
        source, _ = model_paths(model_dir, variant)
        destination, _ = model_paths(target, variant)
        entry = {"project_trained": None, "driver_trained": None,
                 "same": False, "problem": None}
        if os.path.exists(destination):
            try:
                entry["driver_trained"] = read_settings(destination).get("trained")
            except ValueError:
                entry["driver_trained"] = "(unreadable)"
        if not os.path.exists(source):
            entry["problem"] = f"no {variant} model trained yet"
        else:
            try:
                entry["project_trained"] = check_publishable(source, variant).get("trained")
            except ValueError as e:
                entry["problem"] = str(e)
            entry["same"] = (os.path.exists(destination)
                             and filecmp.cmp(source, destination, shallow=False))
        out["variants"][variant] = entry
    return out


def publish(driver: str, model_dir: str, variants=VARIANTS) -> list[dict]:
    """
    Copies the project's models of the given variants into the driver. All
    are checked before any is copied, so a problem leaves the driver as it
    was. Returns, per variant, the trained date before and after.
    """
    target = driver_models_dir(driver)
    checked = []
    for variant in variants:
        source, report = model_paths(model_dir, variant)
        if not os.path.exists(source):
            raise ValueError(f"{source} not found: train the {variant} model first")
        checked.append((variant, source, report, check_publishable(source, variant)))

    results = []
    for variant, source, report, settings in checked:
        destination, destination_report = model_paths(target, variant)
        old = read_settings(destination) if os.path.exists(destination) else {}
        shutil.copyfile(source, destination)
        if os.path.exists(report):
            shutil.copyfile(report, destination_report)
        results.append({"variant": variant, "old_trained": old.get("trained"),
                        "new_trained": settings.get("trained")})
    return results


# ── command line ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--driver", default=None,
                        help="the multiscan-driver repository (default: the project's "
                             "\"driver\" setting, else <workspace>/src/multiscan_driver)")
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    config.add_arguments(parser)
    args = parser.parse_args()
    settings = config.from_args(args, need_sensor=False)
    driver = os.path.abspath(args.driver) if args.driver else settings["driver"]

    try:
        results = publish(driver, settings["paths"]["models"], args.variant)
    except ValueError as e:
        sys.exit(str(e))
    for result in results:
        print(f"artifact_{result['variant']}: trained {result['old_trained'] or '(none)'}"
              f" -> {result['new_trained']}")
    print(f"\nPublished to {driver_models_dir(driver)}. {NEXT_STEPS}")


if __name__ == "__main__":
    main()
