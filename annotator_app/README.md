# Lidar artifact annotator

A local web app for finding erroneous lidar returns in recorded ROS 2 bags:
register bags, label the bad points scan by scan (range image and 3D view),
train a classifier on the labels, review its proposals, and export bags with
every point scored.

It is a plain directory of Python scripts, not a ROS package. ROS is needed
only as a library for reading and writing bags.

## Requirements

- A ROS 2 installation with `rosbag2_py` (developed on Jazzy).
- Python packages in `requirements.txt`: numpy, scikit-learn, joblib, and
  optionally xgboost (trains on an NVIDIA GPU when one is present).
- A browser on the same machine.

```bash
python3 -m venv --system-site-packages .venv      # still sees the ROS packages
.venv/bin/pip install -r requirements.txt
```

`run.sh` sources ROS from `/opt/ros` when no ROS environment is active and
uses `.venv` next to it (or the one named by `ANNOTATOR_VENV`).

Inside the ghost workspace none of this is needed: `../setup_venv.sh`
creates the environment and `../run_annotator.sh` starts the app with the
workspace as the project.

## Starting

```bash
./run.sh [--port 9000]
```

Then open http://localhost:8765. The Help button in the app explains the
controls.

A **project** is a folder that holds everything belonging to one lidar
setup: labels, the list of bags, the scan cache, models and exports. Any
folder will do. On the first start the page shows a project screen: pick
the folder, say which lidar it is (a shipped profile, or "measure it from a
recording" for any other), and optionally the folder your bags are in. The
answers are saved in the project (`annotator.json`, `sensor.json`), so the
next start opens it directly. The Project button brings the screen back to
switch projects or change the bags folder.

Which project opens at start: `--project <dir>` when given, else the
current folder when it is a project, else the one used last (remembered in
`~/.config/annotator_app/recent.json`).

The same can be done without the page: `--sensor multiscan` or
`--sensor-from-bag <bag>`, `--bags-root <folder>`, or by writing
`annotator.json` by hand.

A Python environment only works with the Python that created it. If the
environment was made inside a container, start the app inside that
container too; `run.sh` says so when the versions differ.

## Realtime models

By default the classifier judges a return with the two scans before and the
two scans after it, which needs a recording. For a model that runs on a
live sensor, choose "before only (realtime)" beside Train (or
`train_artifact_gbdt.py --time-offsets -2 -1`): it uses earlier scans only
and is slightly less accurate. To make that the default, set
`"features": {"time_offsets": [-2, -1]}`. The offsets are stored in the
model, so exports and proposals always match the model in use.

The trained models are `models/artifact_xyz.joblib` (range only) and
`models/artifact_full.joblib` (with intensity). Training also writes each
as `artifact_<variant>.model.txt`: the trees and every setting needed to
build their inputs, as plain text, for use without Python. The C++ filter in
`../ghost_tester` reads that file (see the README one level up).
`export_model.py` writes the same file for an older model, and makes check
files that compare the C++ result with Python's.

## Sensor profile

Scans are stored on a fixed grid of rings (layers) by columns (azimuth
samples). Labels and models are grid cells, so the grid must be the same for
every bag in a project. It is described by a sensor profile:

| key | meaning |
| --- | --- |
| `cols` | azimuth samples per revolution in one layer |
| `layer_elevations_deg` | elevation of every layer, ascending |
| `layer_azimuth0_deg` | azimuth of one sample of every layer; layers may be offset from each other |
| `scan_rate_hz` | scans per second |
| `intensity_field` | point field holding intensity, or null |
| `intensity_max` | raw intensity value that counts as full scale |

`sensors/multiscan.json` is the SICK multiScan profile. For another lidar,
`sensor_profile.py <bag> -o <project>/sensor.json` (or `--sensor-from-bag`)
measures these from a recording and reports how well the recording fits. It
needs a lidar that spins about its z axis with fixed layers and a fixed
azimuth step, recorded in the sensor's own frame. Non-repeating scan
patterns and layers with different azimuth steps are not supported.

The profile is looked up as: `--sensor`, then `"sensor"` in
`annotator.json` (a shipped name, a file, or the profile itself), then
`<project>/sensor.json`. The cache directory remembers its profile and the
app refuses to start on a different one.

## Settings

`<project>/annotator.json` holds only what differs from the defaults.
`annotator.example.json` is a starting point; `config.py` lists every key
with its default, and `python3 config.py --project <dir>` prints the values
in effect. Unknown keys are an error. Relative paths are relative to the
project.

| key | default | meaning |
| --- | --- | --- |
| `sensor` | `<project>/sensor.json` | sensor profile |
| `bags_root` | the project directory | folder the Add dialog lists bags from; a bag elsewhere can be added by path |
| `bag_extensions` | `.mcap`, `.db3` | files that count as bags |
| `default_topic` | none | lidar topic preferred when a bag has several point clouds; without it the app offers a choice and the command line tools ask for `--topic` |
| `lidar_frame` | frame of the lidar messages | frame whose `/tf_static` rotation levels the 3D view |
| `host`, `port` | `127.0.0.1`, `8765` | where the app listens |
| `paths.*` | `annotations.jsonl`, `datasets.json`, `cache`, `models`, `out` | ledger, bag registry, scan cache, models, exports |
| `queue.*` | every 5th scan, 2 s apart, 40 long | which scans the review queue suggests |
| `features.*` | see `config.py` | what the classifier sees: minimum range, neighbourhood size, thresholds, neighbouring scans. A model is tied to these; changing them means retraining |
| `training.*` | 600 rounds, 10 s CV groups, 6 labels, mirror on, device auto | training options |

The topic is stored per bag and can be changed in the app (✎ on a bag).
`exclude_sessions` and `exclude_scans` in `datasets.json` name ledger
sessions and scans that training skips.

## Files

| file | purpose |
| --- | --- |
| `annotator_app.py`, `annotator_app.html` | the app: server and page |
| `train_artifact_gbdt.py` | trains the models from the ledger (the app runs it; also usable alone) |
| `classify_bag.py` | scores every scan of a bag, optionally writes a classified bag |
| `scan_grid.py` | points to grid cells, bag reading, scan cache |
| `artifact_features.py` | features of a return |
| `sensor_profile.py` | measures a sensor profile from a recording |
| `export_model.py` | writes a model as plain text for the C++ filter; makes check files |
| `config.py` | settings |

The command line tools take the same `--project`, `--config` and `--sensor`
flags as the app; `--help` on each lists the rest.

## Limits

- Range is cached in millimetres as 16 bit, so returns beyond 65.5 m
  saturate.
- The grid wraps in azimuth; a lidar with a partial field of view works but
  its edges are treated as neighbours.
- Bags are single files (`.mcap`, `.db3`). Split recordings are added file
  by file.
