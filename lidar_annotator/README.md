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

## Starting

A **project** is a directory that holds everything belonging to one sensor
setup: labels, the list of bags, the scan cache, models and exports. Any
empty directory will do.

First start with a new lidar, measuring its scan geometry from a recording:

```bash
./run.sh --project ~/lidar-project --bags-root /path/to/bags \
         --sensor-from-bag /path/to/bags/some_recording.mcap
```

Afterwards, and for a lidar with a shipped profile (`--sensor multiscan`):

```bash
./run.sh --project ~/lidar-project --bags-root /path/to/bags
```

Then open http://localhost:8765. The Help button in the app explains the
controls. Flags you use every time can go in `<project>/annotator.json`.

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
