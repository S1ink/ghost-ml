# ghost-ml

Finds erroneous lidar returns ("ghost points") with a trained classifier:
label recorded bags in a web app, train a model, and run it in C++ on live
scans.

| folder | what it is |
| --- | --- |
| `annotator_app/` | the web app: label bags, train, review, export. Plain Python, not a ROS package. [README](annotator_app/README.md) |
| `ghost_tester/` | ROS 2 C++ package with the realtime filter example (`artifact_filter_node`) and the C++/Python check, built against `multiscan_driver` |
| `ghost_annotation/` | the earlier Foxglove and joystick annotator (ROS 2 package) |
| `launch_utils/` | launch helpers (git submodule) |
| `csm_utils/` | shared C++ utility headers and CMake helpers, used by `ghost_tester` (git submodule) |
| `multiscan_driver/`, `csm_metrics/` | the multiScan driver, which owns the C++ classifier and ships the default models, and its metrics dependency (git submodules, see [The driver](#the-driver)) |

## Quick start

Needs ROS 2 (Jazzy or later) and Python 3. This folder is meant to be the
`src/` of a workspace; everything else is created next to it:

```
<workspace>/
  src/            this repository
  .venv/          Python environment      (setup_venv.sh)
  build/ install/ ROS build               (build_ros.sh)
  annotator.json  cache/ models/ out/ ... the annotator's project files
```

```bash
git clone --recurse-submodules <this repository> <workspace>/src
cd <workspace>
source /opt/ros/$ROS_DISTRO/setup.bash
./src/setup_venv.sh          # once: creates .venv with the Python packages
./src/run_annotator.sh       # then open http://localhost:8765
```

On the first start the page asks which lidar it is (SICK multiScan is built
in; any other is measured from one of your recordings) and where your bags
are. After that:

1. **Add** a bag. It is read once into a cache.
2. **Label**: select the bad points in the range image or the 3D view and
   press Enter to mark the scan complete. Help in the app lists the controls.
3. **Train** (about a minute). The model's proposals then appear on every
   scan, and the review queue suggests where labelling helps most.
4. Repeat 2 and 3 until the proposals are right.

Run the scripts from the same Python that made `.venv`. If it was made
inside a container, start the app inside that container too.

## Running the model in C++

A model for a live sensor may only look at earlier scans. Under **Train**,
set "scans used" to **before only (realtime)** and train. Next to the model
this writes a plain text copy that needs no Python:

```
<workspace>/models/artifact_xyz.model.txt    range only
<workspace>/models/artifact_full.model.txt   also uses intensity
```

Build, then launch the filter with a bag playing into it:

```bash
cd <workspace>
./src/build_ros.sh
./src/run_filter.sh bag:=<a recording>
```

`run_filter.sh` uses `<workspace>/models/artifact_xyz.model.txt` unless you
pass `model:=<file>`. Further arguments choose presets from
`ghost_tester/config/`, in the same way as the annotation launch:

| argument | effect |
| --- | --- |
| `filter:=lance1` or `lance2` | also start the Foxglove bridge and that robot's transforms |
| `filter:=debug` | Foxglove bridge, exact probabilities, and the `classified` cloud |
| `artifact_filter:=multiscan_debug` | only the node's debug settings |
| `bag_play:=lidar_loop` | loop the bag (`all` plays every topic, starting paused) |
| `test_node:=enabled` | also run the PCL shadow point test node |

The Foxglove bridge listens on port 8765, the same port as the annotator
app, so do not run both at once without changing one of them.

To run the node by itself on a live topic:

```bash
source install/setup.bash
ros2 run ghost_tester artifact_filter_node --ros-args \
    -p model_file:=$PWD/models/artifact_xyz.model.txt \
    -p input_topic:=/multiscan/lidar_scan
```

It publishes `/artifact_filter/filtered` (the scan without the flagged
points) and `/artifact_filter/artifacts` (the flagged points). The other
parameters (`threshold`, `early_exit`, `keep_near`, `publish_classified`)
are described at the top of `ghost_tester/src/artifact_filter_node.cpp`.

**Speed.** Most points are clearly clean after a fraction of the trees, so
the node stops evaluating them early (`early_exit`, on by default; the
cut-offs are measured at training and stored in the model file). On a
desktop CPU a 5,000 point scan takes about 10 to 15 ms on one core with a
600 tree model, against about 50 ms without it, and the same points were
flagged in every test. Probabilities of points under the threshold are then
approximate; set `early_exit:=false` when you need them exact. The work is
spread over all cores with OpenMP when available. If that is still too slow
on the target, fewer trees (`"training": {"rounds": 300}` in
`annotator.json`) cost time in proportion and a little accuracy.

To use the classifier in your own node, include
`<multiscan_driver/artifact_filter/filter.hpp>` (from the `multiscan_driver`
package). It is header only, with no dependencies beyond the standard
library:

```cpp
artifact_filter::Filter filter{artifact_filter::Model::load(path)};
filter.setEarlyExit(true);  // optional: several times faster, see Speed
// for every scan, in order:
filter.process(xyz, num_points, intensity_or_null, stamp_ns, point_probability);
// point_probability[i] >= filter.model().threshold  ->  point i is an artifact
```

A lidar that sends each scan in pieces (the multiScan sends 12 UDP segments
per revolution) can feed them as they arrive: `begin`, then per piece `add`,
`cover` for every beam it carried and `classifyReady`, then `finish`. A
cell is classified as soon as every cell its inputs look at is complete, so
only the cells next to the last piece are left when the scan ends. The
result is identical to `process` (which is built on these calls). On the
multiScan about 21 % of the cells wait for the last segment: rings 0 to 6
and 7 to 13 are sent half a revolution apart and the middle rings look at
both. See `Filter::begin` in the header.

The header repeats the Python feature code, so the two must stay in step.
To check that they do, on your own data:

```bash
source .venv/bin/activate
python3 src/annotator_app/export_model.py --project . --variant xyz \
    --check-file /tmp/check.bin --bag <a recording>
ros2 run ghost_tester artifact_filter_check models/artifact_xyz.model.txt /tmp/check.bin
```

This compares the C++ result with Python's for the same scans, and prints
the time per scan with and without early exit and whether early exit
changed any decision. Run it after changing `artifact_features.py` or
`scan_grid.py` here, or the header in the driver.

## The driver

The robot runs the classifier inside `multiscan_driver` (its
`ghost_filter.*` parameters), which filters every frame segment by segment
as the sensor sends it. The driver owns the C++ side:

- `include/multiscan_driver/artifact_filter/`: the classifier, which
  `ghost_tester` builds against, so the tools here run exactly the code
  the robot runs;
- `models/`: the default models (`artifact_full`, `artifact_xyz`), which
  it installs and loads by name.

So this workspace depends on the driver, and the robot needs nothing from
here. Get the driver either as submodules of this repository:

```bash
cd <workspace>/src
git submodule add https://github.com/Cardinal-Space-Mining/multiscan-driver multiscan_driver
git submodule add https://github.com/Cardinal-Space-Mining/csm-metrics csm_metrics
```

or by building on top of a workspace that has it (the robot's), e.g.
`source <lance-ws>/install/setup.bash` before `./src/build_ros.sh`.

To make newly trained models the robot's defaults, publish them into the
driver, then rebuild and test it and commit the models there. In the app,
open Files ("Manage files…"): "Driver defaults" compares the models in use
with the driver's and has a "Publish to driver…" button. Or from a shell:

```bash
source .venv/bin/activate
python3 src/annotator_app/publish_model.py      # --driver <path> if not the submodule
```

Both publish into the `multiscan_driver` submodule unless the project's
`annotator.json` sets `"driver"` to another checkout.

`publish_model.py` copies `models/artifact_full.*` and
`models/artifact_xyz.*` (model and training report) after checking that
they are realtime models with early exit checkpoints.

## Other scripts

| script | purpose |
| --- | --- |
| `setup_venv.sh` | creates `<workspace>/.venv` |
| `run_annotator.sh` | starts the web app with the workspace as its project |
| `build_ros.sh` | `colcon build` of the ROS packages (run from the workspace) |
| `run_filter.sh` | launches the C++ filter, usually with `bag:=<recording>` |
| `run_ros.sh` | launches the earlier Foxglove annotator |
| `annotator_app/publish_model.py` | makes the trained models the driver's defaults |
