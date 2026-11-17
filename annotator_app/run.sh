#!/bin/bash
# Starts the lidar artifact annotator, then open http://localhost:8765
#
#   ./run.sh [--project <project dir>] [--bags-root <folder with bags>] [--port 9000] ...
#
# Without --project the page offers to choose or set up a project.
#
# Needs a ROS 2 Python environment for reading bags (rosbag2_py). If none is
# active, the newest one under /opt/ros is sourced. Python packages beyond
# ROS (requirements.txt) may live in a virtual environment: .venv next to
# this script, or the one named by ANNOTATOR_VENV.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
if ! python3 -c 'import rosbag2_py' 2>/dev/null; then
    for setup in $(ls -d /opt/ros/*/setup.bash 2>/dev/null | sort -r); do
        source "$setup" && break
    done
fi
VENV="${ANNOTATOR_VENV:-$HERE/.venv}"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    # A virtual environment only works with the Python version that made it.
    PYV=$(python3 -c 'import sys; print("python%d.%d" % sys.version_info[:2])')
    if [ ! -d "$VENV/lib/$PYV" ]; then
        MADE=$(ls "$VENV/lib" 2>/dev/null | head -1)
        echo "The Python environment $VENV was made with $MADE, but python3 here is $PYV," >&2
        echo "so its packages cannot be used. It was probably created inside a container:" >&2
        echo "start this from the same place, or create an environment for this Python." >&2
        exit 1
    fi
fi
if ! python3 -c 'import rosbag2_py' 2>/dev/null; then
    echo "rosbag2_py is not importable: source your ROS 2 setup.bash first." >&2
    exit 1
fi
MISSING=$(python3 - <<'PY'
import importlib.util
print(" ".join(m for m in ("numpy", "joblib", "sklearn") if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$MISSING" ]; then
    echo "Missing Python packages: $MISSING" >&2
    echo "Install them with: pip install -r $HERE/requirements.txt" >&2
    echo "(inside the ghost workspace: run src/setup_venv.sh once)" >&2
    exit 1
fi
exec python3 -u "$HERE/annotator_app.py" "$@"
