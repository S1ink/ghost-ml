#!/bin/bash
# Starts the lidar artifact annotator, then open http://localhost:8765
#
#   ./run.sh --project <project dir> [--bags-root <folder with bags>] [--port 9000] ...
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
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
if ! python3 -c 'import rosbag2_py' 2>/dev/null; then
    echo "rosbag2_py is not importable: source your ROS 2 setup.bash first." >&2
    exit 1
fi
exec python3 -u "$HERE/annotator_app.py" "$@"
