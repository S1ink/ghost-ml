#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

source "$SCRIPTPATH/../install/setup.bash"
source "$SCRIPTPATH/../.venv/bin/activate"

# Base ROS2 launch command
BASE_CMD=(
    ros2
    launch
    ghost_annotation
    annotation.launch.py
)
"${BASE_CMD[@]}" "$@"
