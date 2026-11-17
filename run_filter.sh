#!/bin/bash
# Launches the C++ nodes of ghost_tester (the lidar artifact filter), usually
# with a bag playing into them:
#
#   ./run_filter.sh bag:=<bag> [model:=<artifact_xyz.model.txt>] [filter:=lance1] ...
#
# Without model:= the workspace's models/artifact_xyz.model.txt is used.
# Every other argument is <action>:=<preset>; see
# ghost_tester/launch/ghost_tester.launch.py for the list.

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
WS=$(dirname "$SCRIPTPATH")

source "$WS/install/setup.bash"

ARGS=("$@")
case " $* " in
    *" model:="*) ;;
    *) ARGS+=("model:=$WS/models/artifact_xyz.model.txt") ;;
esac

# Base ROS2 launch command
BASE_CMD=(
    ros2
    launch
    ghost_tester
    ghost_tester.launch.py
)
"${BASE_CMD[@]}" "${ARGS[@]}"
