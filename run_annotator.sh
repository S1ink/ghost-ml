#!/bin/bash
# Starts the lidar artifact annotator (annotator_app/) for this workspace,
# then open http://localhost:8765
#
#   ./run_annotator.sh [--bags-root <folder with bags>] [--port 9000] ...
#
# Uses the Python environment setup_venv.sh creates, <workspace>/.venv, where
# the workspace is the directory above this one. The workspace is also the
# project (labels, cache, models, exports; settings in its annotator.json).
# On the first start the page asks which lidar it is and where the bags are.
# Pass --project <dir> to use another project. See annotator_app/README.md.

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
WS=$(dirname "$SCRIPTPATH")

export ANNOTATOR_VENV="$WS/.venv"
if [ ! -d "$ANNOTATOR_VENV" ]; then
    echo "No Python environment at $ANNOTATOR_VENV. Create it first:" >&2
    echo "    $SCRIPTPATH/setup_venv.sh" >&2
    exit 1
fi

case " $* " in
    *" --project "*|*" --project="*) exec "$SCRIPTPATH/annotator_app/run.sh" "$@" ;;
    *) exec "$SCRIPTPATH/annotator_app/run.sh" --project "$WS" "$@" ;;
esac
