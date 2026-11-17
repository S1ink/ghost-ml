#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

APT_DEPS=(python3-pip python3-venv)
MISSING_DEPS=()
for pkg in "${APT_DEPS[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_DEPS+=("$pkg")
    fi
done
if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo "Installing missing packages: ${MISSING_DEPS[*]}"
    sudo apt update
    sudo apt install -y "${MISSING_DEPS[@]}"
fi

VENV_DIR="$SCRIPTPATH/../.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."

    python3 -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    python -m pip install --upgrade pip

    # Gradient-boosted tree training (train_artifact_gbdt.py). The xgboost
    # wheel uses the GPU when one is present; scikit-learn is the fallback.
    pip install scikit-learn xgboost joblib
else
    source "$VENV_DIR/bin/activate"
fi
