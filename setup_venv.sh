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

    # Determine which PyTorch wheel to install
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')

        case "$CUDA_VERSION" in
            13.2*)
                TORCH_CUDA="cu132"
                ;;
            12.8*|12.9*)
                TORCH_CUDA="cu128"
                ;;
            12.6*|12.7*)
                TORCH_CUDA="cu126"
                ;;
            12.4*|12.5*)
                TORCH_CUDA="cu124"
                ;;
            *)
                echo "Unsupported or unknown CUDA version ($CUDA_VERSION)."
                echo "Installing CPU-only PyTorch."
                TORCH_CUDA=""
                ;;
        esac
    else
        echo "No NVIDIA GPU detected."
        TORCH_CUDA=""
    fi

    if [ -n "$TORCH_CUDA" ]; then
        echo "Installing PyTorch ($TORCH_CUDA)..."
        pip install torch torchvision \
            --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    else
        pip install torch torchvision
    fi

    pip install h5py scikit-learn
else
    source "$VENV_DIR/bin/activate"
fi
