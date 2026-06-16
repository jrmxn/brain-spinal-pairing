#!/bin/bash
# This script sets up a local .venv using the python-311 conda environment.

# Load conda functions
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Activating conda environment: python-311..."
conda activate python-311

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'python-311'."
    exit 1
fi

echo "Creating virtual environment in .venv..."
python -m venv .venv

echo "Installing dependencies..."
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install .[dev]

# ./.venv/bin/pip uninstall jax jaxlib
# ./.venv/bin/pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "-------------------------------------------------------"
echo "Setup complete!"
echo "Activate the environment with: source .venv/bin/activate"
echo "-------------------------------------------------------"
