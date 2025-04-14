#!/bin/bash

# Check if the hbie environment already exists
if conda info --envs | grep -q "^hbie"; then
  # Remove the existing hbie environment
  conda env remove --name hbie -y
fi

# Create the hbie environment with Python 3.11
conda create --name hbie python=3.11 -y

# Activate the hbie environment
source activate hbie

# Verify that the environment was activated
if [[ "$CONDA_DEFAULT_ENV" == "hbie" ]]; then
  echo "Environment hbie activated successfully."

  # Install the required packages in development mode
  pip install .[dev]
else
  echo "Failed to activate environment hbie."
  exit 1
fi

# pip uninstall jax jaxlib

# pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
