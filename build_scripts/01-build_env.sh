#!/bin/bash -e

# ENV_NAME must be defined
if [[ -z "${ENV_NAME}" ]]; then
  echo "Please, provide ENV_NAME environment variable"
  exit 1
fi

# source $CONDA_PREFIX_1/bin/activate

conda create -n "$ENV_NAME" python=3.9 pip -y --force
# conda activate $ENV_NAME

echo "Installing omniscripts and modin at the same time to avoid conflicts"
conda run -n $ENV_NAME pip install .[reporting] "modin[all] @ git+https://github.com/modin-project/modin"
