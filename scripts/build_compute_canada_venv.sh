#!/bin/bash
module load singularity python/3.10.2 cuda/11.4 cudnn
rm -rf ./.venv
virtualenv .venv
source .venv/bin/activate
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
poetry config virtualenvs.options.always-copy true
poetry install -vvv --only-root
