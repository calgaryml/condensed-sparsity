#!/bin/bash
module load singularity python/3.8.10 cuda cudnn
rm -rf ./.venv
virtualenv .venv
source .venv/bin/activate
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pip install --upgrade pip
pip install -r requirements.txt
poetry config virtualenvs.options.always-copy true
poetry install -vvv --only-root
