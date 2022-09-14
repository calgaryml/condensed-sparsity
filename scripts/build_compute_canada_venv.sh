#!/bin/bash
module load singularity python/3.8.10 cuda cudnn
rm -rf ../.venv
virtualenv .venv
source .venv/bin/activate
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pip install --upgrade pip
pip install poetry==1.2.0
poetry config experimental.new-installer false
pip download --no-deps torchvision==0.13.1
pip download --no-deps torch==1.12.1
poetry install -vvv
