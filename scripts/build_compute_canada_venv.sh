#!/bin/bash
module load singularity python/3.8.10 cuda cudnn
rm -rf ./.venv
virtualenv .venv
source .venv/bin/activate
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pip install --upgrade pip
pip install poetry==1.2.0
poetry config experimental.new-installer false
poetry config virtualenvs.options.always-copy true
cd ./bin/
pip download --no-deps torchvision==0.13.1
pip download --no-deps torch==1.12.1
pip download --no-deps Pillow==9.2.0
pip download --no-deps Pillow_SIMD==9.0.0.post1+computecanada
pip download --no-deps scipy==1.8.0
pip download --no-deps grpcio==1.47.0+computecanada
pip download --no-deps numpy==1.23.0+computecanada
pip download --no-deps pandas==1.4.1+computecanada
cd ..
poetry install -vvv
