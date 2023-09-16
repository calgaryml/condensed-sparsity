#!/bin/bash
# Set up python and pip alias
cd ~
echo "alias python=python3" >> ~/.bashrc && echo "alias pip=pip3" >> ~/.bashrc

# We add a previously generated ssh key that has been added to github already and pass it via param 1
# touch ~/.ssh/id_ed22519
# cat >> ~/.ssh/id_ed22519 << EOT
# ${1} 
# EOT
# echo ${1} > ~/.ssh/id_ed22519

# Clone repo and cd into it
git clone git@github.com:calgaryml/condensed-sparsity.git
cd condensed-sparsity

# Build venv
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
rm -rf ./.venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install poetry==1.4.0
poetry install -vvv
cp -r ~/shared/ILSVRC ~/scratch/ILSVRC

# git conifg
git config --global user.name "Mike Lasby"
git config --global user.email "mklasby@gmail.com"
exec bash

