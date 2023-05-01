#!/bin/bash
# TODO: Check where we end up here and test this!
cd ~
# We add a previously generated ssh key that has been added to github already and pass it via param 1
touch ~/.ssh/id_ed22519
cat >> ~/.ssh/id_ed22519 << EOT
${1} 
EOT
echo ${1} > ~/.ssh/id_ed22519
# Clone repo and cd into it
git clone git@github.com:calgaryml/condensed-sparsity.git
cd condensed-sparsity
# Build venv
rm -rf ./.venv
sudo y | apt-get install python3.8-venv
python -m venv .venv
source .venv/bin/activate
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
poetry config virtualenvs.options.always-copy true
poetry install -vvv --only-root
