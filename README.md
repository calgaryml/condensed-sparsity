# condensed-sparsity
![CI Pipeline](https://github.com/calgaryml/condensed-sparsity/actions/workflows/ci.yaml/badge.svg)
![CD Pipeline](https://github.com/calgaryml/condensed-sparsity/actions/workflows/cd.yaml/badge.svg)


## Installation
This project was developed using Python version 3.8.10 and uses `poetry` to manage dependencies and build the project. 

Installation instructions are provided for virtual enviornments, Compute Canada clusters, and Docker: 

## Virtual Env
        python -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install poetry==1.2.0
        poetry install -vvv  # With install all dependency groups
        pre-commit install-hooks  # For development

## Docker
For reproduction and instantiating replica's during training / inference, please use `Dockerfile` to build the image. Alternatively, you can pull the image from [Docker Hub](https://hub.docker.com/repository/docker/mklasby/condensed-sparsity). A large shm-size is required for pytorch to train ImageNet as this directory is used by the dataloader workers

### Replica / Reproduction Container

    docker build --file ./Dockerfile --shm-size 2G -t mklasby:condensed-sparsity
    docker run ??? TBD

### Development Container
For development, we recommend using vscode's devcontainer functionality to build and launch the development container. A `devcontainer.json` schema is provided in `./.devcontainer/` and if the project working directory is opened in vscode, the application will prompt the user to reopen in the development container. Please refer to the `devcontainer.json` schema and `Dockerfile.dev` for specifics on the development container environment and build process. 

## Compute Canada
Compute Canada pre-builds many python packages into wheels that are stored in a local wheelhouse. It is best practice to use these wheels rather than use package distributions from PyPI. Therefore, the dependencies pinned in `pyproject.toml` have been carefully selected to ensure that the project enviornment can be replicated using the Compute Canada wheels that will match a local enviornment using PyPI package distributions. 

For simplicity, a bash script for installing the project and dependencies is included, see: `./scripts/build_compute_canada_venv.sh`. Simply run this script from the project working directory after cloning the project from github. 

## Tests
This repository uses `pytest`.

Run tests using `pytest`
