# condensed-sparsity
![CI Pipeline](https://github.com/calgaryml/condensed-sparsity/actions/workflows/ci.yaml/badge.svg)
![CD Pipeline](https://github.com/calgaryml/condensed-sparsity/actions/workflows/cd.yaml/badge.svg)


## Repository Structure
* ./src/rigl_torch contains the source code for SRigL. 
* ./src/condensed_sparsity contains the source code for our naive pytorch GEMM implementation that leverages the constant fan-in structure learned by SRigL. 
* ./configs/config.yaml contains the settings for the various hyperparameters and runtime options. 

## Example Commands
* ResNet50 trained on Imagenet using 4-GPUs on a single node:

```bash
python ./train_rigl.py \
    dataset=imagenet \
    model=resnet50 \
    rigl.dense_allocation=0.01 \
    rigl.delta=800 \
    rigl.grad_accumulation_n=8 \
    rigl.min_salient_weights_per_neuron=0.3 \
    rigl.keep_first_layer_dense=True \
    rigl.use_sparse_initialization=True \
    rigl.init_method_str=grad_flow_init \
    training.batch_size=512 \
    training.max_steps=256000 \
    training.weight_decay=0.0001 \
    training.label_smoothing=0.1 \
    training.lr=0.2 \
    training.epochs=104 \
    training.warm_up_steps=5 \
    training.scheduler=step_lr_with_warm_up \
    training.step_size=[30,70,90] \
    training.gamma=0.1 \
    compute.distributed=True \
    compute.world_size=4
```

* Resnet18 trained on CIFAR-10:
```bash
python ./train_rigl.py \
  dataset=cifar10 \
  model=resnet18 \
  rigl.dense_allocation=0.01 \
  rigl.delta=100 \
  rigl.grad_accumulation_n=1 \
  rigl.min_salient_weights_per_neuron=0.05 \
  rigl.use_sparse_initialization=True \
  rigl.init_method_str=grad_flow_init \
  training.batch_size=128 \
  training.max_steps=null \
  training.weight_decay=5.0e-4 \
  training.label_smoothing=0 \
  training.lr=0.1 \
  training.epochs=250 \
  training.warm_up_steps=0 \
  training.scheduler=step_lr \
  training.step_size=77 \
  training.gamma=0.2 \
  compute.distributed=False
```

## Installation
This project was developed using Python version >=3.10 and uses `poetry` to manage dependencies and build the project. 

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

    docker build --file ./Dockerfile -t rigl-agent --shm-size=16gb .
    docker run -itd --env-file ./.env --mount source=/datasets/ILSVRC2012,target=/datasets/ILSVRC2012,type=bind --gpus all --shm-size 16G rigl-agent:latest


### Development Container
For development, we recommend using vscode's devcontainer functionality to build and launch the development container. A `devcontainer.json` schema is provided in `./.devcontainer/` and if the project working directory is opened in vscode, the application will prompt the user to reopen in the development container. Please refer to the `devcontainer.json` schema and `Dockerfile.dev` for specifics on the development container environment and build process. 

To get started, ensure you copy the `.env.template` file to your own `.env` environiment file and edit it to add environmental variables. Without the `.env` file the dev container will not start.

## Compute Canada
Compute Canada pre-builds many python packages into wheels that are stored in a local wheelhouse. It is best practice to use these wheels rather than use package distributions from PyPI. Therefore, the dependencies pinned in `pyproject.toml` have been carefully selected to ensure that the project enviornment can be replicated using the Compute Canada wheels that will match a local enviornment using PyPI package distributions. 

For simplicity, a bash script for installing the project and dependencies is included, see: `./scripts/build_cc_venv.sh`. Simply run this script from the project working directory after cloning the project from github. 

## Tests
This repository uses `pytest`.

Run tests using `pytest`
