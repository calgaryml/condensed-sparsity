#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=cifar_wandb_agent
#SBATCH -p gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G    
#SBATCH --time=01-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

## --- Migrate venv / data to local node storage --- ##
export PATH=~/software/miniconda3/bin:$PATH
export WORKDIR=/work/souza_lab/lasby/condensed-sparsity
export WANDB_SWEEP_ID=1fik36lx

## SET ENV ##:
eval "$(conda shell.bash hook)"
conda activate py38
source ${WORKDIR}/.venv/bin/activate

## RUN SCRIPT ##
wandb online
wandb agent condensed-sparsity/condensed-rigl/${WANDB_SWEEP_ID}
