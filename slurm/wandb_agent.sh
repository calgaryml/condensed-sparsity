#!/bin/bash

## GET RESOURCES ##

# SBATCH --job-name=wandb_agent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G    
#SBATCH --time=0-04:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
## SET ENV ##:
module load singularity python/3.8.10 cuda cudnn
source ../.venv/bin/activate

## RUN SCRIPT ##
# wandb agent condensed-sparsity/condensed-rigl/4hsfzsa9
python3 ../unetter/validation/test_all_models.py