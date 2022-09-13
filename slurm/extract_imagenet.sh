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
export WORKDIR=/home/mklasby/projects/def-rmsouza/mklasby/condensed-sparsity/
python3 ${WORKDIR}/scripts/extract_imagenet.py
