#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=train_imagenet
#SBATCH --ntasks=1
#SBATCH -p gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

## SET ENV ##:
export PATH=~/software/miniconda3/bin:$PATH
export WORKDIR=/work/souza_lab/lasby/condensed-sparsity
eval "$(conda shell.bash hook)"
conda activate py38
source ${WORKDIR}/.venv/bin/activate

## RUN SCRIPT ##
python ${WORKDIR}/train_rigl.py
