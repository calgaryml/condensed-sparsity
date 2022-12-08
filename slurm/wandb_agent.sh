#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=wandb_agent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G    
#SBATCH --time=03-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani

## --- Migrate venv / data to local node storage --- ##
export WORKDIR=/home/mklasby/projects/def-yani/mklasby/condensed-sparsity/
export WANDB_SWEEP_ID=1jyv1omw
cp -r $WORKDIR/.venv $SLURM_TMPDIR/.venv

## SET ENV ##:
module load singularity python/3.8.10 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
wandb agent condensed-sparsity/condensed-rigl/${WANDB_SWEEP_ID}
