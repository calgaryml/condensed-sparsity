#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=run_train_rigl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G    
#SBATCH --time=01:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani

## --- Migrate venv / data to local node storage --- ##
export WORKDIR=/home/mklasby/projects/def-yani/mklasby/condensed-sparsity/
export PROJECT_OUTPUT_DIR=${WORKDIR}/slurm/outputs/${SLURM_JOB_ID}_output
mkdir ${SLURM_TMPDIR}/output/
mkdir -p ${PROJECT_OUTPUT_DIR}
mkdir ${SLURM_TMPDIR}/data
cp -r $WORKDIR/.venv ${SLURM_TMPDIR}/.venv
cp -r ${WORKDIR}/data ${SLURM_TMPDIR}/data



## SET ENV ##:
module load singularity python/3.8.10 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
# wandb agent condensed-sparsity/condensed-rigl/1jyv1omw

python3 ./train_rigl.py training.dry_run=True experiment.comment=cedar_speed_test
cp -r ${SLURM_TMPDIR}/output  ${PROJECT_OUTPUT_DIR}

