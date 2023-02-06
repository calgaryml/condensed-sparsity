#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=run_train_rigl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G    
#SBATCH --time=03:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani

## --- Migrate venv / data to local node storage --- ##
export WORKDIR=/home/mklasby/projects/def-yani/mklasby/condensed-sparsity/
# export PROJECT_OUTPUT_DIR=${WORKDIR}/slurm/outputs/${SLURM_JOB_ID}_output
# export WANDB_DIR=${SLURM_TMPDIR}/output/wandb
# mkdir ${SLURM_TMPDIR}/output/
# mkdir ${WANDB_DIR}
# mkdir -p ${PROJECT_OUTPUT_DIR}
# mkdir ${SLURM_TMPDIR}/data
cp -r ${WORKDIR}/.venv ${SLURM_TMPDIR}


# ## SET ENV ##:
module load python/3.10.2 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
id=$1
printf "Id recieved: ${id}"
wandb online
python3 ./train_rigl.py \
    experiment.resume_from_checkpoint=True \
    experiment.run_id=${id} \
    compute.distributed=False
