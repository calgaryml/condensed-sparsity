#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=imagenet_agent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=32G    
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani

## --- Migrate venv / data to local node storage --- ##
export SCRATCH=/scratch/mklasby/ILSVRC/
export WORKDIR=/home/mklasby/projects/def-yani/mklasby/condensed-sparsity/
export WANDB_SWEEP_ID=aw7o7hz7
cp -r $WORKDIR/.venv $SLURM_TMPDIR/.venv
cp $SCRATCH/ILSVRC2012_devkit_t12.tar.gz $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_train.tar $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_val.tar $SLURM_TMPDIR

## SET ENV ##:
module load python/3.8.10 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
wandb online
wandb agent condensed-sparsity/condensed-rigl/${WANDB_SWEEP_ID}
