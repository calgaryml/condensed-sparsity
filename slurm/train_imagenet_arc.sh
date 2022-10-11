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
export SLURM_TMPDIR=/dev/shm
export SCRATCH=/work/souza_lab/Data/ILSVRC2012

cp $SCRATCH/ILSVRC2012_devkit_t12.tar.gz $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_train.tar $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_val.tar $SLURM_TMPDIR
cp -r $WORKDIR/.venv $SLURM_TMPDIR

eval "$(conda shell.bash hook)"
conda activate py38
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
python ${WORKDIR}/train_rigl.py
