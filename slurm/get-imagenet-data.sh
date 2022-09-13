#!/bin/bash

## GET RESOURCES ##

# SBATCH --job-name=get-imagenet-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G    
#SBATCH --time=1-12:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

## RUN SCRIPT ##
# cd ~/slurm
# mkdir ILSVRC && cd ./ILSVRC
# wget  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
# wget  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# wget  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

export SCRATCH=/scratch/mklasby/ILSVRC
cp $SCRATCH/ILSVRC2012_devkit_t12.tar.gz $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_train.tar $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_val.tar $SLURM_TMPDIR

export WORKDIR=/home/mklasby/projects/def-rmsouza/mklasby/condensed-sparsity
module load singularity python/3.8.10 cuda cudnn
source ${WORKDIR}/.venv/bin/activate
python3 ${WORKDIR}/scripts/extract_imagenet.py >/dev/null 2>&1

# tar -xf ./ILSVRC2012_img_val.tar
# tar -xf ./ILSVRC2012_img_train.tar
# tar -xf ./ILSVRC2012_devkit_t12.tar.gz
# Seems like its best to let pytorch setup the folder per ImageFolder API
