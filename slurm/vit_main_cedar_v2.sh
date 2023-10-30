#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=imagenet_vit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=125G    
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani


## --- Migrate data to local node storage --- ##
export SCRATCH=/scratch/mklasby/ILSVRC/
export WORKDIR=/home/mklasby/projects/def-yani/mklasby/condensed-sparsity/
cp -r $WORKDIR/.venv $SLURM_TMPDIR/.venv
cp $SCRATCH/ILSVRC2012_devkit_t12.tar.gz $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_train.tar $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_val.tar $SLURM_TMPDIR

## SET ENV ##:
module load singularity python/3.10.2 cuda/11.7 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
# dense_alloc=$1
# printf "Starting run with dense alloc == ${dense_alloc}\n"

python3 ${WORKDIR}/train_rigl.py \
experiment.resume_from_checkpoint=True \
experiment.run_id=jnk0cgjt

