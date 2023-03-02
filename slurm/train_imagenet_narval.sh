#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=imagenet_x1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=510000M
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-rmsouza


## --- Migrate data to local node storage --- ##
export SCRATCH=/scratch/mklasby/ILSVRC/
export WORKDIR=/home/mklasby/projects/def-yani/mklasby/condensed-sparsity/
cp -r $WORKDIR/.venv $SLURM_TMPDIR/.venv
cp $SCRATCH/ILSVRC2012_devkit_t12.tar.gz $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_train.tar $SLURM_TMPDIR
cp $SCRATCH/ILSVRC2012_img_val.tar $SLURM_TMPDIR

## SET ENV ##:
module load singularity python/3.8.10 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
dense_alloc=$1
printf "Starting run with dense alloc == ${dense_alloc}\n"
wandb offline

python3 ${WORKDIR}/train_rigl.py \
dataset=imagenet \
model=resnet50 \
rigl.dense_allocation=${dense_alloc} \
rigl.delta=400 \
rigl.grad_accumulation_n=4 \
rigl.min_salient_weights_per_neuron=0.3 \
training.batch_size=1024 \
training.max_steps=128000 \
training.weight_decay=0.0001 \
training.label_smoothing=0.1 \
training.seed=8746 \
training.lr=0.4 \
training.epochs=103 \
training.warm_up_steps=5 \
training.scheduler=step_lr_with_warm_up \
training.step_size=[30,70,90] \
training.gamma=0.1 \
training.log_interval=500 \
compute.world_size=4 \
compute.distributed=True \
rigl.use_sparse_initialization=True \
rigl.init_method_str=grad_flow_init
