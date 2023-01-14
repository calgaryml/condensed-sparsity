#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=train_imagenet_x1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G    
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani
#SBATCH --gres=gpu:v100l:2


## --- Migrate data to local node storage --- ##
export SCRATCH=/scratch/mklasby/ILSVRC/
export WORKDIR=/home/mklasby/projects/def-rmsouza/mklasby/condensed-sparsity/
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

python3 ${WORKDIR}/train_rigl.py \
dataset=imagenet \
model=resnet50 \
rigl.dense_allocation=${dense_alloc} \
rigl.delta=800 \
rigl.grad_accumulation_n=8 \
rigl.min_salient_weights_per_neuron=0.3 \
training.batch_size=512 \
training.max_steps=256000 \
training.weight_decay=0.0001 \
training.label_smoothing=0.1 \
training.seed=8746 \
training.lr=0.2 \
training.epochs=104 \
training.warm_up_steps=5 \
training.scheduler=step_lr_with_warm_up \
training.step_size=[30,70,90] \
training.gamma=0.1 \
training.world_size=2 \
compute.distributed=True \
rigl.use_sparse_initialization=True \
rigl.init_method_str=grad_flow_init
