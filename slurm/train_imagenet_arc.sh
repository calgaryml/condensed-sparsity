#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=train_imagenet
#SBATCH --ntasks=1
#SBATCH -p gpu-v100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL

## SET VARS & MOVE DATA TO NODE ##:
export PATH=~/software/miniconda3/bin:$PATH
export WORKDIR=/work/souza_lab/lasby/condensed-sparsity
export SLURM_TMPDIR=/dev/shm
export SCRATCH=/work/souza_lab/Data/ILSVRC2012

# cp $SCRATCH/ILSVRC2012_devkit_t12.tar.gz $SLURM_TMPDIR
# cp $SCRATCH/ILSVRC2012_img_train.tar $SLURM_TMPDIR
# cp $SCRATCH/ILSVRC2012_img_val.tar $SLURM_TMPDIR
cp -r $WORKDIR/.venv $SLURM_TMPDIR

## SET ENV ##:
eval "$(conda shell.bash hook)"
conda activate py38
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
rigl.min_salient_weights_per_neuron=0.5 \
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
compute.world_size=2 \
compute.distributed=True \
rigl.use_sparse_initialization=True \
rigl.init_method_str=grad_flow_init
