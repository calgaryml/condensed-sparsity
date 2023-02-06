#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=imagenet_itop
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
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
module load python/3.10.2 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
dense_alloc=$1
printf "Starting run with dense alloc == ${dense_alloc}\n"

python3 ${WORKDIR}/train_rigl.py \
dataset=imagenet \
model=resnet50 \
rigl.dense_allocation=${dense_alloc} \
rigl.delta=4000 \
rigl.grad_accumulation_n=1 \
rigl.alpha=0.5 \
rigl.const_fan_in=True \
rigl.use_t_end=False \
rigl.dynamic_ablation=True \
rigl.min_salient_weights_per_neuron=0.3 \
rigl.use_sparse_initialization=True \
rigl.init_method_str=grad_flow_init \
rigl.keep_first_layer_dense=False \
training.batch_size=64 \
training.epochs=104 \
training.log_interval=1000 \
training.max_steps=2048000 \
training.optimizer=sgd \
training.weight_decay=0.0001 \
training.label_smoothing=0.1 \
training.lr=0.1 \
training.warm_up_steps=5 \
training.scheduler=step_lr_with_warm_up \
training.step_size=[30,70,90] \
training.gamma=0.1 \
compute.distributed=False \
experiment.comment="_ITOPx1"
