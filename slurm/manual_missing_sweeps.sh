#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=missing_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G    
#SBATCH --time=06:00:00
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


## SET ENV ##:
module load python/3.8.10 cuda/11.4 cudnn
source ${SLURM_TMPDIR}/.venv/bin/activate

## RUN SCRIPT ##
wandb online
# python3 ./train_rigl.py "${@}"
    # experiment.resume_from_checkpoint=True \
    # experiment.run_id=8euvnykc \
    # dataset=cifar10 \
    # model=resnet18 \
    # rigl.dense_allocation=0.01 \
    # rigl.delta=100 \
    # rigl.grad_accumulation_n=1 \
    # rigl.min_salient_weights_per_neuron=1 \
    # training.batch_size=128 \
    # training.max_steps=null \
    # training.weight_decay=5.0e-4 \
    # training.label_smoothing=0 \
    # training.lr=0.1 \
    # training.epochs=250 \
    # training.warm_up_steps=0 \
    # training.scheduler=step_lr \
    # training.step_size=77 \
    # training.gamma=0.2 \
    # compute.distributed=False \
    # rigl.use_sparse_initialization=True \
    # rigl.init_method_str=grad_flow_init \
    # training.seed=42

python train_rigl.py compute.distributed=False dataset=cifar10 model=resnet18 rigl.const_fan_in=True rigl.delta=100 rigl.dense_allocation=0.4 rigl.grad_accumulation_n=1 rigl.init_method_str=grad_flow_init rigl.min_salient_weights_per_neuron=0.005 rigl.use_sparse_initialization=True training.batch_size=128 training.epochs=250 training.gamma=0.2 training.label_smoothing=0 training.lr=0.1 training.max_steps=None training.scheduler=step_lr training.seed=2078 training.step_size=77 training.warm_up_steps=0 training.weight_decay=0.0005
