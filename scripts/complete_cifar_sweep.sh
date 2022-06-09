#!/bin/bash
# resnet18 models
#benchmark -> sgd, adadelta w/ weight decay, adamW
# python train_rigl.py \
#     dataset=cifar10 \
#     model=resnet18 \
#     rigl.dense_allocation=0.3 \
#     rigl.const_fan_in=True \
#     training.seed=6037

# python train_rigl.py \
#     dataset=cifar10 \
#     model=resnet18 \
#     rigl.dense_allocation=0.3 \
#     rigl.const_fan_in=True \
#     training.seed=8746

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    rigl.dense_allocation=0.2 \
    rigl.const_fan_in=True \
    training.seed=42

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    rigl.dense_allocation=0.2 \
    rigl.const_fan_in=True \
    training.seed=2078

# python train_rigl.py \
#     dataset=cifar10 \
#     model=resnet18 \
#     rigl.dense_allocation=0.2 \
#     rigl.const_fan_in=True \
#     training.seed=7303

# python train_rigl.py \
#     dataset=cifar10 \
#     model=resnet18 \
#     rigl.dense_allocation=0.01 \
#     rigl.const_fan_in=True \
#     training.seed=6037

# python train_rigl.py \
#     dataset=cifar10 \
#     model=resnet18 \
#     rigl.dense_allocation=0.01 \
#     rigl.const_fan_in=True \
#     training.seed=8746
