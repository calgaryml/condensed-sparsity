#!/bin/bash
# resnet18 models
#benchmark -> sgd, adadelta w/ weight decay, adamW
python train_rigl.py \
    dataset=cifar10 \
    model=wide_resnet22 \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.seed=42

python train_rigl.py \
    dataset=cifar10 \
    model=wide_resnet22 \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.seed=2078

python train_rigl.py \
    dataset=cifar10 \
    model=wide_resnet22 \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.seed=7303

python train_rigl.py \
    dataset=cifar10 \
    model=wide_resnet22 \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.seed=6037

python train_rigl.py \
    dataset=cifar10 \
    model=wide_resnet22 \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.seed=8746

## 
# python train_rigl.py \
#     dataset=cifar10 \
#     model=wide_resnet22 \
#     rigl.dense_allocation=0.01 \
#     rigl.const_fan_in=False \
#     training.seed=8746

# python train_rigl.py \
#     dataset=cifar10 \
#     model=wide_resnet22 \
#     rigl.dense_allocation=0.05 \
#     rigl.const_fan_in=True \
#     training.seed=42

# python train_rigl.py \
#     dataset=cifar10 \
#     model=wide_resnet22 \
#     rigl.dense_allocation=0.05 \
#     rigl.const_fan_in=True \
#     training.seed=6037

# python train_rigl.py \
#     dataset=cifar10 \
#     model=wide_resnet22 \
#     rigl.dense_allocation=0.05 \
#     rigl.const_fan_in=True \
#     training.seed=8746

# python train_rigl.py \
#     dataset=cifar10 \
#     model=wide_resnet22 \
#     rigl.dense_allocation=0.01 \
#     rigl.const_fan_in=True \
#     training.seed=42