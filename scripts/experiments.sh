#!/bin/bash
# resnet18 models
#benchmark -> sgd, adadelta w/ weight decay, adamW
python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=benchmark_sgd \
    rigl.dense_allocation=null \
    rigl.const_fan_in=True \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=sgd

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=benchmark_adadelta \
    rigl.dense_allocation=null \
    rigl.const_fan_in=True \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=adadelta

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=benchmark_adamW \
    rigl.dense_allocation=null \
    rigl.const_fan_in=True \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=adamW

#const fan
python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=const_fan_sgd \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=True \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=sgd

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=const_fan_adadelta \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=True \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=adadelta

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=const_fan_adamW \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=True \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=adamw

#vanilla rigl
python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=vanilla_rigl_sgd \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=sgd

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=vanilla_rigl_adadelta \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=adadelta

python train_rigl.py \
    dataset=cifar10 \
    model=resnet18 \
    experiment.comment=vanilla_rigl_adamw \
    rigl.dense_allocation=0.1 \
    rigl.const_fan_in=False \
    training.weight_decay=0.0005 \
    training.momentum=0.9 \
    training.optimizer=adamw

