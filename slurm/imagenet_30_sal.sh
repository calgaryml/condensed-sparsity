#!/bin/bash
dense_allocs=(0.2 0.1 0.05 0.01)
for da in ${dense_allocs[@]}
do
    sbatch ./slurm/train_imagenet_x2.sh $da
done
