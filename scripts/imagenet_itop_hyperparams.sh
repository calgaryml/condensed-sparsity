#!/bin/bash

dense_allocs=(0.2 0.1 0.05 0.01)
for da in ${dense_allocs[@]}
do
    sbatch ./slurm/imagenet_itop.sh $da
done
