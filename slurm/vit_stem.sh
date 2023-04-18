#!/bin/bash
# dense_allocs=(0.2 0.1 0.05 0.01)
dense_allocs=(0.2)
for da in ${dense_allocs[@]}
do
    sbatch ./slurm/vit_main_narv.sh $da
done
