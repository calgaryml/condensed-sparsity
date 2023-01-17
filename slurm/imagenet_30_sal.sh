#!/bin/bash
# run_ids=("hkak5806" "p5n21r8q" "bbkeuxvr" "1hfmurvv")
dense_allocs=(0.01 0.05 0.1 0.2)
# dense_allocs=(0.2 0.05)
for da in ${dense_allocs[@]}
do
    sbatch ./slurm/train_imagenet_arc.sh $da
done
