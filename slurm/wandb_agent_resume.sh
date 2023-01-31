#!/bin/bash
# run_ids=("hkak5806" "p5n21r8q" "bbkeuxvr" "1hfmurvv")
run_ids=("lx23b1mp") # "108mv65r")
for id in ${run_ids[@]}
do
    printf "Resuming run from ${id}\n"
    sbatch ./slurm/imagenet_x5_resume.sh $id
done
