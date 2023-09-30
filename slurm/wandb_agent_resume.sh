#!/bin/bash
# run_ids=("hkak5806" "p5n21r8q" "bbkeuxvr" "1hfmurvv")
run_ids=("gnhxedv3" "w9y6rjgl")
for id in ${run_ids[@]}
do
    printf "Resuming run from ${id}\n"
    sbatch ./slurm/imagenet_x5_resume.sh $id
done
