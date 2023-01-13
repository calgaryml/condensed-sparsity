#!/bin/bash
# run_ids=("hkak5806" "p5n21r8q" "bbkeuxvr" "1hfmurvv")
run_ids=("8nquqxxs" "5ygx59tl")
for id in ${run_ids[@]}
do
    printf "Resuming run from ${id}\n"
    sbatch ./slurm/run_train_rigl.sh $id
done
