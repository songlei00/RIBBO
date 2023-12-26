#!/bin/bash

problem=synthetic
funcs=(
    GriewankRosenbrock
    Lunacek
    Rastrigin
    RosenbrockRotated
    SharpRidge
)

for id in ${funcs[@]}
do
    python scripts/rollout_and_vis.py \
        --config scripts/configs/rollout/${problem}.py \
        --ckpt_id "all_id" \
        --eval_id \"$id\" \
        --max_input_seq_len 150
done

for i_id in ${funcs[@]}
do
    for j_id in ${funcs[@]}
    do
        python scripts/rollout_and_vis.py \
            --config scripts/configs/rollout/${problem}.py \
            --ckpt_id \"$i_id\" \
            --eval_id \"$j_id\" \
            --max_input_seq_len 150
    done
done