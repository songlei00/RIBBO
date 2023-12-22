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
        --ckpt_id \"$id\" \
        --eval_id \"$id\" \
        --max_input_seq_len 150
done