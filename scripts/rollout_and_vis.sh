#!/bin/bash

problem=synthetic
funcs=(
    Rastrigin
)

for id in ${funcs[@]}
do
    python scripts/rollout_and_vis.py \
        --config scripts/configs/rollout/${problem}.py \
        --id \"$id\" \
        --max_input_seq_len 150
done