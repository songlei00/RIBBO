#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

designer=dt
problem=synthetic
funcs=(
    GriewankRosenbrock
    Lunacek
    Rastrigin
    RosenbrockRotated
    SharpRidge
)
min_seed=0
max_seed=0

function run() {
    python scripts/run_$1.py \
        --config scripts/configs/$1/$2.py \
        --name $1-default \
        --embed_dim 256 \
        --num_heads 8 \
        --num_layers 12 \
        --pos_encoding embed \
        --input_seq_len 50 \
        --max_input_seq_len 150 \
        --wandb.project IBO-benchmark \
        --num_epoch 5000 \
        --batch_size 64 \
        --eval_interval 99999 \
        --id "[$3]" \
        --seed $4
}

for ((seed=${min_seed}; seed<=${max_seed}; seed++))
do
    # exclude one func
    for ((i=0; i<${#funcs[@]}; i++))
    do
        id=(${funcs[@]:0:$i} ${funcs[@]:$(($i + 1))})
        search_space_id=\"${id[0]}\"
        for j in ${id[@]:1}
        do
            search_space_id=$search_space_id,\"$j\"
        done
        
        echo 'search space id: '$search_space_id

        run $designer $problem $search_space_id $seed
    done

    # run on all funcs
    search_space_id=\"${funcs[0]}\"
    for i in ${funcs[@]:1}
    do
        search_space_id=$search_space_id,\"$i\"
    done
    echo 'search space id: '$search_space_id
    run $designer $problem $search_space_id $seed
done