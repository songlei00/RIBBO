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

available_gpu=(0)
num_gpu=${#available_gpu[@]}
max_proc=1
echo 'GPU idx: '${available_gpu[@]}
echo 'Number of GPU: '$num_gpu
echo 'Max number of processes: '$max_proc

fifo_name="/tmp/$$.fifo"
mkfifo $fifo_name
exec 7<>${fifo_name}
# rm $fifo_name

for ((i=0; i<$max_proc; i++))
do
    echo $i
done >&7

curr_idx=0
for ((seed=${min_seed}; seed<=${max_seed}; seed++))
do
    for id in ${funcs[@]}
    do
        read -u7 proc_id
        curr_gpu=${available_gpu[${curr_idx}]}
        curr_idx=$((( $curr_idx + 1 ) % $num_gpu))
        echo 'current proc id: '$proc_id
        echo 'current func: '$id', current gpu idx: '$curr_gpu
        {
            python scripts/run_${designer}.py \
                --config scripts/configs/${designer}/${problem}.py \
                --name ${designer}-default \
                --embed_dim 128 \
                --num_layers 12 \
                --pos_encoding embed \
                --input_seq_len 50 \
                --max_input_seq_len 150 \
                --wandb.project IBO-benchmark \
                --num_epoch 5000 \
                --batch_size 128 \
                --eval_interval 9999 \
                --id \"$id\" \
                --seed $seed \
                --device cuda:${curr_gpu}

            sleep 1
            echo >&7 $proc_id
        } &
    done
done

wait
exec 7>&-