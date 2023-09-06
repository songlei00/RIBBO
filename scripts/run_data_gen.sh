#!/bin/bash

max_proc=10
min_seed=0
max_seed=19
fifo_name="/tmp/$$.fifo"
mkfifo $fifo_name
exec 7<>${fifo_name}
# rm $fifo_name

for ((i=1; i<=$max_proc; i++))
do
    echo
done >&7

for ((seed=${min_seed}; seed<=${max_seed}; seed++))
do
    read -u7
    {
        python scripts/run_data_gen.py --seed=$seed
        sleep 1
        echo >&7
    } &
done

wait
exec 7>&-