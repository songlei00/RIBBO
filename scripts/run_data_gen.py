import os
try:
    import ujson as json
except:
    import json
import multiprocessing as mp

import numpy as np


designers = [
    'Random',
    # 'GridSearch',
    'ShuffledGridSearch',
    'RegularizedEvolution',
    'HillClimbing',
    'EagleStrategy',
    # 'Vizier',
    'HeBO',
    'CMAES',
]


def get_cmd(designer, search_space_id, dataset_id, out_name, length, seed):
    prefix = 'taskset -c {}-{}'.format(args.cpu_start, args.cpu_end)
    cmd = 'python data_gen/data_gen_main.py \
        --designer={} \
        --search_space_id={} \
        --dataset_id={} \
        --root_dir=./data/downloaded_data/hpob \
        --out_name={} \
        --length={} \
        --seed={}'.format(designer, search_space_id, dataset_id, out_name, length, seed)
    cmd = ' '.join([prefix, cmd])
    return cmd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--cpu_start', type=int, required=True)
    parser.add_argument('--cpu_end', type=int, required=True)
    args = parser.parse_args()

    mode = ['train', 'test', 'validation']
    with open('others/hpob-summary-stats/{}-summary-stats.json'.format(mode[0]), 'rb') as f:
        summary_stats = json.load(f)

    # key = list(summary_stats.keys())[0]
    # value = summary_stats[key][: 2]
    # summary_stats = dict()
    # summary_stats[key] = value
    # print(summary_stats)

    if args.smoke_test:
        search_space_id = list(summary_stats.keys())[0]
        dataset_id = summary_stats[search_space_id][0]
        print(search_space_id, dataset_id)
        for designer in designers:
            print(designer)
            os.system(get_cmd(
                designer, 
                search_space_id, 
                dataset_id, 
                'data/generated_data/smoke_test_{}.json'.format(designer), 
                100,
                seed=args.seed,
            ))
    else:
        seed = args.seed
        failed_cmds = []

        dir_path = 'data/generated_data/hpob/seed{}'.format(seed)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        all_out_names = []
        for designer in designers:
            # print(designer, seed)
            for search_space_id in summary_stats:
                for dataset_id in summary_stats[search_space_id]:
                    print(designer, search_space_id, dataset_id, seed)
                    out_name = dir_path + '/{}_{}_{}_{}.json'.format(designer, search_space_id, dataset_id, seed)
                    cmd = get_cmd(
                        designer,
                        search_space_id, 
                        dataset_id,
                        out_name,
                        300,
                        seed=seed,
                    )
                    ret = os.system(cmd)
                    if ret != 0:
                        failed_cmds.append(cmd)

                        print('----------------------------')
                        print('Error: {}'.format(cmd))
                        print('----------------------------')

                    all_out_names.append(out_name)

        print('----------------------------')
        print(failed_cmds)
        print('----------------------------')