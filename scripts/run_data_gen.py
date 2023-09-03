import os
try:
    import ujson as json
except:
    import json
import multiprocessing as mp

import numpy as np

os.environ['XLA_FLAGS'] = ('--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


SMOKE_TEST = False
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
    prefix = 'PYTHONPATH=.:$PYTHONPATH'
    cmd = 'python data_gen/data_gen_main.py \
        --designer={} \
        --search_space_id={} \
        --dataset_id={} \
        --root_dir=~/dataset/hpob \
        --out_name={} \
        --length={} \
        --seed={}'.format(designer, search_space_id, dataset_id, out_name, length, seed)
    cmd = ' '.join([prefix, cmd])
    return cmd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    mode = ['train', 'test', 'validation']
    with open('others/hpob-summary-stats/{}-summary-stats.json'.format(mode[0]), 'rb') as f:
        summary_stats = json.load(f)

    # key = list(summary_stats.keys())[0]
    # value = summary_stats[key][: 2]
    # summary_stats = dict()
    # summary_stats[key] = value
    # print(summary_stats)

    if SMOKE_TEST:
        search_space_id = list(summary_stats.keys())[0]
        dataset_id = summary_stats[search_space_id][0]
        print(search_space_id, dataset_id)
        for designer in designers:
            print(designer)
            os.system(get_cmd(
                designer, 
                search_space_id, 
                dataset_id, 
                'datasets/hpob_data/smoke_test_{}.json'.format(designer), 
                100,
                seed=0,
            ))
    else:
        # num_studies = 1e7
        num_studies = 0

        cnt = 0
        seed = args.seed
        failed_cmds = []
        while True:
            dir_path = 'datasets/hpob_data/seed{}'.format(seed)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            all_out_names = []
            for designer in designers:
                # print(designer, seed)
                for search_space_id in summary_stats:
                    for dataset_id in summary_stats[search_space_id]:
                        print(designer, search_space_id, dataset_id, seed)
                        out_name = 'datasets/hpob_data/seed{}/{}_{}_{}_{}.json'.format(seed, designer, search_space_id, dataset_id, seed)
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
                        cnt += 1

            # merge json file
            # merged_data = {}
            # for out_name in all_out_names:
            #     with open(out_name, 'rb') as f:
            #         data = json.load(f)
            #     key = os.path.basename(out_name).rstrip('.json')
            #     merged_data[key] = data
            #     os.system('rm {}'.format(out_name))
            
            # with open('datasets/hpob_data/seed_{}.json'.format(seed), 'w') as f:
            #     json.dump(merged_data, f)

            seed += 1
            if cnt >= num_studies:
                break

        print('----------------------------')
        print(failed_cmds)
        print('----------------------------')