import pickle
import os
try:
    import ujson as json
except:
    import json
import argparse

import matplotlib.pyplot as plt 

from plot.plot_dataset import plot_dataset
from datasets.load_datasets import load_hpob_dataset
from datasets.datasets import TrajectoryDataset


def x_besty_fn(trajectory):
    """
    function that converts trajectory into tuple of x and y
    """
    y = trajectory.y.tolist()
    best_y = [y[0]]
    for i in y[1: ]:
        best_y.append(max(best_y[-1], i))
    return list(range(len(y))), best_y

def xy_fn(trajectory):
    """
    function that converts trajectory into tuple of x and y
    """
    y = trajectory.y.tolist()
    return list(range(len(y))), y

def split_fn(trajectory):
    """
    function that converts trajectory into keys to split trajectories into groups
    """
    metadata = trajectory.metadata
    key = '{}_{}'.format(metadata['search_space_id'], metadata['dataset_id'])
    return key

def group_fn(trajectory):
    """
    function that converts trajectories in the same group into keys to average the results
    """
    return trajectory.metadata['designer']


parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, required=True)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'validation'])
args = parser.parse_args()

mode = args.mode
if args.problem == 'hpob':
    with open('others/hpob-summary-stats/{}-summary-stats.json'.format(mode), 'r') as f:
        meta_data = json.load(f)
elif args.problem == 'synthetic':
    from problems.synthetic import bbob_func_names
    meta_data = {name: [str(i) for i in range(50)] for name in bbob_func_names}
else:
    raise NotImplementedError

if mode == 'train':
    save_dir = 'plot/{}'.format(args.problem)
    path = 'cache/{}'.format(args.problem)
else:
    save_dir = 'plot/{}_{}'.format(args.problem, mode)
    path = 'cache/{}_{}'.format(args.problem, mode)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for search_space_id in meta_data:
# for search_space_id in ['6767']:
    trajectory_dataset = TrajectoryDataset(
        search_space_id,
        '',
        path,
    )

    print('Load search space id: {}'.format(search_space_id))
    print(trajectory_dataset.id2info)

    for mode in ['y', 'best_y']:
        if mode == 'y':
            fn = xy_fn
            save_name = '{}_y.png'.format(search_space_id)
        else:
            fn = x_besty_fn
            save_name = '{}.png'.format(search_space_id)

        f, axs = plot_dataset(
            trajectory_dataset,
            fn,
            split_fn,
            group_fn,
            ncols=5,
            shaded_std=False,
        )
        plt.savefig(os.path.join(save_dir, save_name))