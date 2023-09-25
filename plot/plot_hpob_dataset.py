import pickle
import os
try:
    import ujson as json
except:
    import json

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


with open('others/hpob-summary-stats/train-summary-stats.json', 'r') as f:
    meta_data = json.load(f)

save_dir = 'plot/hpob'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# for search_space_id in meta_data:
for search_space_id in ['6767']:
    path = 'cache/hpob'
    trajectory_dataset = TrajectoryDataset(
        search_space_id,
        '',
        path,
    )

    f, axs = plot_dataset(
        trajectory_dataset,
        xy_fn,
        split_fn,
        group_fn,
        ncols=5,
        shaded_std=False,
    )
    save_name = '{}_y.png'.format(search_space_id)
    plt.savefig(os.path.join(save_dir, save_name))

    f, axs = plot_dataset(
        trajectory_dataset,
        x_besty_fn,
        split_fn,
        group_fn,
        ncols=5,
        shaded_std=False,
    )
    save_name = '{}.png'.format(search_space_id)
    plt.savefig(os.path.join(save_dir, save_name))