import os
from collections import defaultdict
import pickle
import logging

from datasets.datasets import TrajectoryDataset
from datasets.trajectory import Trajectory

logger = logging.getLogger(__name__)


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


def load_dataset(base_dir, filter_fn, split_fn):
    """
    The directory structures are as follows, 
    and the file name is denoted as designer, search space id, dataset id and seed. 
    ├── seed0
    │   ├── Random_4796_1093_0.json
    │   ├── Random_4796_10101_0.json
    │   ├── ...
    │   └── CMAES_7609_9983_0.json
    ├── seed1
    │   ├── ...
    │   └── ...
    ├── ...
    │   ├── ...
    │   └── ...
    └── seed100
        ├── ...
        └── ...

    Inputs:
        base_dir: path for the above directory
        split_fn: 
    """
    # obtain all file names of json data
    all_file_paths = []
    for dir_name in os.listdir(base_dir):
        if not dir_name.startswith('seed'):
            continue

        sub_path = os.path.join(base_dir, dir_name)
        for file_name in os.listdir(sub_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(sub_path, file_name)
                all_file_paths.append(file_path)

    # filter
    all_file_paths = filter(filter_fn, all_file_paths)

    # split
    sk2t = defaultdict(list) # splitkey2trajectory
    for file_path in all_file_paths:
        k = split_fn(file_path)
        v = Trajectory.load_from_json(file_path)
        sk2t[k].append(v)

    return sk2t


def load_hpob_dataset(search_space_id: str):
    def filter_fn(file_path):
        file_name = file_path.split(os.sep)[-1].rstrip('.json')
        curr_designer, curr_search_space_id, _, _ = file_name.split('_')
        if curr_designer in designers and curr_search_space_id == search_space_id:
            return True
        else:
            return False

    def split_fn(file_path):
        file_name = file_path.split(os.sep)[-1].rstrip('.json')
        search_space_id = file_name.split('_')[1]
        return search_space_id

    cache_dir = './cache/hpob/'
    cache_file_name = search_space_id + '.pkl'
    cache_path = os.path.join(cache_dir, cache_file_name)
    if not os.path.exists(cache_path):
        base_dir = './datasets/hpob_data'
        sp2t = load_dataset(base_dir, filter_fn, split_fn) # searchspace2trajectory
        dataset = TrajectoryDataset(sp2t[search_space_id])
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info('Save dataset cache to {}'.format(cache_path))
    else:
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
        logger.info('Load data set from {}'.format(cache_path))

    return dataset

