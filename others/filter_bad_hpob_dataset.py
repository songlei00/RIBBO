import os
try:
    import ujson as json
except:
    import json
import logging

import numpy as np

from datasets.load_datasets import load_hpob_dataset
from problems.hpob_problem import HPOBProblem

logging.basicConfig(level=logging.INFO)


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

with open('./others/hpob-summary-stats/train-summary-stats.json', 'r') as f:
    summary_stats = json.load(f)

bad_file_names = []
for search_space_id in summary_stats:
    for dataset_id in summary_stats[search_space_id]:
        problem = HPOBProblem(search_space_id, dataset_id, './data/downloaded_data/hpob')
        base_dir = './data/generated_data/hpob/'
        for designer in designers:
            for seed in range(50):
                file_name = '{}_{}_{}_{}.json'.format(designer, search_space_id, dataset_id, seed)
                file_path = os.path.join(base_dir, 'seed{}'.format(seed), file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # filter
                    X = np.array(data['X'])
                    if X.shape[-1] != problem.dim:
                        bad_file_names.append(file_name)

print(bad_file_names)
print(len(bad_file_names))