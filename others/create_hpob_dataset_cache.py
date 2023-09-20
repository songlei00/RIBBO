try:
    import ujson as json
except:
    import json
import logging

import torch

# from datasets.load_datasets import load_hpob_dataset
from datasets.datasets import TrajectoryDataset

logging.basicConfig(level=logging.INFO)


# with open('./others/hpob-summary-stats/train-summary-stats.json', 'r') as f:
#     summary_stats = json.load(f)

# for search_space_id in summary_stats:
for search_space_id in ['6767']:
    dataset = TrajectoryDataset(
        search_space_id='6767',
        data_dir='data/generated_data/hpob/',
        cache_dir='cache/hpob', 
        input_seq_len=300, 
        normalize_method='random',
    )