try:
    import ujson as json
except:
    import json
import logging

import torch

from datasets.load_datasets import load_hpob_dataset

logging.basicConfig(level=logging.INFO)


with open('./others/hpob-summary-stats/train-summary-stats.json', 'r') as f:
    summary_stats = json.load(f)

for search_space_id in summary_stats:
# for search_space_id in ['6767']:
    trajectory_dataset = load_hpob_dataset(search_space_id, update=True)