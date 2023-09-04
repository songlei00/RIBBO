try:
    import ujson as json
except:
    import json
import logging

from datasets.load_datasets import load_hpob_dataset

logging.basicConfig(level=logging.INFO)


with open('./others/hpob-summary-stats/train-summary-stats.json', 'r') as f:
    summary_stats = json.load(f)

for search_space_id in summary_stats:
    trajectory_dataset = load_hpob_dataset(search_space_id)