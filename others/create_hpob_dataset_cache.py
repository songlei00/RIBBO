try:
    import ujson as json
except:
    import json
import logging
import argparse

from datasets.datasets import TrajectoryDataset

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'validation'])
args = parser.parse_args()

mode = args.mode
with open('./others/hpob-summary-stats/{}-summary-stats.json'.format(mode), 'r') as f:
    summary_stats = json.load(f)

if mode == 'train':
    data_dir = 'data/generated_data/hpob/'
    cache_dir = 'cache/hpob'
else:
    data_dir = 'data/generated_data/hpob_{}/'.format(mode)
    cache_dir = 'cache/hpob_{}'.format(mode)

# for search_space_id in summary_stats:
for search_space_id in ['6767']:
    dataset = TrajectoryDataset(
        search_space_id=search_space_id,
        data_dir=data_dir,
        cache_dir=cache_dir, 
        input_seq_len=300, 
        normalize_method='random',
    )