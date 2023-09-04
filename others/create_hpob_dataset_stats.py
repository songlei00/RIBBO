import os
try:
    import ujson as json
except:
    import json


if os.getenv('HPOB_ROOT_DIR') is None:
    os.environ['HPOB_ROOT_DIR'] = os.path.expanduser('./data/downloaded_data/hpob')

path = os.path.join(os.environ['HPOB_ROOT_DIR'], 'hpob-data')

for mode in ['train', 'validation', 'test']:
    data_path = os.path.join(path, 'meta-{}-dataset.json'.format(mode))

    with open(data_path, 'rb') as f:
        meta_data = json.load(f)

    summary_stats = dict()
    for search_space_id in meta_data.keys():
        summary_stats[search_space_id] = []
        for dataset_id in meta_data[search_space_id].keys():
            summary_stats[search_space_id].append(dataset_id)

    with open('./hpob-summary-stats/{}-summary-stats.json'.format(mode), 'w') as f:
        json.dump(summary_stats, f)
