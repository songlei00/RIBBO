from collections import defaultdict

import torch


def filter_designer(trajectory_list):
    designers = [
        # 'Random',
        # 'GridSearch',
        # 'ShuffledGridSearch',
        # 'RegularizedEvolution',
        # 'HillClimbing',
        'EagleStrategy',
        # 'Vizier',
        # 'HeBO',
        # 'CMAES',
    ]
    def filter_fn(trajectory):
        metadata = trajectory.metadata
        return metadata['designer'] in designers
    ret = list(filter(filter_fn, trajectory_list))
    print('Filter designers')
    return ret


def filter_dataset(trajectory_list):
    bad_dataset = [
        '145833',
        '145855',
        '14971',
        '272',
        '3903',
        '3918',
        '7295',
        '9971',
        '9978',

        # good
        # '145839',
        # '9980',
        # '6566',
        # '146085',
        # '34536',
        # '49',
        # '145953',
        # '145872',
        # '146066',
    ]
    def filter_fn(trajectory):
        metadata = trajectory.metadata
        return metadata['dataset_id'] not in bad_dataset
    ret = list(filter(filter_fn, trajectory_list))
    print('Filter dataset')
    return ret


def map_smally(trajectory_list):
    bad_dataset = [
        '145833',
        '145855',
        '14971',
        '272',
        '3903',
        '3918',
        '7295',
        '9971',
        '9978',
    ]
    def map_fn(trajectory):
        metadata = trajectory.metadata
        if metadata['dataset_id'] in bad_dataset:
            trajectory.y = torch.rand_like(trajectory.y) * 0.1
        return trajectory
    ret = list(map(map_fn, trajectory_list))
    print('Map dataset')
    return ret


def rule_based_filter_dataset(trajectory_list):
    id2info = defaultdict(dict)
    id2group = defaultdict(list)
    for t in trajectory_list:
        dataset_id = t.metadata['dataset_id']
        id2group[dataset_id].append(t)

    for id in id2group:
        y_max = max(t.y.max() for t in id2group[id]).item()
        y_min = min(t.y.min() for t in id2group[id]).item()
        id2info[id].update({
            "y_max": y_max, 
            "y_min": y_min, 
        })

    metrics = {id: info['y_max']-info['y_min'] for id, info in id2info.items()}
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1])
    n = int(0.1 * len(metrics))
    bad_dataset = list(zip(*sorted_metrics[: n]))[0]
    print(bad_dataset)

    def filter_fn(trajectory):
        metadata = trajectory.metadata
        return metadata['dataset_id'] not in bad_dataset
    ret = list(filter(filter_fn, trajectory_list))
    print('Filter {} dataset, Total {}'.format(n, len(metrics)))
    return ret