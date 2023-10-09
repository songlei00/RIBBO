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