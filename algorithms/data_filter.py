def filter_designer(dataset):
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
    ret = list(filter(filter_fn, dataset.trajectory_list))
    logger.info('Filter designers')
    return ret


def filter_dataset(dataset):
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
    ret = list(filter(filter_fn, dataset.trajectory_list))
    logger.info('Filter dataset')
    return ret