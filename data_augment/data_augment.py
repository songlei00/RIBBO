from typing import Optional, List, Callable

import numpy as np

from datasets.datasets import TrajectoryDataset


def get_augmented_dataset(base_dataset, num_augmented, operators: List[Callable], p: Optional[List[int]] = None):
    augmented_list = []
    selected_idx = np.random.choice(len(base_dataset), size=(num_augmented, ))
    selected_operators = np.random.choice(operators, size=(num_augmented, ), p=p)
    for idx, operator in zip(selected_idx, selected_operators):
        trajectory = base_dataset.trajectory_list[idx]
        augmented_trajectory = operator(trajectory)
        augmented_list.append(augmented_trajectory)

    return TrajectoryDataset(augmented_list)
    