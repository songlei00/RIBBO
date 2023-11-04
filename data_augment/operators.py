from typing import Tuple, List
from functools import partial

import numpy as np 
import torch

from datasets.trajectory import Trajectory


def delete_by_idx(trajectory, delete_idx):
    selected_idx = [i for i in range(len(trajectory)) if i not in delete_idx]
    return Trajectory(dict(), trajectory.X[selected_idx], trajectory.y[selected_idx]), delete_idx


def random_delete(trajectory, num_delete) -> Tuple[Trajectory, List]:
    """
    Delete some (x, y) pairs from the trajectory randomly
    Inputs:
        trajectory:
        num_delete:
    """
    delete_idx = torch.randperm(len(trajectory))[: num_delete]
    return delete_by_idx(trajectory, delete_idx)


def keep_topk_delete(trajectory, num_delete, k):
    """
    Delete some (x, y) pairs from the trajectory and keep the best topk
    """
    y = trajectory.y
    topk_idx = torch.topk(y, k=k).indices
    to_delete_idx = [i for i in range(len(trajectory)) if i not in topk_idx]
    delete_idx = np.random.choice(to_delete_idx, num_delete, replace=False)
    return delete_by_idx(trajectory, delete_idx)


def duplicate_delete(trajectory, num_delete):
    """
    Delete `num_delete` pairs from the most similar X
    """
    X = trajectory.X 
    length, dim = X.shape
    X1 = X.unsqueeze(-2)
    X2 = X.unsqueeze(-3)
    diff = X1 - X2
    assert diff.shape == (length, length, dim)
    norm = torch.square(diff).sum(-1)
    difference = torch.triu(norm)
    MAX_VALUE = 9999999
    tril_idx = torch.tril_indices(length, length)
    for i, j in zip(tril_idx[0], tril_idx[1]):
        difference[i][j] = MAX_VALUE

    flatten_idx = torch.topk(difference.flatten(), num_delete, largest=False).indices
    similar_pair_idx = np.unravel_index(flatten_idx.numpy(), difference.shape)
    delete_idx = []
    for i, j in zip(similar_pair_idx[0], similar_pair_idx[1]):
        delete_idx.append(np.random.choice([i, j], 1).item())

    return delete_by_idx(trajectory, delete_idx)


def swap_contiguous(trajectory, trajectory_list, num_swap):
    pass


def swap_discrete(trajectory1, trajectory2, num_swap):
    pass


def add_auxiliary_points(trajectory, trajectory_list, num_add):
    pass


def optimization(trajectory):
    pass


def get_increasing_subtrajectory(trajectory, increasing=True):
    y = trajectory.y
    increasing_idx_y_pair = sorted(zip(range(len(y)), y), key=lambda x: x[1])
    sample_idx = list(range(0, len(y), 2))
    curr_idx_y_pair = [increasing_idx_y_pair[i] for i in sample_idx]
    curr_idx = list(zip(*(curr_idx_y_pair)))[0]

    if not increasing:
        curr_idx = sorted(curr_idx)

    curr_X = [trajectory.X[i] for i in curr_idx]
    curr_y = [trajectory.y[i] for i in curr_idx]
    metadata = {
        'designer': 'synthetic',
        'search_space_id': trajectory.metadata['search_space_id'],
        'dataset_id': trajectory.metadata['dataset_id'],
        'length': len(curr_X),
        'seed': trajectory.metadata['seed'],
    }
    curr_t = Trajectory(metadata, curr_X, curr_y)
    return curr_t


augment_transform = (
    (lambda t: True, partial(get_increasing_subtrajectory, increasing=False), 0.25),
)