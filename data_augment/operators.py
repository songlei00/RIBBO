import numpy as np 

from data_augment.trajectory import Trajectory


def random_delete(trajectory, num_delete, keep_topk=1):
    """
    Delete some (x, y) pairs from the trajectory, and keep the topk pairs
    Inputs:
        trajectory:
        num_delete:
        keep_topk:
    """
    partition_idx = np.argpartition(trajectory.y, -keep_topk)
    to_delete_idx, topk_idx = partition_idx[: -keep_topk], partition_idx[-keep_topk: ]
    delete_idx = np.random.choice(to_delete_idx, num_delete, replace=False)


def swap_contiguous(trajectory1, trajectory2, length):
    pass


def swap_discrete(trajectory1, trajectory2, length):
    pass