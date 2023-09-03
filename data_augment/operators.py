import numpy as np 
import torch

from datasets.trajectory import Trajectory


def random_delete(trajectory, num_delete):
    """
    Delete some (x, y) pairs from the trajectory, and keep the topk pairs
    Inputs:
        trajectory:
        num_delete:
    """
    delete_idx = torch.randperm(len(trajectory))[: num_delete]
    seleted_idx = [i for i in range(len(trajectory)) if i not in delete_idx]
    metadata = dict()
    return Trajectory(metadata, trajectory.X[seleted_idx], trajectory.y[seleted_idx])


def keep_topk_delete(trajectory, num_delete, num_keep_topk):
    pass


def duplicate_delete(trajectory, num_delete):
    pass


def swap_contiguous(trajectory1, trajectory2, length):
    pass


def swap_discrete(trajectory1, trajectory2, length):
    pass
