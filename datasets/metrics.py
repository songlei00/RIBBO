import numpy as np 
import torch


def metric_best_y(trajectory):
    metadata, X, y = trajectory.metadata, trajectory.X, trajectory.y
    return torch.max(y)


def metric_regret(trajectory, best_y):
    metadata, X, y = trajectory.metadata, trajectory.X, trajectory.y
    regret = torch.flip(torch.cumsum(torch.flip(best_y - y, dims=[0]), dim=0), dims=[0])
    return regret


def metric_diversity_X(trajectory):
    pass


def metric_diversity_y(trajectory):
    pass