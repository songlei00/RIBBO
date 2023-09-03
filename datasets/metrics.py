import numpy as np 
import torch


def metric_best_y(trajectory):
    metadata, X, y = trajectory.metadata, trajectory.X, trajectory.y
    return torch.max(y)


def metric_regret(trajectory, best_y):
    metadata, X, y = trajectory.metadata, trajectory.X, trajectory.y
    regret = torch.norm(best_y - y, p=1)
    return regret


def metric_diversity_X(trajectory):
    pass


def metric_diversity_y(trajectory):
    pass