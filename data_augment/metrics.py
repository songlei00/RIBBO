import numpy as np 
import torch


def metric_best_y(trajectory):
    metadata, X, y = trajectory.metadata, trajectory.X, trajectory.y
    return np.max(y)


def metric_regret(trajectory, best_y=None):
    metadata, X, y = trajectory.metadata, trajectory.X, trajectory.y
    if best_y is None:
        best_y = metric_best_y(trajectory)

    regret = np.linalg.norm(best_y - np.asarray(y), ord=1)
    return regret


def metric_diversity_X(trajectory):
    pass


def metric_diversity_y(trajectory):
    pass