from typing import List

import torch
from torch.utils.data import Dataset, IterableDataset

from datasets.trajectory import Trajectory
from datasets.metrics import metric_regret


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectory_list: List[Trajectory],
    ):
        self.trajectory_list = trajectory_list
        self.best_y = torch.max(
            torch.tensor([torch.max(trajectory.y) for trajectory in self.trajectory_list])
        )

    def __getitem__(self, idx):
        trajectory = self.trajectory_list[idx]
        timesteps = torch.arange(0, trajectory.X.shape[0]).long()
        return {
            "x": trajectory.X, 
            "y": trajectory.y.unsqueeze(-1), 
            "regrets": metric_regret(trajectory, self.best_y).unsqueeze(-1), 
            "timesteps": timesteps, 
            "masks": torch.ones_like(timesteps).float()
        }

    def __len__(self):
        return len(self.trajectory_list)

    def transform_x(self, fn):
        for i in range(len(self.trajectory_list)):
            self.trajectory_list[i].X = fn(self.trajectory_list[i].X)