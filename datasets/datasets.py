from typing import List

import torch
import numpy as np
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
        self.set_regrets()
        self.input_seq_len = None

    def __getitem__(self, idx):
        assert self.input_seq_len is not None, "input seq len must be set before iterating over the dataset"
        trajectory = self.trajectory_list[idx]
        traj_len = trajectory.X.shape[0]
        start_idx = np.random.choice(traj_len-self.input_seq_len)
        timesteps = torch.arange(start_idx, start_idx + self.input_seq_len).long()
        return {
            "x": trajectory.X[start_idx:start_idx+self.input_seq_len], 
            "y": trajectory.y[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            "regrets": metric_regret(trajectory, self.best_y)[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            # "regrets": trajectory.regrets[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            "timesteps": timesteps, 
            "masks": torch.ones_like(timesteps).float()
        }

    def __len__(self):
        return len(self.trajectory_list)
    
    def set_regrets(self):
        for i in range(len(self.trajectory_list)):
            self.trajectory_list[i].regrets = metric_regret(self.trajectory_list[i], self.best_y)

    def transform_x(self, fn):
        for i in range(len(self.trajectory_list)):
            self.trajectory_list[i].X = fn(self.trajectory_list[i].X)

    def set_input_seq_len(self, input_seq_len):
        self.input_seq_len = input_seq_len