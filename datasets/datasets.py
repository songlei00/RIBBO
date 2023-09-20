from typing import List
from collections import defaultdict

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

        # global best_y
        self.best_y = torch.max(
            torch.tensor([t.y.max() for t in self.trajectory_list])
        ) 

        # best_y for each dataset
        self.id2info = dict()
        id2best_y = self.get_dataset_best_y()
        for k, v in id2best_y.items():
            if self.id2info.get(k, None) is None:
                self.id2info[k] = dict()
            self.id2info[k].update(v)

        # self.set_regrets()
        self.input_seq_len = None

    def get_dataset_best_y(self):
        dataset_id2info = defaultdict(dict)

        # group the trajectory by dataset_id
        dataset_id2trajectory = defaultdict(list)
        for t in self.trajectory_list:
            dataset_id = t.metadata['dataset_id']
            dataset_id2trajectory[dataset_id].append(t)

        for dataset_id in dataset_id2trajectory:
            best_y = torch.max(
                torch.tensor([t.y.max() for t in dataset_id2trajectory[dataset_id]])
            )
            dataset_id2info[dataset_id]['best_y'] = best_y

        return dataset_id2info

    def __getitem__(self, idx):
        assert self.input_seq_len is not None, "input seq len must be set before iterating over the dataset"
        trajectory = self.trajectory_list[idx]
        best_y = self.id2info[trajectory.metadata['dataset_id']]['best_y']
        traj_len = trajectory.X.shape[0]
        start_idx = np.random.choice(traj_len+1-self.input_seq_len)
        timesteps = torch.arange(start_idx, start_idx + self.input_seq_len).long()
        return {
            "x": trajectory.X[start_idx:start_idx+self.input_seq_len], 
            "y": trajectory.y[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            "regrets": metric_regret(trajectory, best_y)[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            # "regrets": trajectory.regrets[start_idx:start_idx+self.input_seq_len].unsqueeze(-1), 
            "timesteps": timesteps, 
            "masks": torch.ones_like(timesteps).float()
        }

    def __len__(self):
        return len(self.trajectory_list)
    
    # def set_regrets(self):
    #     for i in range(len(self.trajectory_list)):
    #         self.trajectory_list[i].regrets = metric_regret(self.trajectory_list[i], self.best_y)

    def transform_x(self, fn):
        for i in range(len(self.trajectory_list)):
            self.trajectory_list[i].X = fn(self.trajectory_list[i].X)

    def set_input_seq_len(self, input_seq_len):
        self.input_seq_len = input_seq_len