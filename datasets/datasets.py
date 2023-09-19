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
        self.id2info = dict()

        # calculate min_y and max_y for each dataset
        id2minmax = self.get_dataset_minmax()
        for k, v in id2minmax.items():
            if self.id2info.get(k, None) is None:
                self.id2info[k] = dict()
            self.id2info[k].update(v)

        self.best_original_y = torch.max(
            torch.tensor([torch.max(trajectory.y) for trajectory in self.trajectory_list])
        ) 

        # normalize y for each dataset
        for t in self.trajectory_list:
            info = self.id2info[t.metadata['dataset_id']]
            max_y, min_y = info['max_y'], info['min_y']
            t.y = (t.y - min_y) / (max_y - min_y + 1e-6)

        # best_y should be calculated by the normalized y
        id2best_normalized_y = self.get_dataset_best_y()
        for k, v in id2best_normalized_y.items():
            if self.id2info.get(k, None) is None:
                self.id2info[k] = dict()
            self.id2info[k].update(v)

        self.best_y = torch.max(
            torch.tensor([torch.max(trajectory.y) for trajectory in self.trajectory_list])
        ) 

        # calculate average best_y and regret using the normalized y
        id2average = self.get_dataset_average()
        for k, v in id2average.items():
            if self.id2info.get(k, None) is None:
                self.id2info[k] = dict()
            self.id2info[k].update(v)

        self.set_regrets()
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
            dataset_id2info[dataset_id]['best_normalized_y'] = best_y

        return dataset_id2info

    def get_dataset_minmax(self):
        dataset_id2info = defaultdict(dict)

        # group the trajectory by dataset_id
        dataset_id2trajectory = defaultdict(list)
        for t in self.trajectory_list:
            dataset_id = t.metadata['dataset_id']
            dataset_id2trajectory[dataset_id].append(t)

        # calculate min_y and max_y for each dataset_id
        for dataset_id in dataset_id2trajectory:
            max_y = torch.max(
                torch.tensor([t.y.max() for t in dataset_id2trajectory[dataset_id]])
            ).item()
            min_y = torch.min(
                torch.tensor([t.y.min() for t in dataset_id2trajectory[dataset_id]])
            ).item()
            dataset_id2info[dataset_id]['max_y'] = max_y 
            dataset_id2info[dataset_id]['min_y'] = min_y

        return dataset_id2info

    def get_dataset_average(self):
        dataset_id2info = defaultdict(dict)

        # group the trajectory by dataset_id
        dataset_id2trajectory = defaultdict(list)
        for t in self.trajectory_list:
            dataset_id = t.metadata['dataset_id']
            dataset_id2trajectory[dataset_id].append(t)

        # calculate best_y and regret
        for dataset_id in dataset_id2trajectory:
            dataset_id2info[dataset_id]['average_best_y'] = torch.mean(
                torch.tensor([t.y.max().item() for t in dataset_id2trajectory[dataset_id]])
            ).item()
            dataset_id2info[dataset_id]['average_regret'] = torch.mean(
                torch.tensor([(self.best_y - t.y).sum().item() for t in dataset_id2trajectory[dataset_id]])
            ).item()

        return dataset_id2info

    def __getitem__(self, idx):
        assert self.input_seq_len is not None, "input seq len must be set before iterating over the dataset"
        trajectory = self.trajectory_list[idx]
        traj_len = trajectory.X.shape[0]
        start_idx = np.random.choice(traj_len+1-self.input_seq_len)
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