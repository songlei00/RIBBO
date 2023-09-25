import os
import pickle
from typing import List, Optional
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from datasets.trajectory import Trajectory
from datasets.metrics import metric_regret
from datasets.load_datasets import load_hpob_dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self, 
        search_space_id: str,
        data_dir: str,
        cache_dir: str, 
        input_seq_len: int=300, 
        normalize_method: str="random",  # choices are ["random", "dataset", "none"]
        update: bool = False,
    ) -> None:
        cache_file = search_space_id + '.pkl'
        cache_path = os.path.join(cache_dir, cache_file)

        if not os.path.exists(cache_path) or update:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            trajectory_list = self.create_cache(search_space_id, data_dir, cache_path)
        else:
            trajectory_list = self.load_cache(cache_path)
            assert isinstance(trajectory_list, list)
        self.trajectory_list = trajectory_list
        
        # get raw metrics
        self.id2info, self.global_info = self.get_dataset_info()
        
        # calculate regrets
        self.set_regrets()
        self.input_seq_len = input_seq_len
        self.normalize_method = normalize_method

    def create_cache(self, search_space_id, data_dir, cache_path):
        trajectory_list = load_hpob_dataset(data_dir, search_space_id)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(trajectory_list, f)
        print('Save trajectory to {}'.format(cache_path))

        return trajectory_list
        
    def load_cache(self, cache_path):
        with open(cache_path, 'rb') as f:
            trajectory_list = pickle.load(f)
        print('Load trajectory from {}'.format(cache_path))
        return trajectory_list

    def get_dataset_info(self):
        id2info = defaultdict(dict)
        global_info = dict()

        # group the trajectory by id, and calc the metrics
        id2group = defaultdict(list)        
        for t in self.trajectory_list:
            dataset_id = t.metadata["dataset_id"]
            id2group[dataset_id].append(t)
        for id in id2group:
            y_max = max(t.y.max() for t in id2group[id]).item()
            y_min = min(t.y.min() for t in id2group[id]).item()
            best_y_average = sum(t.y.max() for t in id2group[id]) / len(id2group[id])
            id2info[id].update({
                "y_max": y_max, 
                "y_min": y_min, 
                "best_y_average": best_y_average
            })
        
        # global info
        x_min = min(t.X.min() for t in self.trajectory_list).item()
        x_max = max(t.X.max() for t in self.trajectory_list).item()
        y_min = min([id2info[id]["y_min"] for id in id2info])
        y_max = max([id2info[id]["y_max"] for id in id2info])
        y_max_mean = sum([id2info[id]["y_max"] for id in id2info]) / len(id2info)
        y_min_mean = sum([id2info[id]["y_min"] for id in id2info]) / len(id2info)
        global_info.update({
            "x_min": x_min, 
            "x_max": x_max, 
            "y_min": y_min, 
            "y_max": y_max, 
            "y_max_mean": y_max_mean, 
            "y_min_mean": y_min_mean, 
            "train_datasets": set(id2info.keys())
        })
        return id2info, global_info

    def set_regrets(self):
        for i in range(len(self.trajectory_list)):
            id = self.trajectory_list[i].metadata["dataset_id"]
            y_max = self.id2info[id]["y_max"]
            self.trajectory_list[i].regrets = metric_regret(self.trajectory_list[i], y_max)
        
    def transform_x(self, fn):
        for i in range(len(self.trajectory_list)):
            self.trajectory_list[i].X = fn(self.trajectory_list[i].X)
            
    def __len__(self):
        return len(self.trajectory_list)
    
    def __getitem__(self, idx):
        trajectory = self.trajectory_list[idx]
        # best_y = self.id2info[trajectory.metadata['dataset_id']]['best_y']
        traj_len = trajectory.X.shape[0]
        start_idx = np.random.choice(traj_len+1-self.input_seq_len)
        
        timesteps = torch.arange(start_idx, start_idx+self.input_seq_len)
        
        y, regrets = self.normalize_y_and_regrets(
            trajectory.metadata["dataset_id"], 
            trajectory.y, 
            trajectory.regrets
        )
        return {
            "x": trajectory.X[start_idx:start_idx+self.input_seq_len], 
            "y": y[start_idx:start_idx+self.input_seq_len], 
            "regrets": regrets[start_idx:start_idx+self.input_seq_len], 
            "timesteps": timesteps, 
            "masks": torch.ones_like(timesteps).float()
        }
        
    def normalize_y_and_regrets(self, id, y, regrets):
        y = y.unsqueeze(-1)
        regrets = regrets.unsqueeze(-1)
        if self.normalize_method == "none":
            return y, regrets
        elif self.normalize_method == "random":
            dataset_y_min, dataset_y_max = self.id2info[id]["y_min"], self.id2info[id]["y_max"]
            span = (dataset_y_max - dataset_y_min + 1e-6) / 2.0
            l = np.random.uniform(low=dataset_y_min-span/2, high=dataset_y_min+span/2)
            h = np.random.uniform(low=dataset_y_max-span/2, high=dataset_y_max+span/2)
            scale = max(h-l, 0.3)
            return (y-l) / scale, regrets / scale
        elif self.normalize_method == "dataset": 
            dataset_y_min, dataset_y_max = self.id2info[id]["y_min"], self.id2info[id]["y_max"]
            scale = max(dataset_y_max - dataset_y_min + 1e-6, 0.3)
            return (y - dataset_y_min) / scale, regrets / scale
            
