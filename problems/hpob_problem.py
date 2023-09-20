from typing import Union, Optional
from functools import partial
from collections import defaultdict
import os
try:
    import ujson as json
except:
    import json

import numpy as np
import xgboost as xgb
import torch
from torch import Tensor

from problems.base import ProblemBase
from datasets.datasets import TrajectoryDataset


def load_summary(root_dir):
    file_name = 'summary-stats.json'
    path = os.path.join(root_dir, 'saved-surrogates', file_name)
    with open(path, 'rb') as f:
        summary_stats = json.load(f)
    return summary_stats


class HPOBProblem(ProblemBase):
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        root_dir: str, 
        noise_std: float = 0,
    ):
        assert search_space_id.isnumeric()
        assert dataset_id.isnumeric()
        if root_dir.startswith('~'):
            root_dir = os.path.expanduser(root_dir)

        # load model
        self.summary_stats = load_summary(root_dir)
        self.surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
        self.bst_surrogate = xgb.Booster()
        self.bst_surrogate.load_model(os.path.join(root_dir, 'saved-surrogates', self.surrogate_name+'.json'))

        self.dim = self.bst_surrogate.num_features()
        self.lb = torch.zeros(self.dim)
        self.ub = torch.ones(self.dim)
        self.name = 'HPOB_{}'.format(search_space_id)

    def forward(self, X: Tensor) -> Tensor:
        """
        Inputs:
            X: Tensor [bs, dim]
        Outputs:
            Tensor [bs, 1]
        """
        assert X.ndim == 2
        X_np = X.cpu().detach().numpy()
        Y = []
        dim = X_np.shape[-1]
        for x in X_np:
            x_q = xgb.DMatrix(x.reshape(-1, dim))
            y = self.bst_surrogate.predict(x_q)
            Y.append(y)
        return torch.from_numpy(np.array(Y)).reshape(-1, 1).to(X)

    def info(self):
        return self.summary_stats[self.surrogate_name]


class HPOBMetaProblem():
    def __init__(
        self, 
        search_space_id: str, 
        root_dir: str, 
        cache_dir: str, 
        input_seq_len: int=300, 
        normalize_method: str="random"
    ):
        assert search_space_id.isnumeric()
        self.search_space_id = search_space_id
        self.root_dir = root_dir
        self.input_seq_len = input_seq_len

        self.bst_surrogate = xgb.Booster()
        self.name = 'HPOB_{}'.format(search_space_id)
        self.dataset = TrajectoryDataset(
            cache_dir=cache_dir, 
            input_seq_len=input_seq_len, 
            normalize_method=normalize_method
        )

        # transform the dataset x
        self.dataset.transform_x(partial(self.transform_x, reverse=True))
        
    def transform_x(self, x, reverse: bool=False):
        if reverse:
            return x * 2 - 1.0
        else:
            return x / 2 + 0.5
    
    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.surrogate_name = 'surrogate-'+self.search_space_id+'-'+dataset_id
        self.bst_surrogate.load_model(os.path.join(
            self.root_dir, 
            'saved-surrogates', 
            self.surrogate_name+'.json'
        ))
        
    def forward(self, X: torch.Tensor):
        assert X.ndim == 2
        assert (X >= -1 - 1e-6).all() and (X <= 1 + 1e-6).all()
        device = X.device
        X_np = X.cpu().detach().numpy()
        X_np = xgb.DMatrix(self.transform_x(X_np))
        y = self.bst_surrogate.predict(X_np)
        normalized_y = self.dataset_normalize(y)
        return torch.from_numpy(y).reshape(-1, 1).to(device), torch.from_numpy(normalized_y).reshape(-1, 1).to(device)
    
    def get_dataset(self):
        return self.dataset

    def dataset_normalize(self, y):
        if self.dataset_id in self.dataset.global_info["train_datasets"]:
            info = self.id2info[self.dataset_id]
            y_max, y_min = info["y_max"], info["y_min"]
        else:
            cheat_table = {
                
            }
            if self.dataset_id in cheat_table:
                y_max = cheat_table[self.dataset_id]["y_max"]
                y_min = cheat_table[self.dataset_id]["y_min"]
            else:
                y_max = self.dataset.global_info["y_max_mean"]
                y_min = self.dataset.global_info["y_min_mean"]
        normalized_y = (y-y_min) / (y_max-y_min+1e-6)
        return normalized_y
        
