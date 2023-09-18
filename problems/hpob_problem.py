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
from datasets.load_datasets import load_hpob_dataset


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
    ):
        assert search_space_id.isnumeric()
        self.root_dir = root_dir
        self.search_space_id = search_space_id

        self.bst_surrogate = xgb.Booster()
        self.name = 'HPOB_{}'.format(search_space_id)
        self.dataset = load_hpob_dataset(
            search_space_id=search_space_id
        )
        self.get_datasets_info()
        
        # transform the dataset x
        self.dataset.transform_x(partial(self.transform_x, reverse=True))
        
    def get_datasets_info(self):
        sample_data = self.dataset.trajectory_list[0]
        self.seq_len = sample_data.X.shape[0]
        self.x_dim = sample_data.X.shape[1]
        self.y_dim = 1
        self.best_y = self.dataset.best_y
        self.best_original_y = self.dataset.best_original_y
        self.id2info = self.dataset.id2info
        
        self.x_low = 0.0
        self.x_high = 1.0
        
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
        Y = self.bst_surrogate.predict(X_np)
        normalized_y = self.y_normalize(Y)
        return torch.from_numpy(Y).reshape(-1, 1).to(device), torch.from_numpy(normalized_y).reshape(-1, 1).to(device)
    
    def get_dataset(self):
        return self.dataset

    def y_normalize(self, y):
        if self.dataset_id in self.id2info.keys(): # train dataset
            info = self.id2info[self.dataset_id]
            min_y, max_y = info['min_y'], info['max_y']
        else: # test dataset
            cheat_table = {
                # ''
            }
            if self.dataset_id in cheat_table.keys(): 
                # normalize by a predefined value
                min_y = cheat_table[self.dataset_id]['min_y']
                max_y = cheat_table[self.dataset_id]['max_y']
            else:
                # normalize by min_y and max_y in the training dataset
                min_y = min([self.id2info[dataset_id]['min_y'] for dataset_id in self.id2info])
                max_y = max([self.id2info[dataset_id]['max_y'] for dataset_id in self.id2info])

        new_y = (y - min_y) / (max_y - min_y + 1e-6)
        return new_y
