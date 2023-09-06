from typing import Union, Optional
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
        
    def get_datasets_info(self):
        sample_data = self.dataset.trajectory_list[0]
        self.best_y = max([
            t.y.max() for t in self.dataset.trajectory_list
        ]).item()
        self.seq_len = sample_data.X.shape[0]
        self.x_dim = sample_data.X.shape[1]
        self.y_dim = 1
        
        id2info = {}
        for t in self.dataset.trajectory_list:
            id = t.metadata["dataset_id"]
            best_y = t.y.max().item()
            regret = (self.best_y - t.y).sum().item()
            if id not in id2info:
                id2info[id] = {"best_y": [], "regret": []}
            id2info[id]["best_y"].append(best_y)
            id2info[id]["regret"].append(regret)
        for id, d_ in id2info.items():
            id2info[id] = {
                "best_y": sum(d_["best_y"])/len(d_["best_y"]), 
                "regret": sum(d_["regret"])/len(d_["regret"])
            }
        self.id2info = id2info
    
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
        device = X.device
        X_np = X.cpu().detach().numpy()
        X_np = xgb.DMatrix(X_np)
        Y = self.bst_surrogate.predict(X_np)
        return torch.from_numpy(Y).reshape(-1, 1).to(device)
    
    def get_dataset(self):
        return self.dataset