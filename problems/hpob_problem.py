from typing import Union, Optional, List
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
# from datasets.datasets import TrajectoryDataset
from datasets.datasets import TrajectoryIterableDataset
from torch.utils.data import Dataset


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
        data_dir: str,
        cache_dir: str, 
        input_seq_len: int=300, 
        normalize_method: str="random",
        scale_clip_range: Optional[List[float]]=None, 
        augment: bool=False,
        prioritize: bool=False, 
        prioritize_alpha: float=1.0, 
    ):
        assert search_space_id.isnumeric()
        self.search_space_id = search_space_id
        self.root_dir = root_dir
        self.input_seq_len = input_seq_len
        self.scale_clip_range = scale_clip_range

        self.bst_surrogate = xgb.Booster()
        self.name = 'HPOB_{}'.format(search_space_id)
        self.dataset = TrajectoryIterableDataset(
            search_space_id=search_space_id,
            data_dir=data_dir,
            cache_dir=cache_dir, 
            input_seq_len=input_seq_len, 
            normalize_method=normalize_method, 
            scale_clip_range=scale_clip_range, 
            augment=augment,
            prioritize=prioritize, 
            prioritize_alpha=prioritize_alpha
        )

        # transform the dataset x
        self.dataset.transform_x(partial(self.transform_x, reverse=True))

        self.get_problem_info()

        # self.cheat_table = {
        #     '6767': {
        #         '146065': {'y_max': 0.767839789390564, 'y_min': 0.5621159076690674, 'best_y_average': 0.6984}, 
        #         '9967': {'y_max': 1.056007981300354, 'y_min': 0.3921036124229431, 'best_y_average': 0.8885}, 
        #         '9914': {'y_max': 0.9863658547401428, 'y_min': 0.9326095581054688, 'best_y_average': 0.9678}, 
        #         '146064': {'y_max': 1.0539295673370361, 'y_min': 0.36846235394477844, 'best_y_average': 0.7139}, 
        #         '145804': {'y_max': 1.0046387910842896, 'y_min': 0.24435663223266602, 'best_y_average': 0.7277}, 
        #         '31': {'y_max': 0.7703194618225098, 'y_min': 0.6587125658988953, 'best_y_average': 0.7246},
        #     },
        # }
        self.cheat_table = {}
        if self.search_space_id in self.cheat_table:
            print('Use cheat table for search space id {}'.format(self.search_space_id))
        
    def get_problem_info(self):
        sample_data = self.dataset.trajectory_list[0]
        self.seq_len = sample_data.X.shape[0]
        self.x_dim = sample_data.X.shape[1]
        self.y_dim = 1
        
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
        # device = X.device
        X_np = X.cpu().detach().numpy()
        X_np = xgb.DMatrix(self.transform_x(X_np))
        y = self.bst_surrogate.predict(X_np)
        normalized_y, normalized_regret = self.get_normalized_y_and_regret(y)
        return torch.from_numpy(normalized_y).reshape(-1, 1), {
            "raw_y": torch.from_numpy(y).reshape(-1, 1), 
            "normalized_onestep_regret": torch.from_numpy(normalized_regret).reshape(-1, 1)
        }

    def get_dataset(self):
        return self.dataset

    def get_normalized_y_and_regret(self, y, id=None):
        id = id or self.dataset_id
        if id in self.dataset.global_info["train_datasets"]:
            info = self.dataset.id2info[id]
            y_max, y_min = info["y_max"], info["y_min"]
        else:
            if self.search_space_id in self.cheat_table and id in self.cheat_table[self.search_space_id]:
                y_max = self.cheat_table[self.search_space_id][id]["y_max"]
                y_min = self.cheat_table[self.search_space_id][id]["y_min"]
            else:
                y_max = self.dataset.global_info["y_max_mean"]
                y_min = self.dataset.global_info["y_min_mean"]
        unnormalized_y = y
        unnormalized_regret = y_max - y
        scale = y_max - y_min + 1e-6
        if self.scale_clip_range is not None:
            scale = np.clip(scale, self.scale_clip_range[0], self.scale_clip_range[1])
        normalized_y = (unnormalized_y-y_min) / scale
        normalized_regret = (unnormalized_regret) / scale
        return normalized_y, normalized_regret
        
