import logging
from typing import Optional, List
from functools import partial

import numpy as np
import torch
from torch import Tensor

from problems.base import ProblemBase
from problems import bbob
from datasets.datasets import TrajectoryIterableDataset

bbob_func_names = (
    'Rastrigin',
    'LinearSlope',
    'AttractiveSector',
    'StepEllipsoidal',
    'RosenbrockRotated',
    'Discus',
    'BentCigar',
    'SharpRidge',
    'DifferentPowers',
    'Weierstrass',
    'SchaffersF7',
    'SchaffersF7IllConditioned',
    'GriewankRosenbrock',
    'Katsuura',
    'Lunacek',
    'Gallagher101Me',
    'Gallagher21Me',
    'NegativeSphere',
    'NegativeMinDifference',
)
bbob_func_dict = dict()
for name in bbob_func_names:
    bbob_func_dict[name] = getattr(bbob, name) # minimize

log = logging.getLogger(__name__)


class SyntheticNumpy(ProblemBase):
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        dim: int,
        lb: float = -5,
        ub: float = 5,
        noise_std: float = 0,
    ):
        assert search_space_id in bbob_func_dict
        self.search_space = search_space_id
        self.dataset_id = dataset_id # seed
        self.dim = dim
        
        self.func = bbob_func_dict[search_space_id]
        self.lb = np.ones(dim) * lb
        self.ub = np.ones(dim) * ub
        self.name = 'Synthetic_{}'.format(search_space_id)

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert (X >= self.lb).all() and (X <= self.ub).all()
        Y = []
        for x in X:
            y = self.func(x, int(self.dataset_id))
            Y.append(y)
        return - np.array(Y).reshape(-1, 1) # maximize

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id


class SyntheticTorch(SyntheticNumpy):
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        dim: int,
        lb: float = -5,
        ub: float = 5,
        noise_std: float = 0,
    ):
        super().__init__(
            search_space_id,
            dataset_id,
            dim,
            lb,
            ub,
            noise_std,
        )

    def forward(self, X: Tensor) -> Tensor:
        X_np = X.cpu().detach().numpy()
        Y_np = super().forward(X_np)
        return torch.from_numpy(Y_np)


class SyntheticMetaProblem:
    def __init__(
        self,
        search_space_id: str,
        root_dir: str,
        data_dir: str,
        cache_dir: str,
        input_seq_len: int = 300,
        normalize_method: str = 'random',
        scale_clip_range: Optional[List[float]] = None,
        prioritize: bool = False,
        prioritize_alpha: float = 1.0,
    ):
        self.dim = 10
        self.lb, self.ub = -5, 5
        self.search_space_id = search_space_id
        self.root_dir = root_dir
        self.input_seq_len = input_seq_len
        self.scale_clip_range = scale_clip_range

        self.func = SyntheticNumpy(
            self.search_space_id,
            self.dataset_id,
            self.dim,
            self.lb,
            self.ub,
        )
        self.dataset = TrajectoryIterableDataset(
            search_space_id=search_space_id,
            data_dir=data_dir,
            cache_dir=cache_dir,
            input_seq_len=input_seq_len,
            normalize_method=normalize_method,
            scale_clip_range=scale_clip_range,
            prioritize=prioritize,
            prioritize_alpha=prioritize_alpha,
        )

        self.dataset.transform_x(partial(self.transform_x, reverse=True))

        self.get_problem_info()

        # TODO: cheat_table
        self.cheat_table = dict()

    def get_problem_info(self):
        sample_data = self.dataset.trajectory_list[0]
        self.seq_len = sample_data.X.shape[0]
        self.x_dim = sample_data.X.shape[1]
        self.y_dim = 1

    def transform_x(self, x, reverse: bool=False):
        lb, ub = self.lb, self.ub
        if reverse:
            x = (x - lb) / (ub - lb)
            return x * 2 - 1.0
        else:
            x = x / 2 + 0.5
            return x * (ub - lb) + lb

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.func.reset_task(dataset_id)

    def forward(self, X: Tensor):
        assert X.ndim == 2
        assert (X >= -1 - 1e-6).all() and (X <= 1 + 1e-6).all()
        X_np = X.cpu().detach().numpy()
        Y_np = self.func(X_np)
        normalized_y, normalized_regret = self.get_normalized_y_and_regret(Y_np)
        return torch.from_numpy(normalized_y).reshape(-1, 1), {
            'raw_y': torch.from_numpy(Y_np).reshape(-1, 1),
            'normalized_onestep_regret': torch.from_numpy(normalized_regret).reshape(-1, 1),
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
        