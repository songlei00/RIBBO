from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor


class ProblemBase(metaclass=ABCMeta):
    def __call__(self, X: Tensor) -> Tensor:
        assert torch.is_tensor(X)
        assert X.ndim == 2
        assert X.shape[-1] == self.dim
        return self.forward(X)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass


class MetaProblemBase(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass

    def get_problem_info(self):
        sample_data = self.dataset.trajectory_list[0]
        self.seq_len = sample_data.X.shape[0]
        self.x_dim = sample_data.X.shape[1]
        self.y_dim = 1

    def transform_x(self, x, reverse: bool=False, lb=0.0, ub=1.0):
        if reverse:
            x = (x - lb) / (ub - lb)
            return x * 2 - 1.0
        else:
            x = x / 2 + 0.5
            return x * (ub - lb) + lb

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.func.reset_task(dataset_id)

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
        