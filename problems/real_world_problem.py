from typing import Any, Optional, List
from functools import partial

import numpy as np
import torch
from torch import Tensor
from scipy.stats.qmc import Sobol

from problems.base import ProblemBase, MetaProblemBase
from datasets.datasets import TrajectoryIterableDataset


def scale_from_unit_square_to_domain(X, domain):
    # X contains elements in unit square, stretch and translate them to lie domain
    return X * domain.ptp(axis=1) + domain[:, 0]


class RealWorldProblem:
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        root_dir: str,
        noist_std: float = 0,
    ):
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id

        if search_space_id in (
            'LunarLander',
            'PDE',
            'Optics',
        ):
            if search_space_id == 'LunarLander':
                from problems.real_world_utils.lunar_lander import LunarLanderProblem
                func_cls = LunarLanderProblem
            elif search_space_id == 'PDE':
                from problems.real_world_utils.pdes import PDEVar
                func_cls = PDEVar
            elif search_space_id == 'Optics':
                from problems.real_world_utils.optics import Optics
                func_cls = Optics
            else:
                raise ValueError
            self.func = func_cls()
            self.dim = self.func.dim
            # we normalize X in evaluate_true function within the problem
            # so the bound is [0, 1] here
            self.lb = torch.zeros(self.dim)
            self.ub = torch.ones(self.dim)

            # transform
            bound_translation = 0.1
            bound_scaling = 0.1
            params_domain = [[-bound_translation, bound_translation] for _ in range(self.dim)]
            params_domain.append([1-bound_scaling, 1+bound_scaling])
            params_domain = np.array(params_domain)
            sobol = Sobol(self.dim+1, seed=0)
            params = sobol.random(128)
            self.params = scale_from_unit_square_to_domain(params, params_domain)

            idx = int(self.dataset_id)
            self.t = self.params[idx, 0: -1]
            self.s = self.params[idx, -1]
        elif search_space_id == 'RobotPush':
            pass
        elif search_space_id == 'Furuta':
            pass
        else:
            raise NotImplementedError
        self.name = search_space_id

    def __call__(self, X) -> Any:
        return self.forward(X)
        
    def forward(self, X: Tensor) -> Tensor:
        assert (X >= self.lb).all() and (X <= self.ub).all()
        if self.search_space_id in (
            'LunarLander',
            'PDE',
            'Optics',
        ):
            Y = self.s * self.func(X - self.t)
            return Y.reshape(-1, 1).to(X)
        elif self.search_space_id == 'RobotPush':
            pass
        elif self.search_space_id == 'Furuta':
            pass
        else:
            raise ValueError

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id

        if self.search_space_id in (
            'LunarLander',
            'PDE',
            'Optics',
        ):
            idx = int(self.dataset_id)
            self.t = self.params[idx, 0: -1]
            self.s = self.params[idx, -1]
        elif self.search_space_id == 'RobotPush':
            pass
        elif self.search_space_id == 'Furuta':
            pass
        else:
            raise ValueError


class RealWorldMetaProblem(MetaProblemBase):
    def __init__(
        self,
        search_space_id: str,
        root_dir: str,
        data_dir: str,
        cache_dir: str,
        input_seq_len: int = 300,
        max_input_seq_len: int = 300,
        normalize_method: str = 'random',
        scale_clip_range: Optional[List[float]] = None,
        augment: bool = False,
        prioritize: bool = False,
        prioritize_alpha: float = 1.0,
    ):
        self.search_space_id = search_space_id
        self.input_seq_len = input_seq_len
        self.scale_clip_range = scale_clip_range

        self.func = RealWorldProblem(
            self.search_space_id,
            '0',
            self.dim,
            self.lb,
            self.ub,
        )
        self.dataset = TrajectoryIterableDataset(
            search_space_id=search_space_id,
            data_dir=data_dir,
            cache_dir=cache_dir,
            input_seq_len=input_seq_len,
            max_input_seq_len=max_input_seq_len,
            normalize_method=normalize_method,
            scale_clip_range=scale_clip_range,
            prioritize=prioritize,
            prioritize_alpha=prioritize_alpha,
        )

        self.dataset.transform_x(partial(self.transform_x, reverse=True, lb=self.lb, ub=self.ub))

        self.get_problem_info()

        # TODO: cheat_table
        self.cheat_table = dict()


