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

def scale_from_domain_to_unit_square(X, domain):
    # X contains elements in domain, translate and stretch them to lie in unit square
    return (X - domain[:, 0]) / domain.ptp(axis=1)


class RealWorldProblem:
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        root_dir: Optional[str] = None,
        noist_std: float = 0,
    ):
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id

        if search_space_id in (
            'LunarLander',
            'PDE',
            'Optics',
            'RobotPush',
            'Rover',
        ):
            if search_space_id in ('LunarLander', 'PDE', 'Optics'):
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
            elif search_space_id in ('RobotPush', 'Rover'):
                if search_space_id == 'RobotPush':
                    from problems.real_world_utils.push_function import PushReward
                    self.func = PushReward()
                    self.dim = 14
                elif search_space_id == 'Rover':
                    from problems.real_world_utils.rover_function import create_rover_problem
                    self.func = create_rover_problem()
                    self.dim = 60
            else:
                raise ValueError
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
            params = sobol.random(512)
            self.params = scale_from_unit_square_to_domain(params, params_domain)

            idx = int(self.dataset_id)
            self.t = self.params[idx, 0: -1]
            self.s = self.params[idx, -1]
        elif search_space_id == 'Furuta':
            self.dim = 4
            self.lb = torch.zeros(self.dim)
            self.ub = torch.ones(self.dim)
            # environment parameter
            true_mass_arm = 0.095
            true_mass_pendulum = 0.024
            true_length_arm = 0.085
            true_length_pendulum = 0.129
            low_mult = 0.1
            high_mult = 2.0

            mass_arm_low = low_mult * true_mass_arm
            mass_arm_high = high_mult * true_mass_arm
            mass_pendulum_low = low_mult * true_mass_pendulum
            mass_pendulum_high = high_mult * true_mass_pendulum
            length_arm_low = low_mult * true_length_arm
            length_arm_high = high_mult * true_length_arm
            length_pendulum_low = low_mult * true_length_pendulum
            length_pendulum_high = high_mult * true_length_pendulum

            physical_params_domain = np.array([
                [mass_pendulum_low, mass_pendulum_high],
                [mass_arm_low, mass_arm_high],
                [length_pendulum_low, length_pendulum_high],
                [length_arm_low, length_arm_high],
            ])
            sobol = Sobol(self.dim, seed=0)
            physical_params = sobol.random(512)
            self.physical_params = scale_from_unit_square_to_domain(
                X=physical_params,
                domain=physical_params_domain
            )

            self.reset_task('0')
        else:
            raise NotImplementedError
        self.name = search_space_id

    def __call__(self, X) -> Any:
        return self.forward(X)
        
    def forward(self, X: Tensor) -> Tensor:
        assert X.ndim == 2
        assert (X >= self.lb).all() and (X <= self.ub).all()
        if self.search_space_id in (
            'LunarLander',
            'PDE',
            'Optics',
        ):
            Y = self.s * self.func(X - self.t)
            return Y.reshape(-1, 1).to(X)
        elif self.search_space_id in ('RobotPush', 'Rover'):
            X_np = X.cpu().detach().numpy()
            Y = []
            for x in X_np:
                y = self.s * self.func(x - self.t)
                Y.append(y)
            return torch.from_numpy(np.array(Y)).reshape(-1, 1).to(X)
        elif self.search_space_id == 'Furuta':
            X_np = X.cpu().detach().numpy()
            Y = []
            for x in X_np:
                y = self.func(x)
                Y.append(y)
            return torch.from_numpy(np.array(Y)).reshape(-1, 1).to(X)
        else:
            raise ValueError

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id

        if self.search_space_id in (
            'LunarLander',
            'PDE',
            'Optics',
            'RobotPush',
            'Rover',
        ):
            idx = int(self.dataset_id)
            self.t = self.params[idx, 0: -1]
            self.s = self.params[idx, -1]
        elif self.search_space_id == 'Furuta':
            from problems.real_world_utils.furuta.furuta import init_furuta_simulation, furuta_simulation
            furuta_domain = np.array(
                [[-0.5, 0.2],
                [-1.6, 4.0],
                [-0.1, 0.04],
                [-0.04, 0.1]]
            )
            idx = int(self.dataset_id)
            mass_pendulum = self.physical_params[idx, 0]
            mass_arm = self.physical_params[idx, 1]
            length_pendulum = self.physical_params[idx, 2]
            length_arm = self.physical_params[idx, 3]

            init_tuple = init_furuta_simulation(
                mass_arm=mass_arm,
                length_arm=length_arm,
                mass_pendulum=mass_pendulum,
                length_pendulum=length_pendulum
            )

            pos = [0, 1, 2, 3]
            self.func = lambda x: -furuta_simulation(
                init_tuple=init_tuple,
                params=scale_from_unit_square_to_domain(x, furuta_domain),
                D=self.dim,
                pos=pos
            ).reshape(-1, 1)
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
        n_block: int = 1,
        filter_data: bool = False,
    ):
        self.search_space_id = search_space_id
        self.input_seq_len = input_seq_len
        self.scale_clip_range = scale_clip_range

        self.func = RealWorldProblem(
            self.search_space_id,
            '0',
        )
        self.dim = self.func.dim
        self.lb, self.ub = self.func.lb, self.func.ub
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
            n_block=n_block,
            filter_data=filter_data,
        )

        self.dataset.transform_x(partial(self.transform_x, reverse=True, lb=self.lb, ub=self.ub))

        self.get_problem_info()

        # TODO: cheat_table
        self.cheat_table = dict()

    def forward(self, X: Tensor) -> Tensor:
        assert X.ndim == 2
        assert (X >= -1 - 1e-6).all() and (X <= 1 + 1e-6).all()
        Y = self.func(self.transform_x(X, lb=self.lb, ub=self.ub))
        Y_np = Y.cpu().detach().numpy()
        normalized_y, normalized_regret = self.get_normalized_y_and_regret(Y_np)
        return torch.from_numpy(normalized_y).reshape(-1, 1), {
            'raw_y': torch.from_numpy(Y_np).reshape(-1, 1),
            'normalized_onestep_regret': torch.from_numpy(normalized_regret).reshape(-1, 1),
        }
