from typing import Optional, List
from functools import partial

import numpy as np
import torch
from torch import Tensor
from scipy.stats.qmc import Sobol

from problems.base import MetaProblemBase
from datasets.datasets import TrajectoryIterableDataset


## Global optimization benchmark functions
# Branin
def bra(x):
    # the Branin function (2D)
    # https://www.sfu.ca/~ssurjano/branin.html
    x1 = x[:, 0]
    x2 = x[:, 1]

    # scale x
    x1 = x1 * 15.
    x1 = x1 - 5.
    x2 = x2 * 15.

    # parameters
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    bra = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    # normalize
    mean = 54.44
    std = 51.44
    bra = 1 / std * (bra - mean)

    # maximize
    bra = -bra

    return bra.reshape(x.shape[0], 1)


def bra_max_min():
    max_pos = np.array([[-np.pi, 12.275]])
    max_pos[0, 0] += 5.
    max_pos[0, 0] /= 15.
    max_pos[0, 1] /= 15.
    max = bra(max_pos)

    min_pos = np.array([[0.0, 0.0]])
    min = bra(min_pos)

    return max_pos, max, min_pos, min


def bra_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # bound the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.12, 0.87],
                        [-0.81, 0.18]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return s * bra(x_new)


def bra_max_min_var(t, s):
    max_pos, max, min_pos, min = bra_max_min()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.12, 0.87],
                        [-0.81, 0.18]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t

    return max_pos, s * max, min_pos, s * min


# Hartmann-3
def hm3(x):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html

    # parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])

    x = x.reshape(x.shape[0], 1, -1)
    B = x - P
    B = B ** 2
    exponent = A * B
    exponent = np.einsum("ijk->ij", exponent)
    C = np.exp(-exponent)
    hm3 = -np.einsum("i, ki", alpha, C)

    # normalize
    mean = -0.93
    std = 0.95
    hm3 = 1 / std * (hm3 - mean)

    # maximize
    hm3 = -hm3

    return hm3.reshape(x.shape[0], 1)


def hm3_max_min():
    max_pos = np.array([[0.114614, 0.555649, 0.852547]])
    max = hm3(max_pos)

    min_pos = np.array([[1.0, 1.0, 0.0]])
    min = hm3(min_pos)

    return max_pos, max, min_pos, min


def hm3_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.11, 0.88],
                        [-0.55, 0.44],
                        [-0.85, 0.14]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t

    return s * hm3(x_new)


def hm3_max_min_var(t, s):
    # do the transformation in opposite order as in hm3_var!

    max_pos, max, min_pos, min = hm3_max_min()

    # apply translation
    t_range = np.array([[-0.11, 0.88],
                        [-0.55, 0.44],
                        [-0.85, 0.14]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t

    return max_pos, s * max, min_pos, s * min


class MetaBOSynthetic:
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
    ):
        assert search_space_id in ['Branin2', 'Hartmann3']
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id
        self.bound_translation = 0.1
        self.bound_scaling = 0.1

        if self.search_space_id == 'Branin2':
            self.dim = 2
            self.f = bra_var
        elif self.search_space_id == 'Hartmann3':
            self.dim = 3
            self.f = hm3_var
        else:
            raise ValueError('Unsupported search space: {}'.format(search_space_id))

        fct_params_domain = [[-self.bound_translation, self.bound_translation] for _ in range(self.dim)]
        fct_params_domain.append([1-self.bound_scaling, 1+self.bound_scaling])
        fct_params_domain = np.array(fct_params_domain)

        def scale_from_unit_square_to_domain(X, domain):
            # X contains elements in unit square, stretch and translate them to lie domain
            return X * domain.ptp(axis=1) + domain[:, 0]

        sobol = Sobol(self.dim+1, seed=0)
        fct_params = sobol.random(128)
        self.fct_params = scale_from_unit_square_to_domain(fct_params, fct_params_domain)

        idx = int(self.dataset_id)
        self.t = self.fct_params[idx, 0: -1]
        self.s = self.fct_params[idx, -1]

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def __call__(self, X):
        return self.f(X, self.t, self.s)

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id
        idx = int(self.dataset_id)
        self.t = self.fct_params[idx, 0: -1]
        self.s = self.fct_params[idx, -1]


class MetaBOSyntheticTorch(MetaBOSynthetic):
    def __call__(self, X: Tensor) -> Tensor:
        X_np = X.cpu().detach().numpy()
        Y_np = super().__call__(X_np)
        return torch.from_numpy(Y_np)
    

class MetaBOSyntheticMetaProblem(MetaProblemBase):
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

        self.func = MetaBOSynthetic(
            search_space_id,
            '0',
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

        self.dataset.transform_x(partial(self.transform_x, reverse=True))

        self.get_problem_info()

        # TODO: cheat_table
        self.cheat_table = dict()

    def forward(self, X: Tensor):
        assert X.ndim == 2
        assert (X >= -1 - 1e-6).all() and (X <= 1 + 1e-6).all()
        X_np = X.cpu().detach().numpy()
        Y_np = self.func(self.transform_x(X_np))
        normalized_y, normalized_regret = self.get_normalized_y_and_regret(Y_np)
        return torch.from_numpy(normalized_y).reshape(-1, 1), {
            'raw_y': torch.from_numpy(Y_np).reshape(-1, 1),
            'normalized_onestep_regret': torch.from_numpy(normalized_regret).reshape(-1, 1),
        }