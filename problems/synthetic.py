import logging

import torch
from torch import Tensor
from botorch.test_functions import synthetic

from problems.base import ProblemBase

log = logging.getLogger(__name__)


class Synthetic(ProblemBase):
    options = [
        'Ackley',
        'Branin',
        'Hartmann',
        'Levy',
        'Rosenbrock',
        'Shekel',
        'Michalewicz',
    ]
    fixed_dim_options = [
        'Branin',
        'Shekel',
    ]
    def __init__(
        self,
        name: str,
        dim: int,
        noise_std: float = 0,
        negate: bool = True,
    ):
        assert name in Synthetic.options
        self.name = name
        if name == 'Branin':
            self.dim = 2
        elif name == 'Shekel':
            self.dim = 4
        else:
            self.dim = dim
        if name in Synthetic.fixed_dim_options:
            log.info('Set to a fixed dim {} for {}'.format(self.dim, name))

        self.noise_std = noise_std
        self.negate = negate
        kwargs = {
            'noise_std': noise_std,
            'negate': negate,
        }
        if name not in Synthetic.fixed_dim_options:
            kwargs['dim'] = self.dim
        self.func = getattr(synthetic, name)(**kwargs)
        lb, ub = zip(*self.func._bounds)
        self.lb, self.ub = torch.tensor(lb), torch.tensor(ub)

    def forward(self, X: Tensor) -> Tensor:
        Y = self.func(X)
        return Y.reshape(-1, 1)