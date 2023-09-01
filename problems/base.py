from abc import ABCMeta, abstractmethod

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
