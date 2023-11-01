import unittest

import torch

from problems.synthetic import (
    bbob_func_names,
    SyntheticTorch, 
    SyntheticMetaProblem,
)


class SyntheticTest(unittest.TestCase):
    def test_dataset_id(self):
        dim = 10
        X = torch.rand((3, dim)) * 10 - 5
        for sp in bbob_func_names:
            Y = []
            for dataset_id in [str(i) for i in range(50)]:
                problem = SyntheticTorch(sp, dataset_id, dim)
                y_tmp = problem(X)
                self.assertEqual(y_tmp.shape, (3, 1))
                Y.append(y_tmp)
            for i, y1 in enumerate(Y):
                for j, y2 in enumerate(Y):
                    if i < j:
                        self.assertFalse((y1 == y2).all())
