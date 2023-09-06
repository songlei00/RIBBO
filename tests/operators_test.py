import unittest

import torch

from datasets.trajectory import Trajectory
from data_augment.operators import (
    random_delete,
    keep_topk_delete,
    duplicate_delete,
)


class OperatorTest(unittest.TestCase):
    def setUp(self):
        X = torch.tensor([
            [1, 1, 2, 3, 5],
            [1, 1, 2, 3, 4],
            [2, 2, 3, 4, 5],
            [2, 2, 3, 5, 6],
        ])
        y = torch.tensor([1, 2, 3, 4])
        self.trajectory = Trajectory(dict(), X, y)

    def test_random_delete(self):
        trajectory, delete_idx = random_delete(self.trajectory, 2)
        self.assertTrue(len(trajectory) + len(delete_idx) == len(self.trajectory))

    def test_keep_topk_delete(self):
        trajectory, delete_idx = keep_topk_delete(self.trajectory, 1, 2)
        self.assertTrue(delete_idx[0] in [0, 1])

    def test_duplicate_delete(self):
        trajectory, delete_idx = duplicate_delete(self.trajectory, 2)
        self.assertTrue((0 in delete_idx) or (1 in delete_idx))
        self.assertTrue((2 in delete_idx) or (3 in delete_idx))
