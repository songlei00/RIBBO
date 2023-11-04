import unittest
from functools import partial

import torch

from datasets.datasets import TrajectoryDataset
from datasets.load_datasets import load_hpob_dataset
from data_augment.data_augment import get_augmented_dataset
from data_augment.operators import (
    random_delete,
    get_increasing_subtrajectory,
)


# class AugmentHPOBTest(unittest.TestCase):
#     def test_augment(self):
#         base_dataset = load_hpob_dataset('4796')
#         random_delete_operator = partial(random_delete, num_delete=3)
#         operators = [
#             random_delete_operator,
#         ]
#         augmented_dataset = get_augmented_dataset(base_dataset, 10, operators)


class OperatorTest(unittest.TestCase):
    def setUp(self):
        problem = 'hpob'
        data_dir = 'data/generated_data/{}/'.format(problem)
        cache_dir = 'cache/{}'.format(problem)
        self.base_dataset = TrajectoryDataset(
            search_space_id='6767',
            data_dir=data_dir,
            cache_dir=cache_dir, 
            input_seq_len=300, 
            normalize_method='random',
            update=False,
        )

    def test_increasing_subtrajectory(self):
        t = self.base_dataset.trajectory_list[0]
        curr_t = get_increasing_subtrajectory(t)
        self.assertEqual(len(curr_t), 150)
        self.assertTrue(torch.all(curr_t.y == torch.sort(curr_t.y).values))

    def test_subtrajectory(self):
        t = self.base_dataset.trajectory_list[0]
        curr_t = get_increasing_subtrajectory(t, increasing=False)
        self.assertEqual(len(curr_t), 150)