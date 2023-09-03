import unittest
from functools import partial

from datasets.load_datasets import load_hpob_dataset
from data_augment.data_augment import get_augmented_dataset
from data_augment.operators import (
    random_delete
)


class AugmentHPOBTest(unittest.TestCase):
    def test_augment(self):
        base_dataset = load_hpob_dataset('4796')
        random_delete_operator = partial(random_delete, num_delete=3)
        operators = [
            random_delete_operator,
        ]
        augmented_dataset = get_augmented_dataset(base_dataset, 10, operators)