import unittest
import logging

from datasets.load_datasets import load_hpob_dataset


class LoadHPOBTest(unittest.TestCase):
    def test_load(self):
        logging.basicConfig(level=logging.INFO)
        trajectory_dataset = load_hpob_dataset('4796')
        X, y, regret = trajectory_dataset[0]
        self.assertEqual(X.ndim, 2)
        self.assertTrue(regret >= 0)
