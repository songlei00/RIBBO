import unittest

import numpy as np

from problems.metabo_synthetic import MetaBOSynthetic


class SyntheticTest(unittest.TestCase):
    def test_branin_run(self):
        func = MetaBOSynthetic('Branin2', '0')
        X = np.random.random((10, 2))
        y = func(X)
        self.assertEqual(y.shape, (10, 1))

    def test_hartmann_run(self):
        func = MetaBOSynthetic('Hartmann3', '0')
        X = np.random.random((10, 3))
        y = func(X)
        self.assertEqual(y.shape, (10, 1))
