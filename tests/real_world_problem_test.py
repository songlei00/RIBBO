import unittest

import torch

from problems.real_world_problem import RealWorldProblem


class RealWorldProblemTest(unittest.TestCase):
    def test_run(self):
        problem_names = (
            'LunarLander',
            'PDE',
            'Optics',
            'Furuta',
            'RobotPush',
            'Rover',
        )
        for name in problem_names:
            for seed in range(3):
                problem = RealWorldProblem(name, str(seed), None)
                dim = problem.dim
                lb, ub = problem.lb, problem.ub
                X = torch.rand(5, dim) * (ub - lb) + lb
                Y = problem(X)
                self.assertEqual(Y.shape, (5, 1))