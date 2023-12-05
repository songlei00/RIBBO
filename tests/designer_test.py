import unittest
import random

from vizier import pyvizier as vz
from vizier import algorithms as vza

from data_gen.botorch_designer import BotorchDesigner


class DesignerTest(unittest.TestCase):
    def setUp(self):
        problem = vz.ProblemStatement()
        root = problem.search_space.root
        root.add_float_param('x1', 0, 1)
        root.add_float_param('x2', 0, 2)
        root.add_float_param('x3', 0, 5)
        metric = vz.MetricInformation(
            name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
        )
        problem.metric_information.append(metric)
        self._problem = problem
        
    def test_run(self):
        designers = (
            BotorchDesigner(self._problem),
        )
        for designer in designers:
            for _ in range(20):
                suggestion = designer.suggest()
                self.assertEqual(len(suggestion), 1)
                trial = suggestion[0].to_trial().complete(
                    vz.Measurement(metrics={'obj': random.random()})
                )
                designer.update(vza.CompletedTrials([trial]), vza.ActiveTrials())
