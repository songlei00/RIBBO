from typing import Dict
try:
    import ujson as json
except:
    import json

from data_augment.metrics import (
    metric_best_y,
    metric_regret,
    metric_diversity_X,
    metric_diversity_y,
)


class Trajectory:
    def __init__(self, metadata, X, y):
        self.metadata: dict = metadata
        self.X = X 
        self.y = y 

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return str(self.metadata)

    @classmethod
    def load_from_json(cls, path):
        with open(path, 'r') as f:
            trajectory = json.load(f)
        return Trajectory(trajectory['metadata'], trajectory['X'], trajectory['y'])

    @property
    def best_y(self):
        return metric_best_y(self)

    @property
    def regret(self):
        return metric_regret(self)

    @property
    def diversity_X(self):
        return metric_diversity_X(self)

    @property
    def diversity_y(self):
        return metric_diversity_y(self)
