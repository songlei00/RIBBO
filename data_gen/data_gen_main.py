from typing import Sequence
try:
    import ujson as json
except:
    import json

import numpy as np
import torch
from vizier import pyvizier as vz 
from vizier import algorithms as vza
from vizier.algorithms import designers
from vizier.benchmarks import experimenters

from hill_climbing_designer import HillClimbingDesigner
from regularized_evolution_designer import RegularizedEvolutionDesigner
from hebo_designer import HeBODesigner
from hpob_problem_statement import problem_statement as hpob_problem_statement
from utils import seed_everything


def designer_factory(name, problem, seed):
    designer_config = {
        'Random': {
            'cls': designers.RandomDesigner,
            'config': {'search_space': problem.search_space, 'seed': seed},
        },
        'GridSearch': {
            'cls': designers.GridSearchDesigner, 
            'config': {'search_space': problem.search_space, 'double_grid_resolution': 10},
        },
        'ShuffledGridSearch': {
            'cls': designers.GridSearchDesigner,
            'config': {'search_space': problem.search_space, 'shuffle_seed': seed, 'double_grid_resolution': 10},
        },
        'RegularizedEvolution': {
            'cls': RegularizedEvolutionDesigner,
            'config': {'problem_statement': problem, 'seed': seed},
        },
        'HillClimbing': {
            'cls': HillClimbingDesigner,
            'config': {'problem_statement': problem, 'seed': seed},
        },
        'EagleStrategy': {
            'cls': designers.EagleStrategyDesigner,
            'config': {'problem_statement': problem, 'seed': seed},
        },
        'Vizier': {
            'cls': designers.VizierGPBandit,
            'config': {'problem': problem},
        },
        'HeBO': {
            'cls': HeBODesigner,
            'config': {'problem_statement': problem, 'rand_sample': 500, 'scramble_seed': seed},
        },
        'CMAES': {
            'cls': designers.CMAESDesigner,
            'config': {'problem_statement': problem, 'seed': seed},
        },
    }
    config = designer_config[name]
    designer = config['cls'](**config['config'])

    return designer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--designer', type=str, required=True)
    parser.add_argument('--search_space_id', type=str, required=True)
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--out_name', type=str, required=True)
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    problem, f = hpob_problem_statement(
        args.search_space_id,
        args.dataset_id,
        args.root_dir,
    )
    designer = designer_factory(args.designer, problem, args.seed)
    metric_name = problem.metric_information.item().name

    # manually initialize
    # trials = [vz.Trial(parameters={'x': 0.5, 'y': 0.6}).complete(vz.Measurement(metrics={'obj': 0.3}))]
    # designer.update(vza.CompletedTrials(trials), vza.ActiveTrials())

    trials = []
    data = {
        'metadata': {
            'designer': args.designer,
            'search_space_id': args.search_space_id,
            'dataset_id': args.dataset_id,
            'length': args.length,
            'seed': args.seed,
        },
        'X': [],
        'y': [],
    }
    for i in range(args.length):
        suggestion = designer.suggest(count=1)[0]

        # evaluate
        X = {name: param.value for name, param in suggestion.parameters.items()}
        X_tensor = torch.zeros(f.dim)
        for i in range(f.dim):
            X_tensor[i] = X['x{}'.format(i)]
        objective = f(X_tensor.unsqueeze(0)).item()

        trial = suggestion.to_trial().complete(
            vz.Measurement(metrics={metric_name: objective})
        )

        trials.append(trial)
        designer.update(vza.CompletedTrials([trial]), vza.ActiveTrials())

        data['X'].append(X_tensor.tolist())
        data['y'].append(objective)

    with open(args.out_name, 'w') as f:
        json.dump(data, f)
