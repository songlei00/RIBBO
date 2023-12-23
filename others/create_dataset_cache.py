try:
    import ujson as json
except:
    import json
import logging
import argparse

from datasets.datasets import TrajectoryDataset

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, required=True, choices=['hpob', 'synthetic', 'metabo_synthetic', 'real_world_problem'])
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'validation'])
args = parser.parse_args()

mode = args.mode

if args.mode == 'train':
    s, e = 0, 50
elif args.mode == 'validation':
    s, e = 50, 60
else: # test
    s, e = 60, 70

if args.problem == 'hpob':
    with open('./others/hpob-summary-stats/{}-summary-stats.json'.format(mode), 'r') as f:
        summary_stats = json.load(f)
elif args.problem == 'synthetic':
    from problems.synthetic import bbob_func_names
    summary_stats = {name: [str(i) for i in range(s, e)] for name in bbob_func_names}
elif args.problem == 'metabo_synthetic':
    names = ('Branin2', 'Hartmann3')
    summary_stats = {name: [str(i) for i in range(s, e)] for name in names}
elif args.problem == 'real_world_problem':
    names = (
        'LunarLander',
        'PDE',
        'Optics',
        'RobotPush',
        'Rover',
        'Furuta',
    )
    summary_stats = {name: [str(i) for i in range(s, e)] for name in names}
else:
    raise NotImplementedError

if mode == 'train':
    data_dir = 'data/generated_data/{}/'.format(args.problem)
    cache_dir = 'cache/{}'.format(args.problem)
else:
    data_dir = 'data/generated_data/{}_{}/'.format(args.problem, mode)
    cache_dir = 'cache/{}_{}'.format(args.problem, mode)

for search_space_id in summary_stats:
# for search_space_id in ['6767', '5906', '7609', '7607', '6794']:
# for search_space_id in ['Rastrigin']:
    print('search space: {}'.format(search_space_id))
    dataset = TrajectoryDataset(
        search_space_id=search_space_id,
        data_dir=data_dir,
        cache_dir=cache_dir, 
        input_seq_len=300, 
        normalize_method='random',
    )

    print('length:', len(dataset))
    print('x dim:', dataset.trajectory_list[0].X.shape[1])
    print(dataset.global_info)

    algo2cnt = dict()
    for t in dataset.trajectory_list:
        designer = t.metadata['designer']
        algo2cnt[designer] = algo2cnt.get(designer, 0) + 1
    print(algo2cnt)