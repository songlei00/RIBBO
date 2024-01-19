import yaml
import os 
import copy
from collections import defaultdict

import numpy as np
import torch
from UtilsRL.exp import parse_args, setup

from problems.hpob_problem import HPOBMetaProblem
from problems.synthetic import SyntheticMetaProblem
from problems.metabo_synthetic import MetaBOSyntheticMetaProblem
from problems.real_world_problem import RealWorldMetaProblem
from algorithms.designers.dt_designer import DecisionTransformerDesigner, evaluate_decision_transformer_designer
from algorithms.modules.dt import DecisionTransformer
from algorithms.designers.bc_designer import BCTransformerDesigner, evaluate_bc_transformer_designer
from algorithms.modules.bc import BCTransformer
from algorithms.designers.optformer_designer import OptFormerDesigner, evaluate_optformer_designer
from algorithms.modules.optformer import OptFormerTransformer

def rollout_designer(problem, designer, datasets, eval_episode, eval_mode, **kwargs):
    if isinstance(designer, BCTransformerDesigner):
        metrics, record = evaluate_bc_transformer_designer(
            problem=problem, 
            designer=designer, 
            datasets=datasets, 
            eval_episode=eval_episode, 
            eval_mode=eval_mode, 
        )
    elif isinstance(designer, OptFormerDesigner):
        metrics, record = evaluate_optformer_designer(
            problem=problem, 
            designer=designer, 
            datasets=datasets, 
            eval_episode=eval_episode, 
            eval_mode=eval_mode, 
            algo=kwargs.get("algo")
        )
    elif isinstance(designer, DecisionTransformerDesigner):
        metrics, record = evaluate_decision_transformer_designer(
            problem=problem, 
            designer=designer, 
            datasets=datasets, 
            eval_episode=eval_episode, 
            eval_mode=eval_mode, 
            init_regret=kwargs.get("init_regret"), 
            regret_strategy=kwargs.get("regret_strategy")
        )
        
    return record

def load_model(problem, ckpt_cfgs):
    ckpts = {}
    for name, config in ckpt_cfgs.items():
        print(f'Load from {config["path"]}')
        if config["type"] == "bc":
            new_args = copy.deepcopy(args.bc_config)
            for k in config["args"]:
                new_args[k] = config["args"][k]
            designers = []
            for p in config["path"]:
                transformer = BCTransformer(problem.x_dim, problem.y_dim, new_args.embed_dim, new_args.num_layers, problem.seq_len, new_args.num_heads, True, new_args.mix_method, new_args.attention_dropout, new_args.residual_dropout, new_args.embed_dropout, new_args.pos_encoding)
                designer = BCTransformerDesigner(transformer, problem.x_dim, problem.y_dim, new_args.embed_dim, problem.seq_len, new_args.input_seq_len, new_args.x_type, new_args.y_loss_coeff, new_args.use_abs_timestep, "cpu")
                designer.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
                designer.to(args.device)
                designers.append(designer)
        elif config["type"] == "dt":
            new_args = copy.deepcopy(args.dt_config)
            for k in config["args"]:
                new_args[k] = config["args"][k]
            designers = []
            for p in config["path"]:
                transformer = DecisionTransformer(problem.x_dim, problem.y_dim, new_args.embed_dim, new_args.num_layers, problem.seq_len, new_args.num_heads, True, new_args.mix_method, new_args.attention_dropout, new_args.residual_dropout, new_args.embed_dropout, new_args.pos_encoding)
                designer = DecisionTransformerDesigner(transformer, problem.x_dim, problem.y_dim, new_args.embed_dim, problem.seq_len, new_args.input_seq_len, new_args.x_type, new_args.y_loss_coeff, new_args.use_abs_timestep, "cpu")
                designer.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
                designer.to(args.device)
                designers.append(designer)
        elif config["type"] == "optformer":
            new_args = copy.deepcopy(args.optformer_config)
            for k in config["args"]:
                new_args[k] = config["args"][k]
            designers = []
            for p in config["path"]:
                transformer = OptFormerTransformer(problem.x_dim, problem.y_dim, new_args.embed_dim, new_args.num_layers, problem.seq_len, new_args.num_heads, new_args.algo_num, new_args.mix_method, new_args.attention_dropout, new_args.residual_dropout, new_args.embed_dropout, new_args.pos_encoding)
                designer = OptFormerDesigner(transformer, problem.x_dim, problem.y_dim, new_args.embed_dim, problem.seq_len, new_args.input_seq_len, new_args.x_type, new_args.y_loss_coeff, new_args.use_abs_timestep, "cpu")
                designer.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
                designer.to(args.device)
                designers.append(designer)
        ckpts[name] = designers
    return ckpts

def model_select(data):
    ret = dict()
    for name in data:
        mean = np.stack([v.mean(0) for v in data[name].values()], axis=0).mean(0)
        std = np.stack([v.std(0) for v in data[name].values()], axis=0).mean(0)
        sum_y = np.sum(mean)
        ret[name] = sum_y
    return ret

def extract_data(name2rollout, key):
    data = {
        n: {id: v[key] for id, v in r.items()}
        for n, r in name2rollout.items()
    }
    return data

def post_init(args):
    args.train_datasets = args.train_datasets[args.eval_id][:30]
    args.test_datasets = args.test_datasets[args.eval_id][:15]
    args.eval_episodes = 20
    args.deterministic_eval = False
    args.problem_cls = {
        "hpob": HPOBMetaProblem, 
        "synthetic": SyntheticMetaProblem,
        "metabo_synthetic": MetaBOSyntheticMetaProblem,
        "real_world_problem": RealWorldMetaProblem,
    }.get(args.problem)
    
args = parse_args(post_init=post_init)
setup(args, _seed=0)

# define the problem and the dataset
problem_dict = dict()
# for mode in ('train', 'test', 'validation'):
for mode in ('train', 'test'):
    if mode == 'train':
        data_dir = args.data_dir
        cache_dir = args.cache_dir
    else:
        data_dir = args.data_dir.rstrip('/') + '_' + mode + '/'
        cache_dir = args.cache_dir.rstrip('/') + '_' + mode + '/'

    if os.path.exists(cache_dir + args.eval_id):
        print(f'Load {mode} data')
        problem = args.problem_cls(
            search_space_id=args.eval_id, 
            root_dir=args.root_dir, 
            data_dir=data_dir,
            cache_dir=cache_dir, 
            input_seq_len=1, # does not matter here
            max_input_seq_len=args.max_input_seq_len,
            normalize_method='random', # does not matter here 
            scale_clip_range=None, # does not matter here
            prioritize=False, # does not matter here
            n_block=1,
        )
        problem_dict[mode] = problem

ckpt_cfgs = dict()
with open(f'scripts/ckpt_configs/{args.problem}/{args.ckpt_id}.yaml', 'r') as f:
    load_dict = yaml.safe_load(f)
    for name in load_dict:
        if name == 'Var':
            continue
        cfg = dict()
        path = load_dict[name]['path']
        epochs = load_dict[name]['epochs']
        del load_dict[name]['epochs']
        for epoch in epochs:
            curr_path = [
                f'log/{args.problem}/{p}/ckpt/{epoch}.ckpt' for p in path
            ]
            key = name if len(epochs) == 1 else name + '_' + str(epoch)
            cfg[key] = dict(load_dict[name])
            cfg[key]['path'] = curr_path
        ckpt_cfgs.update(cfg)
        
print('===== ckpt configs =====')
for k, v in ckpt_cfgs.items():
    print(k, v)
print('========================')

for mode, problem in problem_dict.items():
    ckpts = load_model(problem, ckpt_cfgs)
    if mode == 'train':
        rollout_datasets = args.train_datasets
    elif mode == 'validation':
        rollout_datasets = args.validation_datasets
    else: # test
        rollout_datasets = args.test_datasets
    name2rollout = defaultdict(dict)
    for name in ckpt_cfgs:
        rollout_args = ckpt_cfgs[name]["rollout_args"]
        rollout_res = []
        for designer in ckpts[name]:
            rollout_res.append(
                rollout_designer(problem, designer, rollout_datasets, args.eval_episodes, **rollout_args)
            )
        name2rollout[name] = rollout_res

    # concatenate
    for name in name2rollout:
        if isinstance(name2rollout[name], list):
            id2data = copy.deepcopy(name2rollout[name][0])
            for id in id2data:
                for entry in id2data[id]:
                    id2data[id][entry] = np.stack([name2rollout[name][ii][id][entry] for ii in range(len(name2rollout[name]))], axis=0)
            name2rollout[name] = id2data
        else:
            for id in name2rollout[name]:
                for entry in name2rollout[name][id]:
                    if len(name2rollout[name][id][entry].shape) == 1:
                        name2rollout[name][id][entry] = name2rollout[name][id][entry][None, ...]
    for name in name2rollout:
        for id in name2rollout[name]:
            data = name2rollout[name][id]["normalized_regret"]
            name2rollout[name][id]["normalized_cumulative_regret"] = np.flip(np.flip(data, 1).cumsum(axis=1), 1)

    # extract
    normalized_y_data = extract_data(name2rollout, 'normalized_y')
    best_value = dict()
    for name in name2rollout:
        best_value[name] = dict()
        for id in name2rollout[name]:
            best_normalized_y = np.zeros_like(normalized_y_data[name][id])
            for i in range(best_normalized_y.shape[1]):
                best_normalized_y[:, i] = np.max(normalized_y_data[name][id][:, :i+1], axis=1)
            best_value[name][id] = best_normalized_y

    ret = model_select(normalized_y_data)
    print(ret)
