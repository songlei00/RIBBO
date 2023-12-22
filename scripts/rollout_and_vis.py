#%%
import os
import copy
import torch
import wandb
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
from collections import defaultdict
from torch.utils.data import DataLoader
from algorithms.designers.dt_designer import DecisionTransformerDesigner, evaluate_decision_transformer_designer
from algorithms.modules.dt import DecisionTransformer
from algorithms.designers.bc_designer import BCTransformerDesigner, evaluate_bc_transformer_designer
from algorithms.modules.bc import BCTransformer
from algorithms.designers.optformer_designer import OptFormerDesigner, evaluate_optformer_designer
from algorithms.modules.optformer import OptFormerTransformer
from algorithms.modules.dt import DecisionTransformer
from problems.hpob_problem import HPOBMetaProblem
from problems.synthetic import SyntheticMetaProblem
from problems.metabo_synthetic import MetaBOSyntheticMetaProblem

from UtilsRL.exp import parse_args, setup

#%%    

def average(arr):
    return sum(arr) / len(arr)

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

def add_behavior(behavior_cfgs, problem, datasets): 
    if not behavior_cfgs["add_behavior"]:
        return {}
    num = behavior_cfgs["num"]
    dataset = problem.get_dataset()
    tasks = set(datasets)
    algos = set([k for k in behavior_cfgs if k != "num" and k != "add_behavior" and behavior_cfgs[k] == True])
    behavior_rollout = defaultdict(partial(defaultdict, list))
    for l in dataset.trajectory_list:
        if l.metadata["dataset_id"] in tasks and l.metadata["designer"] in algos:
            behavior_rollout[l.metadata["designer"]][l.metadata["dataset_id"]].append(l.y)
            
    name2rollout = {}
    for a in algos:
        if a not in behavior_rollout:
            print(f'Trajectory for {a} is empty')
            continue
        name2rollout[a] = {}
        for t in tasks:
            name2rollout[a][t] = {}
            y = behavior_rollout[a][t][:num]
            y = torch.stack(y, dim=0)
            # y = y.reshape([5, num, -1]).mean(dim=1) # TODO: align testing
            normalized_y, normalized_regret = problem.get_normalized_y_and_regret(y, id=t)
            name2rollout[a][t]["y"] = y.numpy()
            name2rollout[a][t]["normalized_y"] = normalized_y.numpy()
            name2rollout[a][t]["normalized_regret"] = normalized_regret.numpy()
    del behavior_rollout
    return name2rollout          

def plot(name2rollout, datasets, output_path, palette=None):
    if palette is None:
        palette = [
            'violet', 'slategray', 'darkviolet', 'royalblue', 'mediumseagreen', 'orange', 'red',
            'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 
            'teal',  'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 
            'gold',  'darkred', 'darkblue',
        ]
        palette = {
            n: palette[i] for i, n in enumerate(name2rollout.keys())
        }
    print(f"Saving to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    total_num = len(datasets) + 2
    nrows, ncols = 1+(total_num-1)//4, 4
    
    # concatenate the seeds
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
        for id in datasets:
            data = name2rollout[name][id]["normalized_regret"]
            name2rollout[name][id]["normalized_cumulative_regret"] = np.flip(np.flip(data, 1).cumsum(axis=1), 1)
            
    # 1. plot y
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.reshape(-1)
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            mean = name2rollout[name][id]["y"].mean(0)
            std = name2rollout[name][id]["y"].std(0)
            step_metric = np.arange(len(mean))
            axes[idx].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
            axes[idx].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    for name in name2rollout:
        axes[-2].set_title('agg')
        mean = np.stack([v["y"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
        std = np.stack([v["y"].std(0) for v in name2rollout[name].values()], axis=0).mean(0)
        step_metric = np.arange(len(mean))
        axes[-2].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
        axes[-2].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    for name in name2rollout:
        axes[-1].plot([], label=name, color=palette[name])
    axes[-1].legend()
    plt.savefig(os.path.join(output_path, "y.png"))
    plt.clf()
    
    # 2. plot normalized y
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.reshape(-1)
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            mean = name2rollout[name][id]["normalized_y"].mean(0)
            std = name2rollout[name][id]["normalized_y"].std(0)
            step_metric = np.arange(len(mean))
            axes[idx].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
            axes[idx].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    for name in name2rollout:
        axes[-2].set_title('agg')
        mean = np.stack([v["normalized_y"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
        std = np.stack([v["normalized_y"].std(0) for v in name2rollout[name].values()], axis=0).mean(0)
        step_metric = np.arange(len(mean))
        axes[-2].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
        axes[-2].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    for name in name2rollout:
        axes[-1].plot([], label=name, color=palette[name])
    axes[-1].legend()
    plt.savefig(os.path.join(output_path, "normalized_y.png"))
    plt.clf()
    
    # 3. plot regret
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.reshape(-1)
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            mean = name2rollout[name][id]["normalized_cumulative_regret"].mean(0)
            std = name2rollout[name][id]["normalized_cumulative_regret"].std(0)
            step_metric = np.arange(len(mean))
            axes[idx].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
            axes[idx].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    for name in name2rollout:
        axes[-2].set_title('agg')
        mean = np.stack([v["normalized_cumulative_regret"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
        std = np.stack([v["normalized_cumulative_regret"].std(0) for v in name2rollout[name].values()], axis=0).mean(0)
        step_metric = np.arange(len(mean))
        axes[-2].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
        axes[-2].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    for name in name2rollout:
        axes[-1].plot([], label=name, color=palette[name])
    axes[-1].legend()
    plt.savefig(os.path.join(output_path, "regret.png"))
    plt.clf()
    
    # 4. plot the aggregations
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
    axes = axes.reshape(-1)
    axes[0].set_title("Normalized Y")
    for name in name2rollout:
        mean = np.stack([v["normalized_y"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
        std = np.stack([v["normalized_y"].std(0) for v in name2rollout[name].values()], axis=0).mean(0)
        step_metric = np.arange(len(mean))
        axes[0].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
        axes[0].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    axes[1].set_title("Normalized Regret")
    for name in name2rollout:
        mean = np.stack([v["normalized_cumulative_regret"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
        std = np.stack([v["normalized_cumulative_regret"].std(0) for v in name2rollout[name].values()], axis=0).mean(0)
        step_metric = np.arange(len(mean))
        axes[1].plot(step_metric, mean, label=name, alpha=0.9, linewidth=1.5, color=palette[name])
        axes[1].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
    axes[1].legend()
    plt.savefig(os.path.join(output_path, "agg.png"))
    plt.clf()

    
#%% 

def post_init(args):
    args.train_datasets = args.train_datasets[args.eval_id][:30]
    args.test_datasets = args.test_datasets[args.eval_id][:15]
    args.eval_episodes = 20
    args.deterministic_eval = False
    args.problem_cls = {
        "hpob": HPOBMetaProblem, 
        "synthetic": SyntheticMetaProblem,
        "metabo_synthetic": MetaBOSyntheticMetaProblem,
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
        )
        problem_dict[mode] = problem


#%%
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

    
# ckpt_cfgs = {
#     "DT": {
#         "path": [f"log/hpob/dt-max150-input50/5527-seed0-12-04-11-01-850930/ckpt/{e}.ckpt" for e in [1000, 2000, 3000]], 
#         "args": {"input_seq_len": 50, }, 
#         "rollout_args": {"eval_mode": "dynamic", "regret_strategy": "relabel", "init_regret": 20.0}, 
#         "type": "dt"
#     }, 
#     "DT-layer8": {
#         "path": [f"log/hpob/dt-max150-input50-layer8/5527-seed0-12-04-17-42-1353752/ckpt/{e}.ckpt" for e in [1000, 2000, 3000]], 
#         "args": {"input_seq_len": 50, "num_layers": 8}, 
#         "rollout_args": {"eval_mode": "dynamic", "regret_strategy": "relabel", "init_regret": 20.0}, 
#         "type": "dt"
#     }
# }

ckpt_cfgs = dict()
with open(f'scripts/ckpt_configs/{args.problem}/{args.ckpt_id}.yaml', 'r') as f:
    load_dict = yaml.safe_load(f)
    for name in load_dict:
        cfg = dict()
        path = load_dict[name]['path']
        epochs = load_dict[name]['epochs']
        del load_dict[name]['epochs']
        for epoch in epochs:
            curr_path = [
                f'log/{args.problem}/{p}/ckpt/{epoch}.ckpt' for p in path
            ]
            cfg[name + '_' + str(epoch)] = dict(load_dict[name])
            cfg[name + '_' + str(epoch)]['path'] = curr_path
        ckpt_cfgs.update(cfg)
        
print('===== ckpt configs =====')
for k, v in ckpt_cfgs.items():
    print(k, v)
print('========================')

behavior_cfgs = {
    "add_behavior": True, 
    "num": 5, 
    "Random": False, 
    "ShuffledGridSearch": False,
    "CMAES": True, 
    "EagleStrategy": True, 
    "HillClimbing": True, 
    "RegularizedEvolution": True, 
    "BotorchBO": True, 
}

for mode, problem in problem_dict.items():
    ckpts = load_model(problem, ckpt_cfgs)
    if mode == 'train':
        rollout_datasets = args.train_datasets
    elif mode == 'validation':
        rollout_datasets = args.validation_datasets
    else: # test
        rollout_datasets = args.test_datasets
    # rollout_datasets = ["145833", "3891"]
    name2rollout = defaultdict(dict)
    name2rollout.update(add_behavior(behavior_cfgs, problem, rollout_datasets))
    for name in ckpt_cfgs:
        rollout_args = ckpt_cfgs[name]["rollout_args"]
        rollout_res = []
        for designer in ckpts[name]:
            rollout_res.append(
                rollout_designer(problem, designer, rollout_datasets, args.eval_episodes, **rollout_args)
            )
        name2rollout[name] = rollout_res

    # if model_type == 'bc':
        # plot(name2rollout, rollout_datasets, output_path=f"./plot/tune/{args.id}/{dir_name}")
    # elif model_type == 'optformer':
        # plot(name2rollout, rollout_datasets, output_path=f"./plot/tune/{args.id}/{dir_name}-{algo}")
    # elif model_type == 'dt':
        # plot(name2rollout, rollout_datasets, output_path=f"./plot/tune/{args.id}/{dir_name}-{init_regret}-{regret_strategy}-dyna")
    plot(name2rollout, rollout_datasets, output_path=f"./plot/rollout/{args.problem}/{args.ckpt_id}-{args.eval_id}/{mode}/")