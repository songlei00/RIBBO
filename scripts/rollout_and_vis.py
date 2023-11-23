#%%
import os
import copy
import torch
import wandb
import random
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

from UtilsRL.exp import parse_args, setup

#%%    

def rollout_designer(problem, designer, datasets, eval_episode, deterministic_eval, **kwargs):
    dataset = problem.get_dataset()
    if isinstance(designer, BCTransformerDesigner):
        metrics, record = evaluate_bc_transformer_designer(
            problem=problem, 
            designer=designer, 
            datasets=datasets, 
            eval_episode=eval_episode, 
            deterministic_eval=deterministic_eval, 
        )
    elif isinstance(designer, OptFormerDesigner):
        metrics, record = evaluate_optformer_designer(
            problem=problem, 
            designer=designer, 
            datasets=datasets, 
            eval_episode=eval_episode, 
            deterministic_eval=deterministic_eval, 
            algo=kwargs.get("algo")
        )
    elif isinstance(designer, DecisionTransformerDesigner):
        metrics, record = evaluate_decision_transformer_designer(
            problem=problem, 
            designer=designer, 
            datasets=datasets, 
            eval_episode=eval_episode, 
            deterministic_eval=deterministic_eval, 
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
        name2rollout[a] = {}
        for t in tasks:
            name2rollout[a][t] = {}
            y = sum(behavior_rollout[a][t][:num])/num
            normalized_y, normalized_regret = problem.get_normalized_y_and_regret(y, id=t)
            name2rollout[a][t]["y"] = y.numpy()
            name2rollout[a][t]["normalized_y"] = normalized_y.numpy()
            name2rollout[a][t]["normalized_regret"] = normalized_regret.numpy()

    return name2rollout          

def plot(name2rollout, datasets, output_path):
    os.makedirs(output_path, exist_ok=True)
    total_num = len(datasets) + 2
    nrows, ncols = 1+(total_num-1)//4, 4

    # 1. plot y
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.reshape(-1)
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            axes[idx].plot(name2rollout[name][id]["y"].reshape(-1), label=name, alpha=0.6, linewidth=1.5)
        # axes[idx].legend()
    for name in name2rollout:
        axes[-2].set_title('agg')
        mean = np.mean(np.array([
            data["y"].reshape(-1) for data in name2rollout[name].values()
        ]), axis=0)
        axes[-2].plot(mean, label=name, alpha=0.6, linewidth=1.5)
    for name in name2rollout:
        axes[-1].plot([], label=name)
    axes[-1].legend()
    plt.savefig(os.path.join(output_path, "y.png"))
    plt.clf()
    
    # 2. plot normalized y
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.reshape(-1)   
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            axes[idx].plot(name2rollout[name][id]["normalized_y"].reshape(-1), label=name, alpha=0.6, linewidth=1.5)
        # axes[idx].legend()
    for name in name2rollout:
        axes[-2].set_title('agg')
        mean = np.mean(np.array([
            data["normalized_y"].reshape(-1) for data in name2rollout[name].values()
        ]), axis=0)
        axes[-2].plot(mean, label=name, alpha=0.6, linewidth=1.5)
    for name in name2rollout:
        axes[-1].plot([], label=name)
    axes[-1].legend()
    plt.savefig(os.path.join(output_path, "y_normalized.png"))
    plt.clf()
    
    # 3. plot regret
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.reshape(-1)   
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            data = name2rollout[name][id]["normalized_regret"].reshape(-1)
            data = np.flip(np.flip(data, 0).cumsum(), 0)
            axes[idx].plot(data, label=name, alpha=0.6, linewidth=1.5)
    for name in name2rollout:
        axes[-2].set_title('agg')
        data = np.mean(np.array([
            data["normalized_y"].reshape(-1) for data in name2rollout[name].values()
        ]), axis=0)
        data = np.flip(np.flip(data, 0).cumsum(), 0)
        axes[-2].plot(data, label=name, alpha=0.6, linewidth=1.5)
    for name in name2rollout:
        axes[-1].plot([], label=name)
    axes[-1].legend()
    plt.savefig(os.path.join(output_path, "regret.png"))
    plt.clf()

    
#%% 

def post_init(args):
    args.train_datasets = args.train_datasets[args.id][:15]
    args.test_datasets = args.test_datasets[args.id][:15]
    args.eval_episodes = 20
    args.deterministic_eval = False
    args.problem_cls = {
        "hpob": HPOBMetaProblem, 
        "synthetic": SyntheticMetaProblem
    }.get(args.problem)
    
args = parse_args("scripts/configs/rollout/hpob.py", post_init=post_init)
# args = parse_args("scripts/configs/rollout/synthetic.py")
setup(args, _seed=0)

# define the problem and the dataset
problem = args.problem_cls(
    search_space_id=args.id, 
    root_dir=args.root_dir, 
    data_dir=args.data_dir,
    cache_dir=args.cache_dir, 
    input_seq_len=1, # does not matter here
    max_input_seq_len=args.max_input_seq_len,
    normalize_method='random', # does not matter here 
    scale_clip_range=None, # does not matter here
    prioritize=False, # does not matter here
)


#%%
def load_model(ckpt_cfgs):
    ckpts = {}
    for name, config in ckpt_cfgs.items():
        if config["type"] == "bc":
            new_args = copy.deepcopy(args.bc_config)
            for k in config["args"]:
                new_args[k] = config["args"][k]
            transformer = BCTransformer(problem.x_dim, problem.y_dim, new_args.embed_dim, new_args.num_layers, problem.seq_len, new_args.num_heads, True, new_args.mix_method, new_args.attention_dropout, new_args.residual_dropout, new_args.embed_dropout, new_args.pos_encoding)
            designer = BCTransformerDesigner(transformer, problem.x_dim, problem.y_dim, new_args.embed_dim, problem.seq_len, new_args.input_seq_len, new_args.x_type, new_args.y_loss_coeff, new_args.use_abs_timestep, "cpu")
        elif config["type"] == "dt":
            new_args = copy.deepcopy(args.dt_config)
            for k in config["args"]:
                new_args[k] = config["args"][k]
            transformer = DecisionTransformer(problem.x_dim, problem.y_dim, new_args.embed_dim, new_args.num_layers, problem.seq_len, new_args.num_heads, True, new_args.mix_method, new_args.attention_dropout, new_args.residual_dropout, new_args.embed_dropout, new_args.pos_encoding)
            designer = DecisionTransformerDesigner(transformer, problem.x_dim, problem.y_dim, new_args.embed_dim, problem.seq_len, new_args.input_seq_len, new_args.x_type, new_args.y_loss_coeff, new_args.use_abs_timestep, "cpu")
        elif config["type"] == "optformer":
            new_args = copy.deepcopy(args.optformer_config)
            for k in config["args"]:
                new_args[k] = config["args"][k]
            transformer = OptFormerTransformer(problem.x_dim, problem.y_dim, new_args.embed_dim, new_args.num_layers, problem.seq_len, new_args.num_heads, new_args.algo_num, new_args.mix_method, new_args.attention_dropout, new_args.residual_dropout, new_args.embed_dropout, new_args.pos_encoding)
            designer =OptFormerDesigner(transformer, problem.x_dim, problem.y_dim, new_args.embed_dim, problem.seq_len, new_args.input_seq_len, new_args.x_type, new_args.y_loss_coeff, new_args.use_abs_timestep, "cpu")
        designer.load_state_dict(torch.load(config["path"], map_location="cpu"), strict=True)
        designer.to(args.device)
        ckpts[name] = designer
    return ckpts
    

dir_name = 'dt-embed128-layer12-embed-len50-maxlen100'
wandb_name = '7609-seed0-11-21-15-58-2592675'
model_type = 'dt'

ckpt_dir = 'log/{}/{}/ckpt/'.format(dir_name, wandb_name)
ckpt_cfgs = dict()
for epoch in range(1000, 4001, 1000):
    ckpt_cfgs['default-step{}'.format(epoch)] = {
        'path': ckpt_dir + '{}.ckpt'.format(epoch),
        'args': {'input_seq_len': 50},
        'type': model_type,
    }
behavior_cfgs = {
    "add_behavior": True, 
    "num": 20, 
    "CMAES": True, 
    "EagleStrategy": True, 
    "HeBO": False, 
    "HillClimbing": True, 
    "Random": True, 
    "RegularizedEvolution": True, 
    "ShuffledGridSearch": False
}

ckpts = load_model(ckpt_cfgs)
rollout_datasets = args.train_datasets
# rollout_datasets = ["145833", "3891"]
name2rollout = defaultdict(dict)
for name, designer in ckpts.items():
    algo = "HillClimbing"
    init_regret = 0.0
    regret_strategy = "relabel"
    name2rollout[name] = rollout_designer(
        problem=problem, 
        designer=designer, 
        datasets=rollout_datasets, 
        eval_episode=args.eval_episodes, 
        deterministic_eval=False, 
        algo=algo, 
        init_regret=init_regret, 
        regret_strategy=regret_strategy
    )
name2rollout.update(add_behavior(behavior_cfgs, problem, rollout_datasets))

if model_type == 'bc':
    plot(name2rollout, rollout_datasets, output_path=f"./plot/rollout/{args.id}/{dir_name}")
elif model_type == 'optformer':
    plot(name2rollout, rollout_datasets, output_path=f"./plot/rollout/{args.id}/{dir_name}-{algo}")
elif model_type == 'dt':
    plot(name2rollout, rollout_datasets, output_path=f"./plot/rollout/{args.id}/{dir_name}-{init_regret}-clip")
