#%%
import os
import copy
import torch
import wandb
import random
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from problems.real_world_problem import RealWorldMetaProblem

from UtilsRL.exp import parse_args, setup

params = {
    'lines.linewidth': 1.5,
    'legend.fontsize': 20,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
matplotlib.rcParams.update(params)

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
    behavior_rollout = defaultdict(partial(defaultdict, partial(defaultdict, list)))
    for l in dataset.trajectory_list:
        if l.metadata["dataset_id"] in tasks and l.metadata["designer"] in algos:
            behavior_rollout[l.metadata["designer"]][l.metadata["dataset_id"]]['y'].append(l.y)
            behavior_rollout[l.metadata["designer"]][l.metadata["dataset_id"]]['X'].append(l.X)
            
    name2rollout = {}
    for a in algos:
        if a not in behavior_rollout:
            print(f'Trajectory for {a} is empty')
            continue
        name2rollout[a] = {}
        for t in tasks:
            name2rollout[a][t] = {}
            X = behavior_rollout[a][t]['X'][:num]
            y = behavior_rollout[a][t]['y'][:num]
            y = torch.stack(y, dim=0)
            # y = y.reshape([5, num, -1]).mean(dim=1) # TODO: align testing
            normalized_y, normalized_regret = problem.get_normalized_y_and_regret(y, id=t)
            name2rollout[a][t]["X"] = np.stack(X, axis=0)
            name2rollout[a][t]["y"] = y.numpy()
            name2rollout[a][t]["normalized_y"] = normalized_y.numpy()
            name2rollout[a][t]["normalized_regret"] = normalized_regret.numpy()
    del behavior_rollout
    return name2rollout          

def plot(name2rollout, datasets, output_path, palette):
    print(f"Saving to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    total_num = len(datasets) + 2 # agg + legend
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
            
    # plot the path for Branin2
    if args.ckpt_id == 'Branin2':
        plt.figure(dpi=300)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5))
        y_min, y_max = 1e9, -1e9
        for name in name2rollout:
            for id in name2rollout[name]:
                y_min_tmp = name2rollout[name][id]['y'].min()
                y_max_tmp = name2rollout[name][id]['y'].max()
                y_min = min(y_min, y_min_tmp)
                y_max = max(y_max, y_max_tmp)

        problem = problem_dict['train']
        x1 = np.linspace(-1, 1, 100)
        x2 = np.linspace(-1, 1, 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1)
        Y = np.zeros_like(X1)
        n = 200
        for i in range(n):
            problem.reset_task(str(i))
            _, info = problem.forward(torch.from_numpy(X))
            Y += info['raw_y'].detach().cpu().numpy().reshape(Y.shape)
        Y /= n
        # N = np.arange((y_max + y_min)/2, y_max, 0.01)
        CS = axes.contour(X1, X2, Y, 100, cmap=mpl.cm.viridis, zorder=axes.get_zorder()-1)
        fig.colorbar(CS)

        for name in name2rollout:
            X_mean = np.stack([v["X"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
            step = 3
            X_mean = X_mean[::step]
            X_std = np.stack([v["X"].std(0) for v in name2rollout[name].values()], axis=0).std(0)
            y_mean = np.stack([v["y"].mean(0) for v in name2rollout[name].values()], axis=0).mean(0)
            y_std = np.stack([v["y"].std(0) for v in name2rollout[name].values()], axis=0).std(0)
            # alpha = (y_mean - y_min) / (y_max - y_min + 1e-6)
            axes.scatter(X_mean[:, 0], X_mean[:, 1], label=name, alpha=0.5, c=palette[name])
            if name.startswith('DT'):
                for s, e in zip(X_mean[:-1], X_mean[1:]):
                    axes.quiver(
                        s[0], s[1], e[0]-s[0], e[1]-s[1], alpha=0.5, 
                        angles='xy', scale_units='xy', scale = 1
                    )
            
        plt.savefig(os.path.join(output_path, "X.pdf"), bbox_inches="tight")
        plt.clf()

    def extract_data(name2rollout, key):
        data = {
            n: {id: v[key] for id, v in r.items()}
            for n, r in name2rollout.items()
        }
        return data

    def plot_all(data, xlabel, ylabel, title, out_name):
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = axes.reshape(-1)
        for idx, id in enumerate(datasets):
            axes[idx].set_title(str(id))
            for name in data:
                mean = data[name][id].mean(0)
                std = data[name][id].std(0)
                step_metric = np.arange(len(mean))
                axes[idx].plot(step_metric, mean, label=name, alpha=0.9, color=palette[name])
                axes[idx].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
        for name in data:
            axes[-2].set_title('agg')
            mean = np.stack([v.mean(0) for v in data[name].values()], axis=0).mean(0)
            std = np.stack([v.std(0) for v in data[name].values()], axis=0).mean(0)
            step_metric = np.arange(len(mean))
            axes[-2].plot(step_metric, mean, label=name, alpha=0.9, color=palette[name])
            axes[-2].fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name])
        for name in data:
            axes[-1].plot([], label=name, color=palette[name])
        axes[-1].legend()
        plt.savefig(os.path.join(output_path, f"{out_name}.pdf"), bbox_inches="tight")
        plt.clf()

    def plot_agg(data, xlabel, ylabel, title, out_name):
        _, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5))
        axes.set_title(title)
        for name in data:
            mean = np.stack([v.mean(0) for v in data[name].values()], axis=0).mean(0)
            std = np.stack([v.std(0) for v in data[name].values()], axis=0).mean(0)
            step_metric = np.arange(len(mean))

            # trunc
            # if args.eval_id == 'GriewankRosenbrock':
            #     threshold = 0.82
            #     step_metric = step_metric[mean > threshold]
            #     std = std[mean > threshold]
            #     mean = mean[mean > threshold]
            #     axes.set_ylim(threshold)

            if name in behavior_cfgs:
                zorder = axes.get_zorder() - 1
            elif name.startswith('DT'):
                zorder = axes.get_zorder() + 1
            else:
                zorder = axes.get_zorder()
            axes.plot(step_metric, mean, label=name, alpha=0.9, color=palette[name], zorder=zorder)
            axes.fill_between(step_metric, mean-std, mean+std, alpha=0.2, color=palette[name], zorder=zorder)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xticks(np.arange(0, args.max_input_seq_len+1, 50))
        axes.set_xticklabels([str(i) for i in np.arange(0, args.max_input_seq_len+1, 50)])
        plt.savefig(os.path.join(output_path, f"{out_name}.pdf"), bbox_inches="tight")
        plt.clf()

    y_data = extract_data(name2rollout, 'y')
    normalized_y_data = extract_data(name2rollout, 'normalized_y')
    normalized_regret_data = extract_data(name2rollout, 'normalized_cumulative_regret')

    # 1. plot y
    plot_all(y_data, None, None, None, 'y')
    
    # 2. plot normalized y
    plot_all(normalized_y_data, None, None, None, 'normalized_y')
    
    # 3. plot regret
    plot_all(normalized_regret_data, None, None, None, 'regret')

    # 4. plot the aggregations
    plot_agg(
        normalized_y_data, 'Number of evaluations', 'Normalized value',
        args.eval_id, 'agg_normalized_y',
    )
 
    # 5. plot the best aggregations
    best_value = dict()
    for name in name2rollout:
        best_value[name] = dict()
        for id in name2rollout[name]:
            best_normalized_y = np.zeros_like(normalized_y_data[name][id])
            for i in range(best_normalized_y.shape[1]):
                best_normalized_y[:, i] = np.max(normalized_y_data[name][id][:, :i+1], axis=1)
            best_value[name][id] = best_normalized_y
    plot_agg(
        best_value, 'Number of evaluations', 'Normalized value',
        args.eval_id, 'agg_best_normalized_y',
    )
            
    # 6. plot the regret
    plot_agg(
        normalized_regret_data, 'Number of evaluations', 'Cumulative regret',
        'Normalized regret', 'agg_regret',
    )

    
#%% 

def post_init(args):
    args.train_datasets = args.train_datasets[args.eval_id][:50]
    args.test_datasets = args.test_datasets[args.eval_id][:15]
    args.eval_episodes = 5
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


def merge_palette(name2rollout, palette):
    COLORS = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
        'purple', 'pink', 'brown', 'orange', 'teal',  'lightblue', 'lime',
        'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold',
        'darkred', 'darkblue'
    ]
    # assign color for unassigned algorithm
    for n in name2rollout.keys():
        if n in palette:
            continue
        # assign unused color
        colors = list(palette.values())
        for c in COLORS:
            if c not in colors:
                palette[n] = c
                break
    return palette

def plot_legend(output_path, palette, n_col=5):
    print(f"Saving legend to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    labels, colors = list(palette.keys()), list(palette.values())
    colors = [c if isinstance(c, str) else tuple([i/255 for i in c]) for c in colors]
    n = len(colors)
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(n)]

    def reorder(h, l, nc):
        new_h, new_l = [], []
        for row_i in range(nc):
            for col_i in range(int(np.ceil(len(h) / nc))):
                idx = col_i * nc + row_i
                if idx < len(h):
                    new_h.append(h[idx])
                    new_l.append(l[idx])
        return new_h, new_l

    handles, labels = reorder(handles, labels, n_col)
    legend = plt.legend(
        handles, labels, loc=3, framealpha=1, frameon=False, 
        ncol=n_col, bbox_to_anchor=(1,1), columnspacing=1
    )
    fig = legend.figure
    fig.canvas.draw()
    expand=[-1, -1, 1, 1]
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(os.path.join(output_path, "legend.pdf"), bbox_inches=bbox)
    
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

palette = {
    # 'BC': 'turquoise',
    'DT': 'blue',
    'BC-filter': 'orange',
    'Opt-Eagle': 'lime',

    # behavior
    # 'Random': 'violet',
    # 'ShuffledGridSearch': 'salmon',
    'CMAES': 'darkviolet',
    'EagleStrategy': 'gold',
    'HillClimbing': 'slategray',
    'RegularizedEvolution': 'teal',
    'BotorchBO': 'brown',
}

for mode, problem in problem_dict.items():
    ckpts = load_model(problem, ckpt_cfgs)
    if mode == 'train':
        rollout_datasets = args.train_datasets
    elif mode == 'validation':
        rollout_datasets = args.validation_datasets
    else: # test
        rollout_datasets = args.test_datasets
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

    output_path=f"./plot/rollout/{args.problem}/{args.ckpt_id}-{args.eval_id}/{mode}/"
    palette = merge_palette(name2rollout, palette)
    plot_legend(output_path, palette)
    plot(
        name2rollout,
        rollout_datasets,
        output_path=output_path,
        palette=palette,
    )