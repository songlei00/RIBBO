#%%
import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader
from algorithms.designers.dt_designer import DecisionTransformerDesigner, evaluate_decision_transformer_designer
from algorithms.designers.bc_designer import BCTransformerDesigner, evaluate_bc_transformer_designer
from algorithms.modules.dt import DecisionTransformer
from problems.hpob_problem import HPOBMetaProblem

from UtilsRL.exp import parse_args, setup

#%%    
def rollout(problem, designer, datasets, enforce=None, **kwargs):
    def rollout_bc(problem, designer, datasets, enforce=None, **kwargs):
        dataset = problem.get_dataset()
        designer.eval()
        id2y, id2normalized_y = {}, {}
        for id in datasets:
            problem.reset_task(id)
            designer.reset(1)
            last_x, last_y, last_normalized_y = None, None, None
            this_y = np.zeros([problem.seq_len, 1])
            this_normalized_y = np.zeros([problem.seq_len, 1])
            this_normalized_onestep_regret = np.zeros([problem.seq_len, 1])
            if enforce:
                algo = enforce[0]
                trajs = [t for t in dataset.trajectory_list if t.metadata["algorithm"] == algo]
                enforce_traj = np.random.choice(trajs)
            for i in range(problem.seq_len):
                if enforce and i < enforce[1]:
                    last_x = enforce_traj[i]
                else:
                    last_x = designer.suggest(
                        last_x=last_x, 
                        last_y=last_normalized_y, 
                        deterministic=True
                    )
                last_normalized_y, info = problem.forward(last_x)
                last_y = info["raw_y"]
                last_normalized_onestep_regret = info["normalized_onestep_regret"]
                this_y[i] = last_y.detach().cpu().numpy()
                this_normalized_y[i] = last_normalized_y.detach().cpu().numpy()
                this_normalized_onestep_regret[i] = last_normalized_onestep_regret.detach().cpu().numpy()
            id2y[id] = this_y
            id2normalized_y[id] = this_normalized_y
        return id2y, id2normalized_y
    
    def rollout_dt(problem, designer, datasets, enforce=None, init_regret=0.0, **kwargs):
        dataset = problem.get_dataset()
        designer.eval()
        id2y, id2normalized_y = {}, {}
        for id in datasets:
            problem.reset_task(id)
            designer.reset(1, init_regret)
            last_x, last_y, last_normalized_y, last_normalized_regrets = None, None, None, None
            this_y = np.zeros([problem.seq_len, 1])
            this_normalized_y = np.zeros([problem.seq_len, 1])
            this_normalized_onestep_regret = np.zeros([problem.seq_len, 1])
            if enforce:
                algo = enforce[0]
                trajs = [t for t in dataset.trajectory_list if t.metadata["algorithm"] == algo]
                enforce_traj = np.random.choice(trajs)
            for i in range(problem.seq_len):
                if enforce and i < enforce[1]:
                    last_x = enforce_traj[i]
                else:
                    last_x = designer.suggest(
                        last_x=last_x, 
                        last_y=last_normalized_y, 
                        last_regrets=last_normalized_regrets, 
                        determinisitc=True
                    )
                last_normalized_y, info = problem.forward(last_x)
                last_y = info["raw_y"]
                last_normalized_onestep_regret = info["normalized_onestep_regret"]
                last_normalized_regrets = last_normalized_regrets - last_normalized_onestep_regret
                
                this_y[i] = last_y.detach().cpu().numpy()
                this_normalized_y[i] = last_normalized_y.detach().cpu().numpy()
                this_normalized_onestep_regret[i] = last_normalized_onestep_regret.detach().cpu().numpy()
            id2y[id] = this_y
            id2normalized_y[id] = this_normalized_y
        return id2y, id2normalized_y
    
    rollout_fn = rollout_bc if isinstance(designer, BCTransformerDesigner) else rollout_dt
    return rollout_fn(problem, designer, datasets, enforce, **kwargs)

def plot(name2rollout, datasets, output_path):
    total_num = len(datasets)
    _, axes = plt.subplots(nrows=1+(total_num-1)//4, ncols=4)
    axes = axes.reshape(-1)
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            axes[idx].plot(name2rollout[name][0][id].reshape(-1), label=name)
        axes.legend()
    plt.savefig(os.path.join(output_path, "y.png"))
    plt.clf()
    
    for idx, id in enumerate(datasets):
        axes[idx].set_title(str(id))
        for name in name2rollout:
            axes[idx].plot(name2rollout[name][1][id].reshape(-1), label=name)
        axes.legend()
    plt.savefig(os.path.join(output_path, "y_normalized.png"))
    plt.clf()
    

    

#%% 
args = parse_args("scripts/configs/rollout.py")

ckpt_to_visualize = {
    "name": "ckpt_path"
}

problem = HPOBMetaProblem(
    search_space_id=args.id, 
    root_dir=args.hpob_root_dir, 
    data_dir=args.data_dir, 
    cache_dir=args.cache_dir, 
    input_seq_len=300, 
    normalize_method="random", 
    scale_clip_range=None, 
    prioritize=False
)
dataset = problem.get_dataset()

# load ckpts
for name, path in ckpt_to_visualize.items():
    model = torch.load(path, map_location="cpu")
    assert isinstance(model, (BCTransformerDesigner, DecisionTransformerDesigner))
    ckpt_to_visualize[name] = model.cuda()
    
    
#%%
name2rollout = {}
for name, designer in ckpt_to_visualize.items():
    name2rollout[name] = rollout(problem, designer, ["145833"], enforce=None)
