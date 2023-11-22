from UtilsRL.misc.namespace import NameSpace

problem = "hpob"
seed = 0
debug = False
name = "optformer"

id = "6767"

x_type = "stochastic"
y_loss_coeff = 0.0
mix_method = "concat"
normalize_method = "random"
augment = False
prioritize = False
prioritize_alpha = 1.0

algo_num = 7
embed_dim = 128
num_layers = 12
num_heads = 4
attention_dropout = 0.1
residual_dropout = 0.1
embed_dropout = 0.1
pos_encoding = "embed"
clip_grad = None
use_abs_timestep = True
input_seq_len = 300
max_input_seq_len = 300
scale_clip_range = None

batch_size = 128
num_workers = 4

num_epoch = 5000
step_per_epoch = 100
eval_interval = 250
log_interval = 1
save_interval = 500
eval_episodes = 5
deterministic_eval = False

root_dir = "./data/downloaded_data/hpob/"
data_dir = './data/generated_data/hpob'
cache_dir = "./cache/hpob/"

class optimizer_args(NameSpace):
    lr = 2e-4
    weight_decay = 1e-2
    betas = [0.9, 0.999]
    warmup_steps = 10_000
    
class wandb(NameSpace):
    entity = "lamda-rl"
    project = "IBO"

from scripts.configs.dataset_specs_hpob import (
    train_datasets, 
    test_datasets, 
)