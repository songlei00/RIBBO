from UtilsRL.misc.namespace import NameSpace

problem = 'synthetic'
seed = 0
debug = False
name = "dt"

id = "Rastrigin"

x_type = "stochastic"
y_loss_coeff = 0.0
mix_method = "concat"
normalize_method = "random"
augment = False
prioritize = False
prioritize_alpha = 1.0

embed_dim = 128
num_layers = 12
num_heads = 4
attention_dropout = 0.1
residual_dropout = 0.1
embed_dropout = 0.1
pos_encoding = "embed"
clip_grad = None
use_abs_timestep = True
input_seq_len = 100

batch_size = 128
num_workers = 4

num_epoch = 1000
step_per_epoch = 100
eval_interval = 50
log_interval = 1
save_interval = 500
eval_episodes = 5
deterministic_eval = True

init_regrets = [0, 50, 100] 
scale_clip_range = None

root_dir = None
data_dir = './data/generated_data/synthetic'
cache_dir = "./cache/synthetic"

class optimizer_args(NameSpace):
    lr = 2e-4
    weight_decay = 1e-2
    betas = [0.9, 0.999]
    warmup_steps = 10_000
    
class wandb(NameSpace):
    entity = "lamda-rl"
    project = "IBO-benchmark"

from scripts.configs.dataset_specs_synthetic import (
    train_datasets, 
    test_datasets, 
)
