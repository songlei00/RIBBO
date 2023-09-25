from UtilsRL.misc.namespace import NameSpace

seed = 0
debug = False
name = "dt"

id = "6767"

x_type = "stochastic"
y_loss_coeff = 0.0
mix_method = "concat"
normalize_method = "random"

embed_dim = 256
num_layers = 4
num_heads = 4
attention_dropout = 0.1
residual_dropout = 0.1
embed_dropout = 0.1
pos_encoding = "sinusoidal"
clip_grad = None
use_abs_timestep = True
input_seq_len = 300

batch_size = 256
num_workers = 4

num_epoch = 1000
eval_interval = 20
log_interval = 1
eval_episodes = 1

init_regrets = [0] 
scale_clip_range = [0.3, 2.0]

hpob_root_dir = "./data/downloaded_data/hpob/"
data_dir = './data/generated_data/hpob'
cache_dir = "./cache/hpob"

class optimizer_args(NameSpace):
    lr = 2e-4
    weight_decay = 1e-4
    betas = [0.9, 0.999]
    warmup_steps = 10_000
    
class wandb(NameSpace):
    entity = "lamda-rl"
    project = "IBO"

from scripts.configs.dataset_specs import (
    train_datasets, 
    test_datasets, 
)