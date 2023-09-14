from UtilsRL.misc.namespace import NameSpace

seed = 0
debug = False
name = "bc"

id = "6767"

x_type = "deterministic"
y_loss_coeff = 0.0
mix_method = "concat"

embed_dim = 256
num_layers = 4
num_heads = 4
attention_dropout = 0.1
residual_dropout = 0.1
embed_dropout = 0.1
pos_encoding = "embed"
clip_grad = None
use_abs_timestep = True
input_seq_len = 300

batch_size = 64
num_workers = 4

num_epoch = 1000
eval_interval = 10
log_interval = 1
eval_episodes = 1

hpob_root_dir="./data/downloaded_data/hpob/"

class optimizer_args(NameSpace):
    lr = 1e-4
    weight_decay = 1e-4
    betas = [0.9, 0.999]
    warmup_steps = 10_000
    
class wandb(NameSpace):
    entity = None
    project = None

from scripts.configs.dataset_specs import (
    train_datasets, 
    test_datasets, 
)