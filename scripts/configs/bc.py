from UtilsRL.misc.namespace import NameSpace

seed = 42
debug = False
name = "bc"

id = "4796"

embed_dim = 256
num_layers = 4
num_heads = 4
attention_dropout = 0.1
residual_dropout = 0.1
embed_dropout = 0.1
pos_encoding = "embed"

batch_size = 64
num_workers = 4

num_epoch = 100
eval_interval = 1
log_interval = 1

class optimizer_args(NameSpace):
    lr = 1e-4
    weight_decay = 1e-4
    betas = [0.9, 0.999]
    warmup_steps = 10_000
    
class wandb(NameSpace):
    entity = None
    project = None