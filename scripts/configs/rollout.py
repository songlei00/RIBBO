from UtilsRL.misc.namespace import NameSpace

seed = 0
debug = False
name = "rollout"

id = "6767"

hpob_root_dir = "./data/downloaded_data/hpob/"
data_dir = './data/generated_data/hpob'
cache_dir = "./cache/hpob"

class bc_config(NameSpace):
    embed_dim = 256
    num_layers = 4
    num_heads = 4
    attention_dropout = 0.1
    residual_dropout = 0.1
    embed_dropout = 0.1
    pos_encoding = "sinusoidal"
    mix_method = "concat"

    input_seq_len = 300
    x_type = "stochastic"
    y_loss_coeff = 0.0
    use_abs_timestep = True

class dt_config(NameSpace):
    embed_dim = 256
    num_layers = 4
    num_heads = 4
    attention_dropout = 0.1
    residual_dropout = 0.1
    embed_dropout = 0.1
    pos_encoding = "sinusoidal"
    mix_method = "concat"
    
    input_seq_len = 300
    x_type = "stochastic"
    y_loss_coeff = 0.0
    use_abs_timestep = True


from scripts.configs.dataset_specs import (
    train_datasets, 
    test_datasets, 
)