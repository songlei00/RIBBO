from UtilsRL.misc.namespace import NameSpace

problem = 'synthetic'
seed = 0
debug = False
name = "rollout"

ckpt_id = "Rastrigin"
eval_id = "Rastrigin"

root_dir = None
data_dir = './data/generated_data/synthetic/'
cache_dir = "./cache/synthetic/"
eval_episodes = 20
max_input_seq_len = 300

class bc_config(NameSpace):
    embed_dim = 256
    num_layers = 12
    num_heads = 8
    attention_dropout = 0.1
    residual_dropout = 0.1
    embed_dropout = 0.1
    pos_encoding = "embed"
    mix_method = "concat"

    input_seq_len = 300
    x_type = "stochastic"
    y_loss_coeff = 0.0
    use_abs_timestep = True

class dt_config(NameSpace):
    embed_dim = 256
    num_layers = 12
    num_heads = 8
    attention_dropout = 0.1
    residual_dropout = 0.1
    embed_dropout = 0.1
    pos_encoding = "embed"
    mix_method = "concat"
    
    input_seq_len = 300
    x_type = "stochastic"
    y_loss_coeff = 0.0
    use_abs_timestep = True

class optformer_config(NameSpace):
    algo_num = 7
    embed_dim = 256
    num_layers = 12
    num_heads = 8
    attention_dropout = 0.1
    residual_dropout = 0.1
    embed_dropout = 0.1
    pos_encoding = "embed"
    mix_method = "concat"

    input_seq_len = 300
    x_type = "stochastic"
    y_loss_coeff = 0.0
    use_abs_timestep = True
    

from scripts.configs.dataset_specs_synthetic import (
    train_datasets, 
    test_datasets, 
    validation_datasets,
)