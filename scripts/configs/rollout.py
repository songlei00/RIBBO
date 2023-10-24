from UtilsRL.misc.namespace import NameSpace

seed = 0
debug = False
name = "rollout"

id = "6767"

hpob_root_dir = "./data/downloaded_data/hpob/"
data_dir = './data/generated_data/hpob'
cache_dir = "./cache/hpob"

from scripts.configs.dataset_specs import (
    train_datasets, 
    test_datasets, 
)