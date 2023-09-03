import os
import torch
import numpy as np

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from algorithms.designers.bc import BCTransformerDesigner
from algorithms.modules.dt import DecisionTransformer

args = parse_args()
exp_name = "-".join([args.name, "seed"+str(args.seed)])
logger = CompositeLogger(log_dir="./log", name=exp_name, logger_config={
    "TensorboardLogger": {}
}, activate=not args.debug)
setup(args, logger)
