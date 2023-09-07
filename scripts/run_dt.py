import os
import torch
import numpy as np

from tqdm import trange
from torch.utils.data import DataLoader
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from algorithms.designers.dt_designer import DecisionTransformerDesigner, evaluate_decision_transformer_designer
from algorithms.modules.dt import DecisionTransformer
from problems.hpob_problem import HPOBMetaProblem

def post_init(args):
    args.train_datasets = args.train_datasets[args.id][:5]
    args.test_datasets = args.test_datasets[args.id][:5]
    args.x_dim = args.x_dim[args.id]
    args.y_dim = args.y_dim[args.id]
    args.seq_len = args.seq_len[args.id]

args = parse_args(post_init=post_init)
exp_name = "-".join([args.id, "seed"+str(args.seed)])
logger = CompositeLogger(log_dir=f"./log/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}
}, activate=not args.debug)
logger.log_config(args)
setup(args, logger)

# define the problem and the dataset
problem = HPOBMetaProblem(
    search_space_id=args.id, 
    root_dir=args.hpob_root_dir, 
)
dataset = problem.get_dataset()
logger.info(problem.id2info)

transformer = DecisionTransformer(
    x_dim=problem.x_dim, 
    y_dim=problem.y_dim, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=problem.seq_len, 
    num_heads=args.num_heads, 
    attention_dropout=args.attention_dropout, 
    residual_dropout=args.residual_dropout, 
    embed_dropout=args.embed_dropout, 
    pos_encoding=args.pos_encoding
)

designer = DecisionTransformerDesigner(
    transformer=transformer, 
    x_dim=problem.x_dim, 
    y_dim=problem.y_dim, 
    embed_dim=args.embed_dim, 
    seq_len=args.seq_len, 
    x_type=args.x_type, 
    y_loss_coeff=args.y_loss_coeff, 
    device=args.device
)

designer.configure_optimizers(
    **args.optimizer_args
)

designer.train()

trainloader = DataLoader(
    dataset, 
    batch_size=args.batch_size, 
    pin_memory=True, 
    num_workers=args.num_workers, 
    shuffle=True
)

for i_epoch in trange(args.num_epoch):
    for i_batch, batch in enumerate(trainloader):
        train_metrics = designer.update(batch, clip_grad=args.clip_grad)

    if i_epoch % args.eval_interval == 0:
        eval_test_metrics = evaluate_decision_transformer_designer(problem, designer, args.test_datasets, args.eval_episode)