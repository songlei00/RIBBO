import os
import torch
import numpy as np

from tqdm import trange
from torch.utils.data import DataLoader
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from algorithms.designers.bc import BCTransformerDesigner
from algorithms.modules.dt import DecisionTransformer
from datasets.load_datasets import load_hpob_dataset
from problems.hpob_problem import HPOBProblem

args = parse_args()
exp_name = "-".join([args.id, "seed"+str(args.seed)])
logger = CompositeLogger(log_dir=f"./log/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}
}, activate=not args.debug)
setup(args, logger)

# define the problem and the dataset
dataset = load_hpob_dataset(args.id)
problem = HPOBProblem()
x_dim, seq_len, y_dim = problem.dim, problem.seq_len, 1

transformer = DecisionTransformer(
    x_dim=x_dim, 
    y_dim=y_dim, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=seq_len, 
    num_heads=args.num_heads, 
    attention_dropout=args.attention_dropout, 
    residual_dropout=args.residual_dropout, 
    embed_dropout=args.embed_dropout, 
    pos_encoding=args.pos_encoding
)

designer = BCTransformerDesigner(
    transformer=transformer, 
    x_dim=x_dim, 
    y_dim=y_dim, 
    embed_dim=args.embed_dim, 
    seq_len=seq_len, 
    x_type=args.x_type, 
    y_loss_coeff=args.y_loss_coeff, 
    device=args.device
).to(args.device)

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
        eval_metrics = problem.evaluate(designer)
        logger.info(f"Epoch {i_epoch}: \n{eval_metrics}")
        logger.log_scalars("eval", eval_metrics, step=i_epoch)
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
    
