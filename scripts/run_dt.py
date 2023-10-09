import os
import torch
import wandb
import numpy as np

from tqdm import trange
from torch.utils.data import DataLoader
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from algorithms.designers.dt_designer import DecisionTransformerDesigner, evaluate_decision_transformer_designer
from algorithms.modules.dt import DecisionTransformer
from problems.hpob_problem import HPOBMetaProblem
from datasets.datasets import TrajectoryDataset
from algorithms.utils import log_rollout

def post_init(args):
    args.train_datasets = args.train_datasets[args.id][:15]
    args.test_datasets = args.test_datasets[args.id][:15]

args = parse_args(post_init=post_init)
exp_name = "-".join([args.id, "seed"+str(args.seed)])
logger = CompositeLogger(log_dir=f"./log/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
}, activate=not args.debug)
logger.log_config(args)
setup(args, logger)

# define the problem and the dataset
problem = HPOBMetaProblem(
    search_space_id=args.id, 
    root_dir=args.hpob_root_dir, 
    data_dir=args.data_dir,
    cache_dir=args.cache_dir, 
    input_seq_len=args.input_seq_len, 
    normalize_method=args.normalize_method, 
    scale_clip_range=args.scale_clip_range, 
    prioritize=args.prioritize, 
    prioritize_alpha=args.prioritize_alpha, 
)
dataset = problem.get_dataset()

logger.info('dataset length: {}'.format(len(dataset)))
logger.info('x dim: {}'.format(problem.x_dim))

transformer = DecisionTransformer(
    x_dim=problem.x_dim, 
    y_dim=problem.y_dim, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=problem.seq_len, 
    num_heads=args.num_heads, 
    add_bos=True, 
    mix_method=args.mix_method, 
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
    seq_len=problem.seq_len, 
    input_seq_len=args.input_seq_len, 
    x_type=args.x_type, 
    y_loss_coeff=args.y_loss_coeff, 
    use_abs_timestep=args.use_abs_timestep, 
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
    # shuffle=True
)
train_iter = iter(trainloader)

for i_epoch in trange(args.num_epoch):
    for i_batch in range(args.step_per_epoch):
        batch = next(train_iter)
    # for i_batch, batch in enumerate(trainloader):
        train_metrics = designer.update(batch, clip_grad=args.clip_grad)
    if i_epoch % args.eval_interval == 0:
        for init_regret in args.init_regrets:
            eval_test_metrics, _ = evaluate_decision_transformer_designer(problem, designer, args.test_datasets, args.eval_episodes, init_regret)
            eval_train_metrics, _ = evaluate_decision_transformer_designer(problem, designer, args.train_datasets, args.eval_episodes, init_regret)
            logger.log_scalars(f"eval_trainset_regret={str(init_regret)}", eval_train_metrics, step=i_epoch)
            logger.log_scalars(f"eval_testset_regret={str(init_regret)}", eval_test_metrics, step=i_epoch)

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
    

# final rollout
for mode, datasets in zip(["train", "test"], [args.train_datasets, args.test_datasets]):
    for init_regret in args.init_regrets:
        print(f"Evaluating final rollout on {mode} datasets {datasets} with regret {init_regret} ...")
        _, eval_records = evaluate_decision_transformer_designer(problem, designer, datasets, args.eval_episodes, init_regret)
        log_rollout(logger, 'rollout_{}_regret={}'.format(mode, init_regret), eval_records)