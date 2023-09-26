import os
import torch
import wandb
import numpy as np

from tqdm import trange
from torch.utils.data import DataLoader
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from algorithms.designers.bc_designer import BCTransformerDesigner, evaluate_bc_transformer_designer
from algorithms.modules.bc import BCTransformer
from problems.hpob_problem import HPOBMetaProblem
from datasets.datasets import TrajectoryDataset
from algorithms.utils import log_rollout

designers = [
    # 'Random',
    # 'GridSearch',
    # 'ShuffledGridSearch',
    # 'RegularizedEvolution',
    # 'HillClimbing',
    'EagleStrategy',
    # 'Vizier',
    # 'HeBO',
    # 'CMAES',
]

def filter_designer(dataset):
    def filter_fn(trajectory):
        metadata = trajectory.metadata
        return metadata['designer'] in designers
    ret = list(filter(filter_fn, dataset.trajectory_list))
    logger.info('Filter designers')
    return TrajectoryDataset(ret)

def post_init(args):
    args.train_datasets = args.train_datasets[args.id][:5]
    args.test_datasets = args.test_datasets[args.id][:5]

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

transformer = BCTransformer(
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

designer = BCTransformerDesigner(
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
        eval_test_metrics, _ = evaluate_bc_transformer_designer(problem, designer, args.test_datasets, args.eval_episodes)
        eval_train_metrics, _ = evaluate_bc_transformer_designer(problem, designer, args.train_datasets, args.eval_episodes)
        logger.info(f"Epoch {i_epoch}: \n{eval_train_metrics}\n{eval_test_metrics}")
        logger.log_scalars("eval_trainset", eval_train_metrics, step=i_epoch)
        logger.log_scalars("eval_testset", eval_test_metrics, step=i_epoch)
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
    
# final rollout
for mode, datasets in zip(['train', 'test'], [args.train_datasets, args.test_datasets]):
    _, eval_records = evaluate_bc_transformer_designer(problem, designer, datasets, args.eval_episodes)
    log_rollout(logger, 'rollout_{}'.format(mode), eval_records)