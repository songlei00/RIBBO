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
from datasets.datasets import TrajectoryDataset

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
# dataset = filter_designer(dataset)
dataset.set_input_seq_len(args.input_seq_len)

logger.info('dataset length: {}'.format(len(dataset)))
logger.info('x dim: {}'.format(problem.x_dim))
logger.info(problem.id2info)

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
    shuffle=True
)

for i_epoch in trange(args.num_epoch):
    for i_batch, batch in enumerate(trainloader):
        train_metrics = designer.update(batch, clip_grad=args.clip_grad)

    if i_epoch % args.eval_interval == 0:
        for init_regret in args.init_regrets:
            eval_test_metrics, _ = evaluate_decision_transformer_designer(problem, designer, args.test_datasets, args.eval_episodes, init_regret)
            eval_train_metrics, _ = evaluate_decision_transformer_designer(problem, designer, args.train_datasets, args.eval_episodes, init_regret)
            logger.log_scalars(f"eval_trainset_regret={str(init_regret)}", eval_train_metrics, step=i_epoch)
            logger.log_scalars(f"eval_testset_regret={str(init_regret)}", eval_test_metrics, step=i_epoch)

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
    

# final rollout for train datasets
for mode, datasets in zip(["train", "test"], [args.train_datasets, args.test_datasets]):
    for init_regret in args.init_regrets:
        print(f"Evaluating final rollout on {mode} datasets {datasets} with regret {init_regret} ...")
        _, eval_records = evaluate_decision_transformer_designer(problem, designer, datasets, args.eval_episodes, init_regret)
        for key in eval_records:
            id = key.split("_")[-1]
            ys = [y.item() for y in eval_records[key]]
            best_ys = [ys[0]]
            for y in ys[1: ]:
                best_ys.append(max(best_ys[-1], y))

            for i in range(len(ys)):
                logger.log_scalars('rollout_{}_regret={}'.format(mode, str(init_regret)), {'best_y_'+id: best_ys[i], 'y_'+id: ys[i]}, i)