from typing import Optional, Sequence, Union, Dict, Any

import torch
import numpy as np
import pandas as pd
from operator import itemgetter

from offlinerllib.module.net.attention.base import BaseTransformer
from offlinerllib.module.actor import (
    SquashedDeterministicActor, 
    SquashedGaussianActor, 
    DeterministicActor
)
from algorithms.designers.base import BaseDesigner
from algorithms.modules.optformer import OptFormerTransformer
from algorithms.utils import calculate_metrics
from algorithms.optim.scheduler import LinearWarmupCosineAnnealingLR

class OptFormerDesigner(BaseDesigner):
    def __init__(
        self, 
        transformer: BaseTransformer, 
        x_dim: int, 
        y_dim: int, 
        embed_dim: int, 
        seq_len: int, 
        input_seq_len: int, 
        x_type: str="deterministic", 
        y_loss_coeff: float=0.0, 
        use_abs_timestep: bool=False, 
        device: Union[str, torch.device]="cpu", 
        *args, 
        **kwargs
    ) -> None:
        super().__init__()
        assert isinstance(transformer, OptFormerTransformer)
        self.transformer = transformer
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.input_seq_len = input_seq_len
        self.x_type = x_type
        self.y_loss_coeff = y_loss_coeff
        self.use_abs_timestep = use_abs_timestep
        
        if x_type == "deterministic":
            self.x_head = SquashedDeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=x_dim, 
                hidden_dims=[embed_dim, ]
            )
        elif x_type == "stochastic":
            self.x_head = SquashedGaussianActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=x_dim, 
                reparameterize=False, 
                conditioned_logstd=True, 
                logstd_min=-5, 
                hidden_dims=[embed_dim, ]
            )
        else:
            raise ValueError
        if y_loss_coeff:
            # this is because y is not scaled into any range
            self.y_head = DeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=y_dim, 
                hidden_dims=[embed_dim, ]
            )
        
        self.to(device)
        
    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps, max_steps):
        decay, no_decay = self.transformer.configure_params()
        decay_parameters = [*decay, *self.x_head.parameters()]
        if self.y_loss_coeff:
            decay_parameters.extend(self.y_head.parameters())
        self.optim = torch.optim.AdamW([
            {"params": decay_parameters, "weight_decay": weight_decay}, 
            {"params":  no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.optim_scheduler = LinearWarmupCosineAnnealingLR(
            self.optim, 
            warmup_epochs=warmup_steps, 
            max_epochs=max_steps
        )
        self.total_parameters = [*decay, *no_decay, *self.x_head.parameters()]
        if self.y_loss_coeff:
            self.total_parameters.extend(self.y_head.parameters())
        
    @torch.no_grad()
    def reset(self, algo: Union[str, int], eval_num=1):
        if isinstance(algo, str):
            algo = {
                "ShuffledGridSearch": 0, 
                "RegularizedEvolution": 1, 
                "Random": 2, 
                "HillClimbing": 3, 
                "HeBO": 4, 
                "EagleStrategy": 5, 
                "CMAES": 6
            }.get(algo)
        self.cur_algo = algo * torch.ones([eval_num, 1], dtype=torch.int32).to(self.device)
        self.past_x = torch.zeros([eval_num, self.seq_len, self.x_dim], dtype=torch.float).to(self.device)
        self.past_y = torch.zeros([eval_num, self.seq_len, 1], dtype=torch.float).to(self.device)
        self.timesteps = torch.arange(self.seq_len).long().repeat(eval_num, 1).to(self.device)
        self.step_count = 0
        # torch.cuda.empty_cache()
        
    @torch.no_grad() # need to adapt for seq len
    def suggest(
        self, 
        last_x=None, 
        last_y=None, 
        deterministic=False, 
        *args, **kwargs
    ):
        if last_x is not None and last_y is not None:
            last_x = torch.as_tensor(last_x).float().to(self.device)
            last_y = torch.as_tensor(last_y).float().to(self.device)
            self.past_x[:, self.step_count] = last_x
            self.past_y[:, self.step_count] = last_y
            self.step_count += 1
        
        x_pred, y_pred = self.transformer(
            x=self.past_x[:, :self.step_count][:, -self.input_seq_len:], 
            y=self.past_y[:, :self.step_count][:, -self.input_seq_len:], 
            algo=self.cur_algo, 
            timesteps=self.timesteps[:, :self.step_count][:, -self.input_seq_len:] if self.use_abs_timestep else None, 
            attention_mask=None, 
            key_padding_mask=None # during testing all positions are valid
        )
        suggest_x = self.x_head.sample(
            x_pred[:, -1], deterministic=deterministic
        )[0]
        return suggest_x
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        x, y, algo, timesteps, masks = [
            torch.as_tensor(v).to(self.device) 
            for v in itemgetter("x", "y", "algo", "timesteps", "masks")(batch)
        ]
        B, L, X = x.shape
        
        key_padding_mask = ~masks.to(torch.bool)
        x_pred, y_pred = self.transformer(
            x=x, 
            y=y, 
            algo=algo, 
            timesteps=timesteps if self.use_abs_timestep else None, 
            attention_mask=None, 
            key_padding_mask=key_padding_mask
        )
        x_pred = x_pred[:, :-1] # discard the last prediction
        y_pred = y_pred[:, :-1]
        # x reconstruction
        if isinstance(self.x_head, SquashedDeterministicActor):
            x_loss = torch.nn.functional.mse_loss(
                self.x_head.sample(x_pred)[0], 
                x.detach(), 
                reduction="none"
            )
        elif isinstance(self.x_head, SquashedGaussianActor):
            x_loss = - self.x_head.evaluate(
                x_pred, 
                x.detach(), 
            )[0]
        tot_loss = x_loss = (x_loss * masks.unsqueeze(-1)).mean()
        # y reconstruction
        if self.y_loss_coeff:
            y_loss = torch.nn.functional.mse_loss(
                self.y_head.sample(y_pred)[0], 
                y.detach(), 
                reduction="none"
            )
            y_loss = (y_loss * masks.unsqueeze(-1)).mean()
            tot_loss += self.y_loss_coeff * y_loss
        else:
            y_loss = torch.tensor(0.0)
        self.optim.zero_grad()
        tot_loss.backward()
        grad_norm = torch.cat([
            param.grad.detach().flatten()
            for param in self.total_parameters
            if param.grad is not None
        ]).norm()
        bos_norm = self.transformer.bos_embed.weight.grad.norm()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.total_parameters, clip_grad)
        self.optim.step()
        self.optim_scheduler.step()
        return {
            "loss/x_loss": x_loss.item(), 
            "loss/y_loss": y_loss.item(), 
            "loss/tot_loss": tot_loss.item(),
            "loss/learning_rate": self.optim_scheduler.get_last_lr()[0], 
            "loss/grad_norm": grad_norm.item(), 
            "loss/optformer_bos_norm": bos_norm.item()
        }

        
@torch.no_grad()
def evaluate_optformer_designer(
    problem, 
    designer: OptFormerDesigner, 
    datasets, 
    eval_episode, 
    deterministic_eval, 
    algo
):
    print(f"evaluating on {datasets} ...")
    designer.eval()
    id2y, id2normalized_y, id2normalized_onestep_regret = {}, {}, {}
    for id in datasets:
        problem.reset_task(id)
        designer.reset(algo, eval_episode)
        last_x, last_y, last_normalized_y = None, None, None
        this_y = np.zeros([eval_episode, problem.seq_len, 1])
        this_normalized_y = np.zeros([eval_episode, problem.seq_len, 1])
        this_normalized_onestep_regret = np.zeros([eval_episode, problem.seq_len, 1])
        for i in range(problem.seq_len):
            last_x = designer.suggest(
                last_x=last_x, 
                last_y=last_normalized_y, 
                deterministic=deterministic_eval
            )
            last_normalized_y, info = problem.forward(last_x)
            last_y = info["raw_y"]
            last_normalized_onestep_regret = info["normalized_onestep_regret"]

            this_y[:, i] = last_y.detach().cpu().numpy()
            this_normalized_y[:, i] = last_normalized_y.detach().cpu().numpy()
            this_normalized_onestep_regret[:, i] = last_normalized_onestep_regret.detach().cpu().numpy()
            
        id2y[id] = this_y
        id2normalized_y[id] = this_normalized_y
        id2normalized_onestep_regret[id] = this_normalized_onestep_regret
    
    metrics, trajectory_record = calculate_metrics(id2y, id2normalized_y, id2normalized_onestep_regret)
    
    designer.train()
    torch.cuda.empty_cache()
    return metrics, trajectory_record