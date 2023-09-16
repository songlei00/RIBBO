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
from algorithms.modules.bc import BCTransformer

class BCTransformerDesigner(BaseDesigner):
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
        assert isinstance(transformer, BCTransformer)
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
        
    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps):
        decay, no_decay = self.transformer.configure_params()
        decay_parameters = [*decay, *self.x_head.parameters()]
        if self.y_loss_coeff:
            decay_parameters.extend([*self.y_head.parameters()])
        self.optim = torch.optim.AdamW([
            {"params": decay_parameters, "weight_decay": weight_decay}, 
            {"params":  no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda step: min((step+1)/warmup_steps, 1))
        
    @torch.no_grad()
    def reset(self, eval_num=1):
        self.past_x = torch.zeros([eval_num, self.seq_len, self.x_dim], dtype=torch.float).to(self.device)
        self.past_y = torch.zeros([eval_num, self.seq_len, 1], dtype=torch.float).to(self.device)
        self.timesteps = torch.arange(self.seq_len).long().repeat(eval_num, 1).to(self.device)
        self.step_count = 0
        torch.cuda.empty_cache()
        
    @torch.no_grad() # need to adapt for seq len
    def suggest(
        self, 
        last_x=None, 
        last_y=None, 
        determinisitc=False, 
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
            timesteps=self.timesteps[:, :self.step_count][:, -self.input_seq_len:] if self.use_abs_timestep else None, 
            attention_mask=None, 
            key_padding_mask=None # during testing all positions are valid
        )
        suggest_x = self.x_head.sample(
            x_pred[:, -1], deterministic=determinisitc
        )[0]
        return suggest_x
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        x, y, timesteps, masks = [
            torch.as_tensor(v).to(self.device) 
            for v in itemgetter("x", "y", "timesteps", "masks")(batch)
        ]
        B, L, X = x.shape
        
        key_padding_mask = ~masks.to(torch.bool)
        x_pred, y_pred = self.transformer(
            x=x, 
            y=y, 
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
        if clip_grad is not None:
            raise NotImplementedError
        self.optim.step()
        self.optim_scheduler.step()
        return {
            "loss/x_loss": x_loss.item(), 
            "loss/y_loss": y_loss.item(), 
            "loss/tot_loss": tot_loss.item(),
            "misc/learning_rate": self.optim_scheduler.get_last_lr()[0]
        }

        
@torch.no_grad()
def evaluate_bc_transformer_designer(problem, designer: BCTransformerDesigner, datasets, eval_episode):
    print(f"evaluating on {datasets} ...")
    designer.eval()
    all_id_y = {}
    for id in datasets:
        problem.reset_task(id)
        designer.reset(eval_episode)
        last_x, last_y = None, None
        this_y = np.zeros([eval_episode, problem.seq_len, 1])
        for i in range(problem.seq_len):
            last_x = designer.suggest(
                last_x=last_x, 
                last_y=last_y, 
                determinisitc=True
            )
            last_y = problem.forward(last_x)
            this_y[:, i] = last_y.detach().cpu().numpy()
        all_id_y[id] = this_y
        
    metrics = {}
    
    # best y: max over sequence, average over eval num
    best_y_sum = 0
    for id in all_id_y:
        best_y_this = all_id_y[id].max(axis=1).mean()
        metrics["best_y_"+id] = best_y_this
        best_y_sum += best_y_this
    metrics["best_y_agg"] = best_y_sum / len(all_id_y)

    # regret: (best_y - y), sum over sequence, average over eval num
    regret_sum = 0
    for id in all_id_y:
        regret_this = (problem.best_y - all_id_y[id]).sum(axis=1).mean()
        metrics["regret_"+id] = regret_this
        regret_sum += regret_this
    metrics["regret_agg"] = regret_sum / len(all_id_y)

    trajectory_record = {}
    # mean y
    for id in all_id_y:
        mean_y = all_id_y[id].mean(axis=0)
        trajectory_record['mean_y_' + id] = mean_y
    
    designer.train()
    return metrics, trajectory_record