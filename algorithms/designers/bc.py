from typing import Optional, Sequence, Union, Dict, Any

import torch
import numpy as np
import pandas as pd
from operator import itemgetter

from offlinerllib.module.net.attention.base import BaseTransformer
from offlinerllib.module.actor import (
    SquashedDeterministicActor, 
    SquashedGaussianActor, 
    CategoricalActor
)
from algorithms.designers.base import BaseDesigner

class BCTransformerDesigner(BaseDesigner):
    def __init__(
        self, 
        transformer: BaseTransformer, 
        x_dim: int, 
        y_dim: int, 
        embed_dim: int, 
        seq_len: int, 
        x_type: str="deterministic", 
        y_loss_coeff: float=0.0, 
        device: Union[str, torch.device]="cpu", 
        *args, 
        **kwargs
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.x_type = x_type
        self.y_loss_coeff = y_loss_coeff
        
        if x_type == "deterministic":
            self.x_head = SquashedDeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=x_dim, 
            )
        elif x_type == "stochastic":
            self.x_head = SquashedGaussianActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=x_dim, 
                reparameterize=False
            )
        else:
            raise ValueError
        if y_loss_coeff:
            self.y_head = SquashedDeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=y_dim
            )
        
        self.to(device)
        
    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps):
        decay, no_decay = self.transformer.configure_params()
        self.optim = torch.optim.AdamW([
            {"params": [*decay, *self.x_head.parameters()], "weight_decay": weight_decay}, 
            {"params":  no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda step: min((step+1)/warmup_steps, 1))
        
    @torch.no_grad()
    def reset(self):
        self.past_x = torch.zeros([1, self.seq_len, self.x_dim], dtype=torch.float).to(self.device)
        self.past_y = torch.zeros([1, self.seq_len, 1], dtype=torch.float).to(self.device)
        self.past_regrets_to_go = torch.zeros_like([1, self.seq_len, 1], dtype=torch.float).to(self.device)
        self.timesteps = torch.arange(self.seq_len).long().to(self.device).reshape(1, self.seq_len)
        self.step_count = 0
        torch.cuda.empty_cache()
        
    @torch.no_grad()
    def suggest(self, last_x, last_y, determinisitc=False):
        last_x = torch.from_numpy(last_x).float().to(self.device)
        last_y = torch.from_numpy(last_y).float().to(self.device)
        self.past_x[:, self.step_count] = last_x
        self.past_y[:, self.step_count] = last_y
        self.step_count += 1
        
        out = self.transformer(
            x=self.past_x[:, :self.step_count], 
            y=self.past_y[:, :self.step_count], 
            regrets_to_go = self.past_regrets_to_go[:, :self.step_count], 
            timesteps=self.timesteps[:, :self.step_count], 
            attention_mask=None, 
            key_padding_mask=None # during testing all positions are valid
        )
        suggest_x = self.x_head.sample(out[:, 0::3], deterministic=determinisitc)[0]
        return suggest_x[0, self.step_count-1].cpu().numpy()
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        x, y, timesteps, masks = [
            torch.from_numpy(v).float().to(self.device) 
            for v in itemgetter("x", "y", "timesteps", "masks")(batch)
        ]
        B, L, X = x.shape
        
        regrets_to_go = torch.zeros(B, L, 1).float().to(self.device)
        key_padding_mask = ~masks.to(torch.bool)
        out = self.transformer(
            x=x, 
            y=y, 
            regrets_to_go=regrets_to_go, 
            timesteps=timesteps, 
            attention_mask=None, 
            key_padding_mask=key_padding_mask
        )
        # x reconstruction
        if isinstance(self.x_head, SquashedDeterministicActor):
            x_loss = torch.nn.functional.mse_loss(
                self.x_head.sample(out[:, 0::3])[0], 
                x.detach(), 
                reduction="none"
            )
        elif isinstance(self.x_head, SquashedGaussianActor):
            x_loss = self.x_head.evaluate(
                out[:, 0::3], 
                x.detach(), 
            )[0]
        tot_loss = x_loss = (x_loss * masks.unsqueeze(-1)).mean()
        if self.y_loss_coeff:
            y_loss = torch.nn.functional.mse_loss(
                self.y_head.sample(out[:, 1::3])[0], 
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
        
        