from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.gpt2 import GPT2
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding

class DecisionTransformer(GPT2):
    def __init__(
        self, 
        x_dim: int, 
        y_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        num_heads: int=1, 
        attention_dropout: Optional[float]=0.1, 
        residual_dropout: Optional[float]=0.1, 
        embed_dropout: Optional[float]=0.1, 
        pos_encoding: str="embed", 
    ) -> None:
        super().__init__(
            input_dim=embed_dim, # actually not used
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            causal=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout, 
            pos_encoding="none", 
            seq_len=0
        )
        # we manually do the positional encoding here
        self.embed_dim = embed_dim
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.x_embed = nn.Linear(x_dim, embed_dim)
        self.y_embed = nn.Linear(y_dim, embed_dim)
        self.regret_embed = nn.Linear(1, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        regrets_to_go: Optional[torch.Tensor]=None, 
        timesteps: Optional[torch.Tensor]=None, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        B, L, X = x.shape
        x_embedding = self.pos_encoding(self.x_embed(x), timesteps)
        y_embedding = self.pos_encoding(self.y_embed(y), timesteps)
        regret_embedding = self.pos_encoding(self.regret_embed(regrets_to_go), timesteps)
        
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask, key_padding_mask], dim=2).reshape(B, 3*L)
        
        stacked_input = torch.stack([regret_embedding, x_embedding, y_embedding], dim=2).reshape(B, 3*L, self.embed_dim)
        stacked_input = self.embed_ln(stacked_input)
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )

        return out    # (batch size, length, action_shape) # out is not projected to action

