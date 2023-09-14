from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.gpt2 import GPT2
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding
from offlinerllib.module.net.attention.base import NoDecayParameter
import UtilsRL.exp as exp

def get_vector_statistics(v):
    func_dict = {
        'min': torch.min,
        'max': torch.max,
        'mean': torch.mean,
        'median': torch.median,
        'norm': lambda x: torch.norm(x, p=1) / len(x.flatten()),
    }
    ret = dict()
    for key, func in func_dict.items():
        ret[key] = func(v)
    return ret

global_step = 0

class BCTransformer(GPT2):
    def __init__(
        self, 
        x_dim: int, 
        y_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        num_heads: int=1, 
        add_bos: bool=True, 
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
        self.add_bos = add_bos
        if self.add_bos:
            self.bos_embed = NoDecayParameter(torch.zeros([1, embed_dim]).float(), requires_grad=True)
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.x_embed = nn.Linear(x_dim, embed_dim)
        self.y_embed = nn.Linear(y_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        timesteps: Optional[torch.Tensor]=None, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        B, L, X = x.shape
        x_embedding_before = self.x_embed(x)
        y_embedding_before = self.y_embed(y)
        x_embedding = self.pos_encoding(x_embedding_before, timesteps)
        y_embedding = self.pos_encoding(y_embedding_before, timesteps)

        # log
        # x_pos_embedding = x_embedding - x_embedding_before
        # y_pos_embedding = y_embedding - y_embedding_before

        # for key, embedding in zip(
        #     ['x_pos', 'y_pos', 'x', 'y'],
        #     [x_pos_embedding, y_pos_embedding, x_embedding, y_embedding]
        # ):
        #     exp.logger.log_scalers(key, get_vector_statistics(embedding), step=global_step)
        # global_step += 1

        # add up
        x_embedding = x_embedding + y_embedding
        y_embedding = torch.zeros_like(y_embedding)
        
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask], dim=2).reshape(B, 2*L)
        stacked_input = torch.stack([x_embedding, y_embedding], dim=2).reshape(B, 2*L, self.embed_dim)
        if self.add_bos:
            if key_padding_mask is not None:
                key_padding_mask = torch.concat([
                    torch.ones([B, 1]).to(stacked_input.device), 
                    key_padding_mask
                ], dim=1)
            stacked_input = torch.concat([
                self.bos_embed.repeat(B, 1, 1), 
                stacked_input
            ], dim=1)
        stacked_input = self.embed_ln(stacked_input)
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )

        return out    # (batch size, length, action_shape) # out is not projected to action

