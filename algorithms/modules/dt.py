from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.gpt2 import GPT2
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding
from offlinerllib.module.net.attention.base import NoDecayParameter

class DecisionTransformer(GPT2):
    def __init__(
        self, 
        x_dim: int, 
        y_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        num_heads: int=1, 
        add_bos: bool=True, 
        mix_method: str="concat", 
        attention_dropout: Optional[float]=0.1, 
        residual_dropout: Optional[float]=0.1, 
        embed_dropout: Optional[float]=0.1, 
        pos_encoding: str="embed", 
    ) -> None:
        super().__init__(
            input_dim=embed_dim,
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
        # we manually do the positional encoding and bos embedding outside
        self.embed_dim = embed_dim
        self.add_bos = add_bos
        if self.add_bos:
            self.bos_x_embed = NoDecayParameter(torch.zeros([1, embed_dim]).float(), requires_grad=True)
            self.bos_y_embed = NoDecayParameter(torch.zeros([1, embed_dim]).float(), requires_grad=True)
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.x_embed = nn.Linear(x_dim, embed_dim)
        self.y_embed = nn.Linear(y_dim, embed_dim)
        self.regret_embed = nn.Linear(1, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        
        # how to mix the inputs
        self.mix_method = mix_method
        if self.mix_method == "concat":
            self.input_proj = nn.Linear(3*embed_dim, embed_dim)
            
    def encode(self, x, y, regrets, timesteps, key_padding_mask):
        B, L, X = x.shape
        x_embedding = self.x_embed(x)
        y_embedding = self.y_embed(y)
        regret_embedding = self.regret_embed(regrets)
        if self.mix_method == "interleave":
            x_embedding = self.pos_encoding(x_embedding, timesteps)
            y_embedding = self.pos_encoding(y_embedding, timesteps)
            regret_embedding = self.pos_encoding(regret_embedding, timesteps)
            inputs = torch.stack([regret_embedding, x_embedding, y_embedding], dim=2).reshape(B, 3*L, self.embed_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.stack([key_padding_mask, key_padding_mask, key_padding_mask], dim=2).reshape(B, 3*L)
        else:
            # align each triplet
            x_embedding = torch.concat([self.bos_x_embed.repeat(B, 1, 1), x_embedding], dim=1)
            y_embedding = torch.concat([self.bos_y_embed.repeat(B, 1 ,1), y_embedding], dim=1)
            regret_embedding = torch.concat([regret_embedding, torch.zeros([B, 1, self.embed_dim]).to(regret_embedding.device)], dim=1)
            if timesteps is not None:
                timesteps = torch.concat([timesteps, timesteps[:, -1:]+1], dim=-1)
            if key_padding_mask is not None:
                key_padding_mask = torch.concat([torch.zeros([B, 1]).bool().to(key_padding_mask.device), key_padding_mask], dim=-1)
            if self.mix_method == "concat":
                inputs = torch.concat([x_embedding, y_embedding, regret_embedding], dim=-1)
                inputs = self.pos_encoding(self.input_proj(torch.nn.functional.relu(inputs)), timesteps)
            elif self.mix_method == "add":
                inputs = x_embedding + y_embedding + regret_embedding
                inputs = self.pos_encoding(inputs, timesteps)
        return inputs, key_padding_mask
        
    def decode(self, out):
        if self.mix_method == "concat":
            out = out[:, :-1]
            return out, out, out
        elif self.mix_method == "interleave":
            return out[:, 0::3], out[:, 1::3], out[:, 2::3]
        elif self.mix_method == "add":
            out = out[:, :-1]
            return out, out, out
            
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
        inputs, key_padding_mask = self.encode(x, y, regrets_to_go, timesteps, key_padding_mask)
        inputs = self.embed_ln(inputs)
        out = super().forward(
            inputs=inputs, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )
        return self.decode(out)

