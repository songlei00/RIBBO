from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.gpt2 import GPT2
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding
from offlinerllib.module.net.attention.base import NoDecayParameter
from offlinerllib.module.net.mlp import MLP


class OptFormerTransformer(GPT2):
    def __init__(
        self, 
        x_dim: int, 
        y_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        num_heads: int=1, 
        algo_num: int=3, 
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
        self.bos_embed = torch.nn.Embedding(num_embeddings=algo_num, embedding_dim=embed_dim)
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.x_embed = nn.Linear(x_dim, embed_dim)
        self.y_embed = nn.Linear(y_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        
        self.mix_method = mix_method
        if self.mix_method == "concat":
            self.input_proj = MLP(input_dim=2*embed_dim, hidden_dims=[embed_dim, embed_dim, ])
        
    def encode(self, x, y, algo, timesteps, key_padding_mask):
        B, L, X = x.shape
        x_embedding = self.x_embed(x)
        y_embedding = self.y_embed(y)
        if self.mix_method == "concat": 
            inputs = torch.concat([x_embedding, y_embedding], dim=-1)
            inputs = self.pos_encoding(self.input_proj(inputs), timesteps)
        elif self.mix_method == "interleave":
            x_embedding = self.pos_encoding(x_embedding, timesteps)
            y_embedding = self.pos_encoding(y_embedding, timesteps)
            inputs = torch.stack([x_embedding, y_embedding], dim=2).reshape(B, 2*L, self.embed_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.stack([key_padding_mask, key_padding_mask], dim=2).reshape(B, 2*L)
        elif self.mix_method == "add":
            inputs = x_embedding + y_embedding
            inputs = self.pos_encoding(inputs, timesteps)
        # add algo identity
        bos = self.bos_embed(algo)
        inputs = torch.concat([
            bos, 
            inputs
        ], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.concat([
                torch.zeros([B, 1]).bool().to(inputs.device), 
                key_padding_mask
            ], dim=1)
        return inputs, key_padding_mask
    
    def decode(self, out):
        if self.mix_method == "concat":
            return out, out
        elif self.mix_method == "interleave":
            # this is for padding the length of y
            return out[:, 0::2], torch.concat([out[:, 1::2], torch.zeros_like(out[:, -1:]).to(out.device)], dim=1)
        elif self.mix_method == "add":
            return out, out
        
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        algo: torch.Tensor, 
        timesteps: Optional[torch.Tensor]=None, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        B, L, X = x.shape
        inputs, key_padding_mask = self.encode(x, y, algo, timesteps, key_padding_mask)
        inputs = self.embed_ln(inputs)
        out = super().forward(
            inputs=inputs, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )
        return self.decode(out)
        
        
        