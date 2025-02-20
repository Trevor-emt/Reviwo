import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class SCausalSelfAttention(nn.Module):
    """Spatial Causal Self Attention Layer"""

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_tokens_per_frame = config.n_tokens_per_frame

    def forward(self, x):
        B, T, C = x.size()
        num_frames = T // self.n_tokens_per_frame
        x = x.view(B*num_frames, self.n_tokens_per_frame, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(-1, self.n_tokens_per_frame, self.n_head, C // self.n_head).transpose(1, 2) # (B*num_frames, nh, nt_per_frame, hs)
        q = q.view(-1, self.n_tokens_per_frame, self.n_head, C // self.n_head).transpose(1, 2) # (B*num_frames, nh, nt_per_frame, hs)
        v = v.view(-1, self.n_tokens_per_frame, self.n_head, C // self.n_head).transpose(1, 2) # (B*num_frames, nh, nt_per_frame, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B*num_frames, self.n_tokens_per_frame, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        y = y.view(B, T, C)
        return y
    

class TCausalSelfAttention(nn.Module):
    """Temporal Causal Self Attention Layer"""

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_tokens_per_frame = config.n_tokens_per_frame

    def forward(self, x):
        B, T, C = x.size() # T = num_frames * n_tokens_per_frame
        num_frames = T // self.n_tokens_per_frame
        x = x.view(B, num_frames, self.n_tokens_per_frame, C)
        x = rearrange(x, 'b f t c -> b t f c').contiguous() # (batch, n_tokens, num_frames, hs)
        x = x.view(-1, num_frames, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(-1, num_frames, self.n_head, C // self.n_head).transpose(1, 2) # (batch*nt_per_frame, nh, num_frames, hs)
        q = q.view(-1, num_frames, self.n_head, C // self.n_head).transpose(1, 2) # (batch*nt_per_frame, nh, num_frames, hs)
        v = v.view(-1, num_frames, self.n_head, C // self.n_head).transpose(1, 2) # (batch*nt_per_frame, nh, num_frames, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(-1, num_frames, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y)) # (batch*n_tokens, num_frames, hs)
        y = y.view(B, self.n_tokens_per_frame, num_frames, C) # (batch, n_tokens, num_frames, hs)
        y = rearrange(y, 'b t f c -> b f t c').contiguous() # (batch, num_frames, n_tokens, hs)
        y = y.view(B, T, C)
        return y
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.s_attn = SCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.t_attn = TCausalSelfAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.s_attn(self.ln_1(x))
        x = x + self.t_attn(self.ln_2(x))
        x = x + self.mlp(self.ln_3(x))
        return x
    

class STTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        if config.vocab_size is not None:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Identity()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        device = x.device
        b, t, c = x.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


@dataclass
class STTransConfig:
    block_size: int = 8*8*20
    vocab_size: int = 1024
    n_tokens_per_frame: int = 8*8
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


if __name__ == '__main__':
    st_transformer = STTransformer(config=STTransConfig())
    x = torch.randn(8, 8*8*20, 512)
    print(f"input size: {x.size()}")
    logits = st_transformer(x)
    print(f"output size: {logits.size()}")