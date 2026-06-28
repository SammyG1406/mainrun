"""
    Transformer architecture: RMSNorm, causal self-attention with RoPE, MLP, GPT.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


# Root-mean-square layer normalisation without mean subtraction or bias.
class RMSNorm(nn.Module):
    ## Initialises learned scale weight to ones and stores the stability epsilon.
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    ## Normalises x by its RMS then applies the learned per-dimension scale.
    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight


# Multi-head causal self-attention with fused QKV projection and RoPE encoding.
class CausalSelfAttention(nn.Module):
    ## Builds QKV/output projections and precomputes RoPE cos/sin buffers.
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim   = cfg.d_model // cfg.n_head
        self.n_head     = cfg.n_head
        self.drop_p     = cfg.dropout
        self.qkv        = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj       = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

        half  = self.head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
        pos   = torch.arange(cfg.block_size)
        freqs = torch.outer(pos, freqs)
        self.register_buffer('rope_cos', freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('rope_sin', freqs.sin().unsqueeze(0).unsqueeze(0))

    ## Rotates even/odd head-dimension pairs of x using the precomputed position buffers.
    def _apply_rope(self, x, T):
        cos = self.rope_cos[:, :, :T, :]
        sin = self.rope_sin[:, :, :T, :]
        x_e, x_o = x[..., ::2], x[..., 1::2]
        return torch.stack([x_e * cos - x_o * sin,
                            x_e * sin + x_o * cos], dim=-1).flatten(-2)

    ## Projects to QKV, applies RoPE to queries and keys, runs flash attention, projects output.
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q, k = self._apply_rope(q, T), self._apply_rope(k, T)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# Position-wise feed-forward block: linear → GELU → linear → dropout.
class MLP(nn.Module):
    ## Builds the two-layer sequential feed-forward network.
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    ## Passes x through the feed-forward network.
    def forward(self, x):
        return self.net(x)


# Single transformer block: pre-norm attention and pre-norm MLP, both with residuals.
class Block(nn.Module):
    ## Instantiates the two RMSNorm layers, the attention module, and the MLP.
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.d_model)
        self.ln2  = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

    ## Applies the attention residual sub-layer then the MLP residual sub-layer.
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# Autoregressive decoder-only language model with weight-tied input/output embeddings.
class GPT(nn.Module):
    ## Builds token embedding, transformer stack, final norm, and output head.
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg       = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = RMSNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('net.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.head.weight = self.token_emb.weight

    ## Initialises linear and embedding weights as N(0, 0.02) and zeros all biases.
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    ## Embeds tokens, runs all transformer blocks, returns logits and optional CE loss.
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean'
        )
        return logits, loss
