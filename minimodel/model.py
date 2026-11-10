"""Pre-norm decoder with explicit scaled dot-product causal self-attention."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    expansion: int = 4
    tie_embeddings: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.heads, self.layers, self.expansion) < 1:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.context, cfg.context, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = [a.reshape(b, t, self.heads, c // self.heads).transpose(1, 2) for a in self.qkv(x).chunk(3, dim=-1)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(c // self.heads)
        weights = self.dropout(scores.masked_fill(~self.mask[:t, :t], float("-inf")).softmax(dim=-1))
        return self.dropout(self.proj((weights @ v).transpose(1, 2).contiguous().reshape(b, t, c)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attention = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, cfg.expansion * cfg.width), nn.GELU(), nn.Linear(cfg.expansion * cfg.width, cfg.width), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self.head.weight = self.token.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.cfg.context:
            raise ValueError("input must have shape [batch, time] within context limit")
        x = self.dropout(self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device)))
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, tokens=100, temperature=1.0, top_k=None, generator=None):
        if tokens < 0 or temperature <= 0 or (top_k is not None and top_k < 1):
            raise ValueError("invalid sampling settings")
        was_training = self.training
        self.eval()
        try:
            for _ in range(tokens):
                logits = self(ids[:, -self.cfg.context:])[0][:, -1] / temperature
                if top_k is not None:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                ids = torch.cat((ids, torch.multinomial(logits.softmax(-1), 1, generator=generator)), dim=1)
            return ids
        finally:
            self.train(was_training)
