"""Small decoder-only transformer with RMSNorm, RoPE, GQA and SwiGLU."""
import math
import torch
from torch import nn
from .config import LMLConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0):
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype)[None, None], emb.sin().to(x.dtype)[None, None]


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: LMLConfig) -> None:
        super().__init__()
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(q)
        q = q * cos + _rotate_half(q) * sin
        kcos, ksin = cos, sin
        k = k * kcos + _rotate_half(k) * ksin
        repeats = self.n_heads // self.n_kv
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: LMLConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LMLConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = GroupedQueryAttention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class LMLModel(nn.Module):
    def __init__(self, config: LMLConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.Module()
        self.transformer.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.transformer.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        x = self.transformer.embed_tokens(input_ids)
        for layer in self.transformer.layers:
            x = layer(x)
        return self.lm_head(self.transformer.norm(x))
