"""Validated model configuration."""
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class LMLConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    intermediate_size: int = 512
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        positive = ("vocab_size", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads",
                    "intermediate_size", "max_position_embeddings")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token_id = getattr(self, name)
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{name} must be within the vocabulary")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LMLConfig":
        aliases = {"n_layers": "num_hidden_layers", "num_kv_heads": "num_key_value_heads",
                   "max_seq_len": "max_position_embeddings"}
        normalized = {aliases.get(k, k): v for k, v in values.items()}
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in normalized.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
