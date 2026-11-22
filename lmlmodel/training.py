"""Minimal supervised next-token training primitives."""
import torch
from torch import nn
from .model import LMLModel


def causal_lm_loss(model: LMLModel, input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must be [batch, sequence] with sequence >= 2")
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        targets.reshape(-1))


def train_step(model: LMLModel, optimizer: torch.optim.Optimizer,
               input_ids: torch.Tensor, *, max_grad_norm: float = 1.0) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = causal_lm_loss(model, input_ids)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu())
