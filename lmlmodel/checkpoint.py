"""Safe checkpoint I/O. Never silently fall back to unrestricted pickle loading."""
from pathlib import Path
from typing import Any
import torch
from .config import LMLConfig
from .model import LMLModel


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> tuple[LMLModel, dict[str, Any]]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("This PyTorch version lacks safe weights_only loading; upgrade PyTorch.") from exc
    if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
        raise ValueError("checkpoint must contain 'config' and 'state_dict' mappings")
    config = LMLConfig.from_dict(payload["config"])
    model = LMLModel(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload.get("metadata", {})


def save_checkpoint(model: LMLModel, path: str | Path, *, metadata: dict | None = None) -> None:
    payload = {"config": model.config.to_dict(), "state_dict": model.state_dict(),
               "metadata": metadata or {}}
    torch.save(payload, Path(path))
