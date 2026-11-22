"""Inference pipeline with deterministic greedy decoding and sampling."""
import torch
from .model import LMLModel
from .checkpoint import load_checkpoint


class LMLPipeline:
    def __init__(self, model: LMLModel, *, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, path: str, *, device: str | None = None) -> "LMLPipeline":
        model, _ = load_checkpoint(path, map_location="cpu")
        return cls(model, device=device)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], *, max_new_tokens: int = 32,
                 temperature: float = 0.0, top_k: int | None = None,
                 eos_token_id: int | None = None) -> list[int]:
        if not input_ids:
            raise ValueError("input_ids cannot be empty")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be non-negative")
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            context = ids[:, -self.model.config.max_position_embeddings:]
            logits = self.model(context)[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    if top_k <= 0:
                        raise ValueError("top_k must be positive")
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < values[:, -1:]] = -torch.inf
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            if eos_token_id is not None and int(next_id.item()) == eos_token_id:
                break
        return ids[0].tolist()
