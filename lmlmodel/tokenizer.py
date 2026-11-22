"""Explicit vocabulary tokenizer; no implicit byte/BPE assumptions."""
from pathlib import Path
import json


class VocabularyTokenizer:
    def __init__(self, tokens: list[str], *, unk_token: str = "<unk>") -> None:
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError("tokens must be a non-empty list of unique strings")
        self.tokens = tokens
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.unk_token = unk_token
        if unk_token not in self.token_to_id:
            raise ValueError("unk_token must exist in vocabulary")

    @classmethod
    def from_file(cls, path: str | Path) -> "VocabularyTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokens = data["tokens"] if isinstance(data, dict) else data
        return cls(tokens)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(piece, self.token_to_id[self.unk_token])
                for piece in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.tokens[i] if 0 <= i < len(self.tokens) else self.unk_token
                        for i in ids)
