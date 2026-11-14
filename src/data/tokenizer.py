import os
import pickle
from typing import List, Dict

class CharTokenizer:
    """Character-level tokenizer for the Shakespeare corpus."""

    def __init__(self, stoi: Dict[str, int], itos: Dict[int, str]):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.itos[int(i)] for i in token_ids])

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        meta = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
        }
        with open(filepath, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        with open(filepath, "rb") as f:
            meta = pickle.load(f)
        return cls(stoi=meta["stoi"], itos=meta["itos"])
