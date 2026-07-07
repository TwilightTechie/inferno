"""Model configuration, read straight from a HuggingFace ``config.json``.

Every transformer checkpoint ships a ``config.json`` describing its
architecture: how many layers, how wide, how many attention heads, and so on.
We parse just the fields our model code needs instead of hardcoding them, so
the same code can load any Qwen2-family checkpoint (0.5B, 1.5B, 7B, ...).
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int            # number of distinct token ids
    hidden_size: int           # width of the residual stream
    num_hidden_layers: int     # number of transformer blocks
    num_attention_heads: int   # query heads per attention layer
    num_key_value_heads: int   # key/value heads (< query heads => GQA)
    intermediate_size: int     # hidden width of the MLP
    rms_norm_eps: float        # epsilon inside RMSNorm
    rope_theta: float          # base frequency for rotary embeddings
    max_position_embeddings: int  # longest position the model was trained on
    tie_word_embeddings: bool  # reuse the token embedding as the LM head

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        raw = json.loads(path.read_text())
        assert raw["architectures"] == ["Qwen2ForCausalLM"], (
            f"inferno currently implements the Qwen2 architecture, got {raw['architectures']}"
        )
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw["tie_word_embeddings"],
        )
