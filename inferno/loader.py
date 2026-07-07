"""Turn a HuggingFace checkpoint directory into a ready-to-run model.

A checkpoint is just files: ``config.json`` (architecture), one or more
``*.safetensors`` files (a flat dict of tensor name -> tensor), and
``tokenizer.json``. Because our module tree in ``model.py`` mirrors the
checkpoint's naming scheme, loading is a strict ``load_state_dict`` — every
tensor must land somewhere, no silent mismatches.
"""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from inferno.config import ModelConfig
from inferno.model import Qwen2ForCausalLM, rope_frequencies


def fetch_checkpoint(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Path:
    """Download (or reuse from the local HF cache) the checkpoint files."""
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )


def load_model(
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen2ForCausalLM:
    config = ModelConfig.from_json(checkpoint_dir / "config.json")

    # Build the module tree on the "meta" device: shapes only, no memory.
    # The real tensors come from the checkpoint in a moment.
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)

    state_dict = {}
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        state_dict.update(load_file(shard))
    if config.tie_word_embeddings:
        # Tied checkpoints store the embedding once; point lm_head at it.
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    model.load_state_dict(state_dict, strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).eval()
    # The RoPE buffer is computed, not stored in the checkpoint, so the meta-
    # device trick left it empty. Recompute it for real (always float32 —
    # rotation angles need precision).
    model.rope_freqs = rope_frequencies(config, device)
    return model


def load_tokenizer(checkpoint_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
