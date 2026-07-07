"""A Qwen2-architecture transformer, written to be read.

This is the *whole model*. Everything an LLM does at inference time flows
through the ~200 lines below: embed tokens, run them through a stack of
identical blocks (attention + MLP, each wrapped in a norm and a residual
connection), then project back to vocabulary logits.

Stage 0 deliberately implements attention the naive way: every forward pass
recomputes keys and values for the *entire* sequence. That is the problem the
rest of this repo exists to fix — but you cannot appreciate the fixes until
you have seen (and measured) the naive version.

Module and weight names intentionally mirror the HuggingFace checkpoint
layout (``model.layers.0.self_attn.q_proj.weight`` ...), so loading pretrained
weights is a plain ``load_state_dict``.
"""

import torch
from torch import nn

from inferno.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization (LayerNorm minus the mean-centering).

    Scales each hidden vector to unit RMS, then multiplies by a learned
    per-dimension weight. Cheaper than LayerNorm and just as effective, so
    modern LLMs (Llama, Qwen, Mistral) all use it.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for numerical stability, cast back after.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_frequencies(config: ModelConfig, device: torch.device) -> torch.Tensor:
    """Angular frequency for each pair of head dimensions.

    RoPE treats consecutive dimension pairs of a head as 2D points and rotates
    them by position * frequency. Low dimensions spin fast (capture local
    order), high dimensions spin slowly (capture long-range order).
    """
    exponents = torch.arange(0, config.head_dim, 2, device=device).float() / config.head_dim
    return 1.0 / (config.rope_theta ** exponents)  # (head_dim / 2,)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate query/key vectors by their position.

    Attention scores are dot products q·k. After this rotation, that dot
    product depends on the *distance* between the two positions rather than
    where the pair sits in absolute terms — which is exactly the inductive
    bias language needs. x: (batch, heads, seq, head_dim).
    """
    angles = positions.float()[:, None] * freqs[None, :]     # (seq, head_dim/2)
    cos = angles.cos()[None, None, :, :]                     # broadcast over batch, heads
    sin = angles.sin()[None, None, :, :]
    # Split into the two halves that form the 2D points (GPT-NeoX convention).
    x1, x2 = x.float().chunk(2, dim=-1)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return rotated.to(x.dtype)


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention (GQA).

    GQA: there are fewer key/value heads than query heads (14 query vs 2 KV
    heads in Qwen2.5-0.5B), and each KV head is shared by a *group* of query
    heads. The motivation is purely an inference concern: keys and values are
    what we will later cache per token, and 7x fewer KV heads means a 7x
    smaller cache. Stage 1 makes this concrete.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # Qwen2 uses biases on q/k/v projections but not on the output.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        # Project the residual stream into queries, keys, values,
        # then split the flat projection into per-head vectors:
        # (batch, seq, heads * head_dim) -> (batch, heads, seq, head_dim)
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Encode position by rotating queries and keys (RoPE).
        q = apply_rope(q, positions, freqs)
        k = apply_rope(k, positions, freqs)

        # GQA: replicate each KV head across its group of query heads so the
        # shapes line up. (A fused kernel would index instead of copying.)
        group = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        # The textbook equation: softmax(Q Kᵀ / sqrt(d) + causal mask) V.
        # Written out explicitly on purpose — this repo is about seeing it.
        scores = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)  # (batch, heads, seq, seq)
        # Causal mask: position i may only attend to positions <= i.
        causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        out = weights @ v  # (batch, heads, seq, head_dim)

        # Merge heads back into the residual stream width and project out.
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """Gated feed-forward network (SwiGLU).

    Two parallel projections up to a wider space; one is passed through SiLU
    and acts as a gate on the other; then project back down. This is where
    most of the model's parameters (and most of its "knowledge") live.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """One transformer block: attention and MLP, each pre-normed + residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, freqs)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    """Embedding, the stack of decoder layers, and the final norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, freqs)
        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    """The full causal language model: hidden states -> next-token logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Small models reuse the token embedding matrix as the output
            # projection — same table read in both directions.
            self.lm_head.weight = self.model.embed_tokens.weight
        # RoPE frequencies are fixed (not learned); precompute once.
        self.register_buffer("rope_freqs", rope_frequencies(config, torch.device("cpu")), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq) token ids -> (batch, seq, vocab) logits.

        Note what is *not* here: no cache, no state. Every call recomputes
        attention for every position from scratch. Correct, and wasteful.
        """
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.model(input_ids, positions, self.rope_freqs)
        return self.lm_head(hidden)
