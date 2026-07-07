"""Stage 0: the most naive possible text generation.

An LLM predicts one token at a time. To generate text you loop:

    1. run the model on everything so far -> logits for the next token
    2. pick a token from those logits (here: argmax, "greedy")
    3. append it and go again, until an end-of-sequence token or a length cap

The sin committed here — the reason this file will be rewritten in Stage 1 —
is in step 1: "run the model on *everything so far*". Producing token 500
re-processes the previous 499 tokens from scratch, even though nothing about
them changed. Each step does more work than the last, so generation slows
down as it goes. Run ``scripts/bench_stage0.py`` to watch it happen.
"""

import time
from dataclasses import dataclass

import torch

from inferno.model import Qwen2ForCausalLM


@dataclass
class GenerationResult:
    token_ids: list[int]        # the generated tokens (prompt not included)
    step_seconds: list[float]   # wall-clock time of each decode step


@torch.inference_mode()
def generate(
    model: Qwen2ForCausalLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> GenerationResult:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device)  # (1, seq) — batch of one
    generated: list[int] = []
    step_seconds: list[float] = []

    for _ in range(max_new_tokens):
        start = time.perf_counter()

        # Forward pass over the ENTIRE sequence, prompt + everything
        # generated so far. This line is the villain of Stage 0.
        logits = model(input_ids)  # (1, seq, vocab)

        # Only the last position predicts the *next* token; greedy = argmax.
        next_id = int(logits[0, -1].argmax())

        step_seconds.append(time.perf_counter() - start)
        if next_id in eos_token_ids:
            break
        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    return GenerationResult(token_ids=generated, step_seconds=step_seconds)


def qwen_chat_format(user_message: str) -> str:
    """Wrap a user message in Qwen's chat markup.

    Instruct models are trained on conversations serialized in a specific
    plain-text format; deviate and the model behaves erratically. This is
    Qwen's ChatML flavor. (Real servers render the Jinja chat template that
    ships with the tokenizer; hardcoding it keeps Stage 0 honest.)
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Qwen2.5 stops at <|endoftext|> (151643) or the chat end-of-turn <|im_end|> (151645).
QWEN_EOS_TOKEN_IDS = {151643, 151645}
