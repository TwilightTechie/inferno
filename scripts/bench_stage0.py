"""Measure Stage 0's fatal flaw: generation slows down as it goes.

Generates a few hundred tokens greedily and reports the wall-clock time per
decode step, bucketed. Because each step re-runs the model over the entire
sequence, step time grows with sequence length — total time for n tokens is
O(n²). The numbers this prints motivate Stage 1 (the KV cache).
"""

import torch

from inferno.generate import generate, qwen_chat_format
from inferno.loader import best_device, fetch_checkpoint, load_model, load_tokenizer

MAX_NEW_TOKENS = 400
CHUNK = 25
BUCKET = 50


def main() -> None:
    checkpoint = fetch_checkpoint()
    device = best_device()
    print(f"device: {device}")
    model = load_model(checkpoint, device)
    tokenizer = load_tokenizer(checkpoint)

    prompt = qwen_chat_format(
        "Tell me a very long, detailed story about a dragon who learns to code."
    )
    prompt_ids = tokenizer.encode(prompt).ids

    # Warm up once so lazy kernel compilation doesn't pollute the numbers.
    generate(model, prompt_ids, 8, eos_token_ids=set())

    # We generate in chunks purely so we can clear the allocator cache
    # between them (naive generation is stateless, so chunking doesn't change
    # what is computed). Why: every step allocates tensors at a *new*
    # sequence length, and PyTorch's caching allocator keeps a cached buffer
    # for every shape it has seen. Left alone, that hoard trips OS memory
    # pressure and injects multi-second stalls mid-run (try it: remove the
    # empty_cache). Even our measurement problems are memory problems —
    # remember this feeling when we reach PagedAttention in Stage 5.
    ids = list(prompt_ids)
    step_seconds: list[float] = []
    while len(step_seconds) < MAX_NEW_TOKENS:
        if device.type == "mps":
            torch.mps.empty_cache()
        # eos disabled: we want a fixed-length run for clean measurements.
        result = generate(model, ids, CHUNK, eos_token_ids=set())
        ids += result.token_ids
        step_seconds += result.step_seconds

    n = len(step_seconds)
    print(f"\nprompt {len(prompt_ids)} tokens, generated {n} tokens")
    print(f"\n{'tokens':>12} | {'ms/token':>9} | {'tok/s':>6}")
    print("-" * 34)
    for i in range(0, n, BUCKET):
        bucket = step_seconds[i : i + BUCKET]
        avg = sum(bucket) / len(bucket)
        print(f"{i:>5} - {i + len(bucket):>4} | {avg * 1000:>9.1f} | {1 / avg:>6.1f}")

    total = sum(step_seconds)
    print("-" * 34)
    print(f"total: {total:.1f}s for {n} tokens ({n / total:.1f} tok/s average)")
    first = sum(step_seconds[:BUCKET]) / BUCKET
    last = sum(step_seconds[-BUCKET:]) / BUCKET
    print(f"\nthe last {BUCKET} tokens were {last / first:.1f}x slower than the first {BUCKET}.")
    print("nothing about the early tokens changed — we just keep recomputing them.")
    print("fix: Stage 1, the KV cache.")


if __name__ == "__main__":
    main()
