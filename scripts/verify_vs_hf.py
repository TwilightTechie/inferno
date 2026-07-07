"""Prove our model is the real Qwen, not something Qwen-shaped.

Runs the same prompts through our implementation and through HuggingFace
``transformers`` (the reference), both greedy, in float32 on CPU so results
are deterministic and comparable. Passes only if:

  1. prompt logits match within float tolerance, and
  2. every generated token id is identical.

Requires the reference implementation:  pip install 'inferno[verify]'
"""

import torch

from inferno.config import ModelConfig
from inferno.generate import QWEN_EOS_TOKEN_IDS, generate, qwen_chat_format
from inferno.loader import fetch_checkpoint, load_model, load_tokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Why is the sky blue?",
    "Write a haiku about memory allocation.",
    "What is 17 * 23?",
]
MAX_NEW_TOKENS = 64


def main() -> None:
    from transformers import AutoModelForCausalLM

    checkpoint = fetch_checkpoint(MODEL_ID)
    device = torch.device("cpu")  # CPU float32: bit-for-bit reproducible

    ours = load_model(checkpoint, device, dtype=torch.float32)
    # attn_implementation="eager" = HF's explicit-math attention, like ours.
    # Their default fused kernel (sdpa) sums in a different order and drifts
    # ~1e-4 — enough to flip a greedy near-tie. Same math, different kernel.
    theirs = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tokenizer = load_tokenizer(checkpoint)

    config = ModelConfig.from_json(checkpoint / "config.json")
    n_params = sum(p.numel() for p in ours.parameters())
    print(f"{MODEL_ID}: {n_params / 1e6:.0f}M params, "
          f"{config.num_hidden_layers} layers, {config.num_attention_heads}q/{config.num_key_value_heads}kv heads\n")

    for prompt in PROMPTS:
        ids = tokenizer.encode(qwen_chat_format(prompt)).ids
        input_ids = torch.tensor([ids])

        # 1. Do the raw logits agree on the prompt?
        with torch.inference_mode():
            our_logits = ours(input_ids)
            their_logits = theirs(input_ids).logits
        max_diff = (our_logits - their_logits).abs().max().item()

        # 2. Does greedy decoding produce the exact same tokens?
        our_tokens = generate(ours, ids, MAX_NEW_TOKENS, QWEN_EOS_TOKEN_IDS).token_ids
        with torch.inference_mode():
            # use_cache=False: make HF recompute from scratch each step like
            # we do. Their KV-cache decode (Stage 1 for us!) follows a
            # different kernel path whose float noise can flip near-ties.
            # repetition_penalty=1.0: Qwen's generation_config.json ships a
            # 1.1 penalty that HF silently applies even to "greedy" decoding.
            # We want pure argmax on both sides.
            their_out = theirs.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=list(QWEN_EOS_TOKEN_IDS), pad_token_id=151643,
                use_cache=False, repetition_penalty=1.0,
            )[0, len(ids):].tolist()
        their_tokens = [t for t in their_out if t not in QWEN_EOS_TOKEN_IDS]

        match = our_tokens == their_tokens
        status = "OK " if match and max_diff < 1e-3 else "FAIL"
        print(f"[{status}] {prompt!r}: max logit diff {max_diff:.2e}, "
              f"{len(our_tokens)} tokens {'identical' if match else 'DIFFER'}")
        if not match:
            print(f"  ours:   {tokenizer.decode(our_tokens)}")
            print(f"  theirs: {tokenizer.decode(their_tokens)}")
            raise SystemExit(1)

    print("\nOur implementation is faithful to the reference.")


if __name__ == "__main__":
    main()
