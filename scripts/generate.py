"""Generate text with the Stage 0 engine.

    python scripts/generate.py "Why is the sky blue?"
"""

import argparse

from inferno.generate import QWEN_EOS_TOKEN_IDS, generate, qwen_chat_format
from inferno.loader import best_device, fetch_checkpoint, load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="user message to send to the model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--raw", action="store_true", help="skip the chat template (plain completion)")
    args = parser.parse_args()

    checkpoint = fetch_checkpoint()
    device = best_device()
    print(f"loading model on {device} ...")
    model = load_model(checkpoint, device)
    tokenizer = load_tokenizer(checkpoint)

    text = args.prompt if args.raw else qwen_chat_format(args.prompt)
    prompt_ids = tokenizer.encode(text).ids

    result = generate(model, prompt_ids, args.max_new_tokens, QWEN_EOS_TOKEN_IDS)

    print(tokenizer.decode(result.token_ids))
    total = sum(result.step_seconds)
    print(
        f"\n[{len(result.token_ids)} tokens in {total:.1f}s "
        f"= {len(result.token_ids) / total:.1f} tok/s | "
        f"first step {result.step_seconds[0] * 1000:.0f}ms, "
        f"last step {result.step_seconds[-1] * 1000:.0f}ms]"
    )


if __name__ == "__main__":
    main()
