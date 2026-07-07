# inferno

An LLM inference engine built from scratch to teach how vLLM works — KV cache,
continuous batching, PagedAttention — in readable PyTorch.

Most inference engines are optimized for production; this one is optimized for
**understanding**. It is built in stages, and every stage exists to fix a
problem you can *measure* in the previous one. Start at Stage 0 and read
forward: see [ROADMAP.md](ROADMAP.md) for the full arc, and `docs/` for one
chapter per stage.

Runs a real model (Qwen2.5-0.5B-Instruct) on ordinary hardware — Apple
Silicon or CPU, no GPU required — and is verified token-for-token against
HuggingFace `transformers`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/generate.py "Why is the sky blue?"   # downloads the model (~1 GB) on first run
```

## The story so far

| Stage | Chapter | The problem it solves |
|---|---|---|
| 0 | [Naive generation](docs/stage-00-naive-generation.md) | making a correct token come out at all |

Each completed stage is tagged (`stage-0`, `stage-1`, ...), so you can check
out the code exactly as it was at any point in the story.
