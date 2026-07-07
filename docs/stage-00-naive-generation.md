# Stage 0 — Naive generation: just make a token come out

**Code:** [`inferno/model.py`](../inferno/model.py) · [`inferno/generate.py`](../inferno/generate.py) · [`inferno/loader.py`](../inferno/loader.py)
**Run it:**

```bash
python scripts/generate.py "Why is the sky blue?"   # generate text
python scripts/verify_vs_hf.py                      # prove correctness
python scripts/bench_stage0.py                      # feel the problem
```

## What an inference engine actually does

Strip away every optimization and an LLM inference engine is a `for` loop:

```
tokens = tokenize(prompt)
loop:
    logits = model(tokens)          # run the transformer
    next  = pick(logits[last])      # choose the next token
    tokens.append(next)             # and go again
```

The model is a *next-token predictor*: given a sequence, it outputs a score
(logit) for every entry in its 151,936-token vocabulary. Generation is just
asking it repeatedly. Everything vLLM does — KV caching, continuous batching,
PagedAttention — exists to make this loop fast and cheap. In Stage 0 we run
the loop with no tricks at all, so that every later stage has a baseline to
beat and a problem to solve.

## The model, in one page

We implement the Qwen2 architecture ourselves ([`model.py`](../inferno/model.py),
~200 lines) and load the pretrained weights of Qwen2.5-0.5B-Instruct into it.
Why write it ourselves? Because every technique in this repo — the KV cache,
paged attention — lives *inside* the attention code. You can't rearrange the
internals of a model you use as a black box.

A decoder-only transformer is an embedding, a stack of identical blocks, and
an output projection:

```
token ids ──► embed_tokens ──► [ DecoderLayer × 24 ] ──► norm ──► lm_head ──► logits
                                       │
                    x = x + Attention(RMSNorm(x))     ← talk to earlier tokens
                    x = x + MLP(RMSNorm(x))           ← think about what you heard
```

The pieces, and why each one is there:

- **RMSNorm** — keeps activations well-scaled before each sub-block. LayerNorm
  minus the mean-subtraction; what modern LLMs use.
- **RoPE (rotary position embedding)** — the attention math is permutation-
  blind, so position must be injected somehow. RoPE *rotates* each query/key
  vector by an angle proportional to its position; dot products then depend on
  the relative distance between tokens.
- **Grouped-query attention (GQA)** — 14 query heads share just 2 key/value
  heads. This exists purely for *inference*: K and V are what we will cache
  per token in Stage 1, and 7× fewer KV heads is a 7× smaller cache. The
  architecture was shaped by the serving problem before we ever got here.
- **SwiGLU MLP** — the feed-forward block, holding most of the parameters.
- **Weight tying** — at 0.5B, the embedding table doubles as the output
  projection, saving 136M parameters.

Attention itself is written as the textbook equation — an explicit
`softmax(QKᵀ/√d + causal mask)V` — because this repo is about seeing it.
The causal mask enforces the autoregressive contract: position *i* may look
at positions *≤ i* only.

Loading is unglamorous and worth demystifying: a checkpoint is a `config.json`
(architecture hyperparameters) plus `.safetensors` files (a flat dict of
tensor-name → tensor). We name our modules to match the checkpoint's names, so
loading is one strict `load_state_dict` — every tensor must land somewhere.

## Trust, but verify

A model can be *almost* right — one wrong sign in RoPE, one swapped norm —
and still produce fluent text. Fluency proves nothing. The only acceptable
evidence is agreement with the reference implementation, so
[`verify_vs_hf.py`](../scripts/verify_vs_hf.py) checks, in float32 on CPU:

1. **Prompt logits** match HuggingFace `transformers` exactly (max diff 0.00), and
2. **greedy generation** produces the identical token ids, on every test prompt.

Getting to "identical" surfaced two lessons that generalize:

- **Same math ≠ same floats.** Against HF's default fused attention kernel
  (`sdpa`) our logits differed by ~1e-4 — not a bug, just a different
  floating-point summation order. Against their `eager` implementation (the
  same explicit math we wrote): zero difference. When two greedy candidates
  are nearly tied, 1e-4 of kernel noise flips the token and the texts diverge
  wildly from that point. Determinism in LLMs is a *kernel-level* property.
- **Know what your baseline is actually doing.** Even bit-identical logits
  produced different tokens at first, because Qwen's `generation_config.json`
  ships `repetition_penalty: 1.1` and HF applies it silently even to "greedy"
  decoding. Neutralize it and the outputs match token-for-token.

## The numbers — and the problem

[`bench_stage0.py`](../scripts/bench_stage0.py) generates 400 tokens on an
Apple M-series (MPS, bfloat16) and times every step:

| tokens generated | ms/token | tok/s |
|---:|---:|---:|
| 0 – 50 | 59.1 | 16.9 |
| 50 – 100 | 82.6 | 12.1 |
| 100 – 150 | 102.4 | 9.8 |
| 150 – 200 | 123.2 | 8.1 |
| 200 – 250 | 151.7 | 6.6 |
| 250 – 300 | 164.3 | 6.1 |
| 300 – 350 | 192.8 | 5.2 |
| 350 – 400 | 221.8 | 4.5 |

**The last 50 tokens are 3.8× slower than the first 50**, and it only gets
worse: step time grows linearly with sequence length, so total time for *n*
tokens grows as *n²*. At 400 tokens we average 7.3 tok/s; the model is capable
of far more.

The waste is easy to locate. To produce token 401, we feed all 435 previous
tokens through the model and compute hidden states, keys, and values for every
one of them — then throw everything away except the logits of the *last*
position. Those 434 earlier tokens have not changed. Their keys and values
have not changed. We computed them last step, and the step before that, and
we will compute them again next step.

(A smaller instance of the same disease, visible in the benchmark script: the
loop allocates tensors at a new sequence length every step, and PyTorch's
caching allocator hoards a buffer for every shape it has seen — enough to trip
OS memory pressure mid-run. Inference is a memory-management problem wearing a
compute costume. This theme returns in Stage 5.)

## Next

The keys and values of past tokens are the only thing attention needs from
the past — and they never change once computed. So: compute them once, keep
them, and feed the model *one token at a time*.

That is the **KV cache** — Stage 1.
