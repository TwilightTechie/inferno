# Inferno Roadmap

Inferno is built **one stage at a time**. Every stage follows the same loop:

1. **Feel the problem** — run the previous stage, measure it, watch it hurt.
2. **Understand the idea** — a short doc explaining how real engines (vLLM, Orca, TGI) solve it.
3. **Build it** — the smallest readable implementation of that idea.
4. **Prove it** — a benchmark showing the before/after numbers.

No stage starts until the previous one is working and benchmarked. Each finished
stage is tagged (`stage-0`, `stage-1`, …) so a learner can check out the code at
any point in the story.

## Ground rules

- **Pure PyTorch, readable over fast.** We are learning inference *engineering*
  (memory, scheduling, batching), not kernel authoring. Runs on Apple Silicon
  (MPS) or CPU; no CUDA required.
- **Our own model code.** We load pretrained weights (Qwen2.5-0.5B-Instruct)
  into a transformer we write ourselves. You cannot build a KV cache or paged
  attention on top of someone else's forward pass — owning the attention code
  is the whole point.
- **Docs are first-class.** `docs/` gets a chapter per stage: the problem, the
  idea, the numbers. The repo should teach someone who never opens the code.

## The stages

### Stage 0 — Naive generation: just make a token come out
Load Qwen weights into our own transformer (RMSNorm, RoPE, GQA, SwiGLU),
tokenize a prompt, greedy-decode in a loop. No cache, no batching, nothing.
- **Deliverable:** `generate(prompt)` that produces correct text; verified
  against HuggingFace output.
- **Problem it exposes:** every new token re-runs attention over the *entire*
  sequence. Generation gets slower with every token — O(n²) work for n tokens.

### Stage 1 — KV cache: stop recomputing the past
Split generation into **prefill** (process the prompt once) and **decode**
(feed one token, reuse cached keys/values).
- **Deliverable:** KV cache inside our attention; tokens/sec now flat instead
  of degrading.
- **Problem it exposes:** the cache is huge (do the memory math per token per
  layer), we preallocate for max length, and we still serve **one request at a
  time** while the hardware sits mostly idle — decode is memory-bound.

### Stage 2 — Sampling: real generation controls
Temperature, top-k, top-p, stop tokens, max-tokens, seeds. Small stage, but
every engine needs it and it defines our request schema (`SamplingParams`).
- **Deliverable:** a proper `Request` / `SamplingParams` API.
- **Problem it exposes:** none new — this stage arms us for serving.

### Stage 3 — Static batching: throughput, crudely
Run many prompts in one forward pass: padding, attention masks, batched
sampling.
- **Deliverable:** batch generation; measure throughput vs batch size.
- **Problem it exposes:** ragged lengths mean padding waste; the whole batch
  waits for its longest member (head-of-line blocking); a request arriving
  mid-batch waits for the entire batch to finish.

### Stage 4 — Continuous batching: the heart of vLLM
Iteration-level scheduling (the Orca idea): an engine **step loop** where
requests join the batch the moment they arrive and leave the moment they
finish. Waiting queue → running set → finished.
- **Deliverable:** `Engine.step()`, a request queue, per-request state; latency
  and throughput measured under a simulated arrival stream.
- **Problem it exposes:** KV memory is now the bottleneck. Preallocating
  max-length cache per request caps the batch size at a fraction of what
  actually fits; memory is fragmented and mostly wasted.

### Stage 5 — Paged KV cache: PagedAttention
Virtual memory for the KV cache: fixed-size blocks, a block allocator, a block
table per request, attention that gathers K/V through the table.
- **Deliverable:** block manager + paged attention in PyTorch; demonstrate the
  bigger effective batch size and near-zero waste.
- **Problem it exposes:** we *will* run out of blocks under load — someone has
  to decide who waits, who runs, and who gets evicted.

### Stage 6 — The scheduler: policy under pressure
Admission control against free blocks, preemption (recompute vs. swap),
prefill/decode interleaving, and **chunked prefill** so a long prompt can't
stall everyone's decode.
- **Deliverable:** a scheduler with explicit policies; measure TTFT vs ITL
  trade-offs under load.
- **Problem it exposes:** nobody can use this thing except us, via Python.

### Stage 7 — Serving: an OpenAI-compatible API
Async engine loop, FastAPI server, `/v1/completions` + `/v1/chat/completions`,
SSE token streaming, request cancellation.
- **Deliverable:** `curl` inferno like you'd curl vLLM; point any OpenAI client
  at it.
- **Problem it exposes:** we're flying blind — no numbers under real load.

### Stage 8 — Benchmarks & metrics: measure like an engine
A load generator and the vocabulary of inference performance: **TTFT**,
**ITL/TPOT**, throughput, goodput. Re-run every prior stage's engine against
the same workload and publish the table — the repo's payoff chart.
- **Deliverable:** `benchmarks/` with reproducible numbers across stages.

### Stage 9 — Electives (each independent, pick by interest)
- **Prefix caching** — hash blocks, reuse shared prompt prefixes across requests.
- **Speculative decoding** — small model drafts, big model verifies.
- **Quantization** — int8 weights; memory vs quality.
- **Kernels (conceptual)** — why FlashAttention/CUDA graphs matter; swap our
  attention for `torch.sdpa` and measure.
- **Parallelism (conceptual)** — tensor/pipeline parallel, what changes at 70B.

## Status

| Stage | State |
|---|---|
| 0 — Naive generation | not started |
| 1 — KV cache | not started |
| 2 — Sampling | not started |
| 3 — Static batching | not started |
| 4 — Continuous batching | not started |
| 5 — Paged KV cache | not started |
| 6 — Scheduler | not started |
| 7 — Serving | not started |
| 8 — Benchmarks | not started |
| 9 — Electives | not started |
