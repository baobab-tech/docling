# Docling VLM Benchmark on Apple Silicon

## TL;DR — Fastest approach

```bash
# 1. Install vllm-mlx
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# 2. Start the batched inference server (patches granite-docling compatibility issues)
/path/to/vllm-mlx/bin/python start_server.py

# 3. Send pages concurrently (in another terminal)
uv run --with openai python test_concurrent.py
```

This runs [GraniteDocling-258M](https://huggingface.co/ibm-granite/granite-docling-258M-mlx) via [vllm-mlx](https://github.com/waybarrios/vllm-mlx) with continuous batching. On an M5 Max it processes an 8-page PDF in **15s (0.52 pages/sec, 758 tok/s)** — 2x faster than sequential docling.

---

Benchmarking [Docling](https://github.com/docling-project/docling) VLM-based PDF conversion on Apple Silicon (M5 Max, 40 GPU cores, 128GB unified memory).

## Key Finding: Batched Inference via vllm-mlx

Sequential VLM inference is memory-bandwidth-bound — the 258M model is tiny but autoregressive decoding reads all weights per token. **Batching multiple pages** amortizes this cost, reading weights once to generate tokens for all pages simultaneously.

### Results (8-page arXiv PDF)

| Approach | Wall clock | Pages/sec | Aggregate tok/s |
|---|---|---|---|
| Docling + GraniteDocling MLX (sequential) | 36s | 0.22 | ~345 |
| Docling + SmolDocling MLX (sequential) | 41s | 0.20 | ~345 |
| mlx-vlm direct (sequential, warm model) | 28s | 0.28 | ~430 |
| **vllm-mlx batched (8 concurrent)** | **15s** | **0.52** | **758** |

### Why Sequential is Slow

- The GPU is **not** compute-bound — a 258M model barely uses the 40 GPU cores
- The bottleneck is **memory bandwidth**: each token reads ~516MB of weights
- M5 Max has ~546 GB/s bandwidth → theoretical max ~1,058 tok/s single-stream
- Achieved ~430 tok/s = ~40% utilization (typical with vision encoder overhead)
- Multiple processes don't help — they share the same memory bandwidth

### Why Batching Helps

- Batch N pages = read weights **once**, generate **N tokens** per forward pass
- With 8 concurrent pages: 758 tok/s aggregate (1.76x single-stream)
- More pages in flight = more throughput (up to hardware limits)

## Scripts

### `run.py` — Docling + GraniteDocling MLX benchmark
Standard docling VLM pipeline with per-page timing.

### `run_smoldocling.py` — Docling + SmolDocling MLX benchmark
Same as above with SmolDocling-256M model.

### `start_server.py` — vllm-mlx server with GraniteDocling
Patches the Idefics3Processor to expose the chat template (required for vllm-mlx compatibility) and raises the prefill token limit. Starts a continuous-batching server on port 8000.

```bash
# Start the server (requires vllm-mlx tool installed)
/Users/olivier/.local/share/uv/tools/vllm-mlx/bin/python start_server.py
```

### `test_concurrent.py` — Concurrent page benchmark
Sends all 8 pages simultaneously to the vllm-mlx server and measures aggregate throughput.

```bash
uv run --with openai python test_concurrent.py
```

### `run_batched.py` — Raw MLX batched experiment
Exploratory script testing interleaved KV-cache generation (slower than vllm-mlx due to no true batched forward pass).

### `run_parallel.py` — Multiprocessing experiment
Spawns multiple MLX processes — doesn't help because they share GPU memory bandwidth.

## vllm-mlx Patches Required for GraniteDocling

Two issues prevent vllm-mlx from serving `ibm-granite/granite-docling-258M-mlx` out of the box. Both are patched in `start_server.py`.

### 1. Idefics3Processor missing chat template

vllm-mlx calls `processor.apply_chat_template()` for multimodal models, but the Idefics3Processor doesn't expose the chat template even though the underlying tokenizer has one. This causes:

```
ValueError: Cannot use apply_chat_template because this processor does not have a chat template.
```

**Fix:** Monkey-patch `mlx_vlm.load` to copy `tokenizer.chat_template` onto the processor after loading:

```python
processor.chat_template = processor.tokenizer.chat_template
```

This is a known issue in HuggingFace transformers ([#40913](https://github.com/huggingface/transformers/issues/40913)) where processor `chat_template` kwargs get overridden by model defaults during `from_pretrained`.

### 2. Prefill token limit too low

The MLLM scheduler defaults to `prefill_step_size=1024`, but GraniteDocling prompts are ~1142 tokens (image patches + text), causing:

```
Total prompt tokens (1142) exceeds safe limit (1024)
```

**Fix:** Monkey-patch `MLLMSchedulerConfig.__init__` to set `prefill_step_size=4096`.

## Setup

```bash
# Install vllm-mlx
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# Run docling benchmarks
uv run run.py
uv run run_smoldocling.py

# Run batched benchmark
/Users/olivier/.local/share/uv/tools/vllm-mlx/bin/python start_server.py  # terminal 1
uv run --with openai python test_concurrent.py                             # terminal 2
```

## Models

- **[ibm-granite/granite-docling-258M-mlx](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)** — Idefics3 architecture, 258M params, DocTags output format
- **[docling-project/SmolDocling-256M-preview-mlx-bf16](https://huggingface.co/docling-project/SmolDocling-256M-preview-mlx-bf16)** — SmolVLM architecture, 256M params, DocTags output format

## Hardware

- Apple M5 Max — 40 GPU cores, 18 CPU cores, 128GB unified memory
- macOS Darwin 25.3.0
