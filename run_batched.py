# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "mlx-vlm",
#     "pypdfium2",
#     "pillow",
# ]
# ///

"""Batched VLM inference — process multiple pages simultaneously."""

import os
import tempfile
import time
import urllib.request

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.generate import prepare_inputs, generate_step
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
import pypdfium2 as pdfium

# Download PDF
pdf_url = "https://arxiv.org/pdf/2501.17887"
pdf_path = os.path.join(tempfile.gettempdir(), "benchmark.pdf")
if not os.path.exists(pdf_path):
    urllib.request.urlretrieve(pdf_url, pdf_path)

# Rasterize pages
pdf = pdfium.PdfDocument(pdf_path)
images = []
for i in range(len(pdf)):
    page = pdf[i]
    bitmap = page.render(scale=2.0)
    img = bitmap.to_pil()
    images.append(img)
print(f"Rasterized {len(images)} pages")

# Load model once
model_path = "ibm-granite/granite-docling-258M-mlx"
t_load = time.time()
model, processor = load(model_path)
config = load_config(model_path)
t_load = time.time() - t_load
print(f"Model loaded in {t_load:.2f}s")

prompt = "Convert this page to docling."
formatted = apply_chat_template(processor, config, prompt, num_images=1)

# Sequential baseline first
print("\n--- Sequential (1 page at a time) ---")
t0 = time.time()
seq_results = []
for i, img in enumerate(images):
    tmp = f"/tmp/bench_page_{i}.png"
    img.save(tmp)
    t_page = time.time()
    output = ""
    ntokens = 0
    for token in stream_generate(
        model, processor, formatted, [tmp],
        max_tokens=8192, verbose=False, temp=0.0,
    ):
        output += token.text
        ntokens += 1
        if "</doctag>" in output or "<|end_of_text|>" in output:
            break
    elapsed = time.time() - t_page
    seq_results.append((i + 1, ntokens, elapsed))
    tps = ntokens / elapsed if elapsed > 0 else 0
    print(f"  Page {i+1}: {ntokens} tokens in {elapsed:.2f}s ({tps:.0f} tok/s)")
t_seq = time.time() - t0

# Now try batched: process 2 pages at once by interleaving generation
# This is a "pseudo-batch" — prefill all pages first, then round-robin generate
print("\n--- Interleaved (prefill all, then round-robin decode) ---")

from mlx_vlm.generate import prepare_inputs
from mlx_vlm.models import cache as vlm_cache

# Prepare inputs for all pages
all_inputs = []
for i, img in enumerate(images):
    tmp = f"/tmp/bench_page_{i}.png"
    inputs = prepare_inputs(
        processor,
        images=[tmp],
        prompts=formatted,
        image_token_index=getattr(model.config, "image_token_index", None),
    )
    all_inputs.append(inputs)

# Prefill all pages (build KV caches)
t_batch = time.time()
caches = []
first_tokens = []

for i, inputs in enumerate(all_inputs):
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)
    extra_kwargs = {
        k: v for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    prompt_cache = vlm_cache.make_prompt_cache(model.language_model)
    outputs = model(input_ids, pixel_values, cache=prompt_cache, mask=mask, **extra_kwargs)
    logits = outputs.logits[:, -1, :]
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)

    caches.append(prompt_cache)
    first_tokens.append(token)

t_prefill = time.time() - t_batch
print(f"  Prefilled {len(images)} pages in {t_prefill:.2f}s ({t_prefill/len(images):.2f}s/page)")

# Decode all pages by round-robin stepping
t_decode = time.time()
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
page_outputs = [[] for _ in range(len(images))]
page_tokens = list(first_tokens)
page_done = [False] * len(images)
stop_strings = ["</doctag>", "<|end_of_text|>"]

max_tokens = 8192
for step in range(max_tokens):
    all_done = True
    for i in range(len(images)):
        if page_done[i]:
            continue
        all_done = False

        y = page_tokens[i]
        outputs = model.language_model(y[None], cache=caches[i])
        logits = outputs.logits[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        mx.eval(next_token)

        page_tokens[i] = next_token
        token_str = tokenizer.decode([next_token.item()])
        page_outputs[i].append(next_token.item())

        # Check for stop
        decoded = tokenizer.decode(page_outputs[i])
        if any(s in decoded for s in stop_strings):
            page_done[i] = True

    if all_done:
        break

t_decode = time.time() - t_decode
t_total_batch = time.time() - t_batch

print(f"  Decoded all pages in {t_decode:.2f}s")
for i in range(len(images)):
    ntok = len(page_outputs[i])
    print(f"  Page {i+1}: {ntok} tokens")

# Summary
print(f"\n{'=' * 50}")
print("COMPARISON")
print(f"{'=' * 50}")
print(f"Sequential:  {t_seq:.2f}s ({len(images)/t_seq:.2f} pages/sec)")
print(f"Interleaved: {t_total_batch:.2f}s ({len(images)/t_total_batch:.2f} pages/sec)")
print(f"Speedup:     {t_seq/t_total_batch:.2f}x")
