"""Test 8 concurrent pages via vllm-mlx server."""

import base64
import time
import concurrent.futures
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Load all page images
page_data = []
for i in range(8):
    with open(f"/tmp/bench_page_{i}.png", "rb") as f:
        page_data.append(base64.b64encode(f.read()).decode())


def process_page(args):
    idx, b64 = args
    t0 = time.time()
    resp = client.chat.completions.create(
        model="ibm-granite/granite-docling-258M-mlx",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Convert this page to docling."},
        ]}],
        max_tokens=8192,
        temperature=0.0,
    )
    elapsed = time.time() - t0
    tokens = resp.usage.completion_tokens
    return idx + 1, tokens, elapsed


# Send all 8 pages concurrently
print("Sending 8 pages concurrently...")
t_total = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(process_page, enumerate(page_data)))
t_total = time.time() - t_total

print()
print("=" * 50)
print("BENCHMARK: vllm-mlx continuous batching (8 concurrent)")
print("=" * 50)
total_tokens = sum(r[1] for r in results)
print(f"Wall clock:      {t_total:.2f}s")
print(f"Pages:           8")
print(f"Pages/sec:       {8/t_total:.2f}")
print(f"Sec/page:        {t_total/8:.2f}")
print(f"Total tokens:    {total_tokens}")
print(f"Aggregate tok/s: {total_tokens/t_total:.0f}")

print()
print("Per-page:")
for page, tokens, elapsed in sorted(results):
    print(f"  Page {page}: {tokens} tokens, {elapsed:.2f}s wall")
