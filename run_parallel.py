# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "docling[vlm]>=2.64.0",
# ]
# ///

"""Parallel MLX inference — one process per page bypasses the global MLX lock."""

import multiprocessing as mp
import os
import tempfile
import time
import urllib.request

# Download PDF first
pdf_url = "https://arxiv.org/pdf/2501.17887"
pdf_path = os.path.join(tempfile.gettempdir(), "benchmark.pdf")
if not os.path.exists(pdf_path):
    urllib.request.urlretrieve(pdf_url, pdf_path)


_worker_state = {}


def init_worker():
    """Load model once per worker process."""
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model_path = "ibm-granite/granite-docling-258M-mlx"
    model, processor = load(model_path)
    config = load_config(model_path)
    formatted = apply_chat_template(processor, config, "Convert this page to docling.", num_images=1)

    _worker_state["model"] = model
    _worker_state["processor"] = processor
    _worker_state["formatted"] = formatted
    _worker_state["stream_generate"] = stream_generate


def process_page(args):
    """Worker: process a single page image (model already loaded)."""
    page_img_path, page_no = args
    from PIL import Image

    model = _worker_state["model"]
    processor = _worker_state["processor"]
    formatted = _worker_state["formatted"]
    stream_generate = _worker_state["stream_generate"]

    img = Image.open(page_img_path)

    t0 = time.time()
    output = ""
    num_tokens = 0
    for token in stream_generate(
        model, processor, formatted,
        [img],
        max_tokens=8192,
        verbose=False,
        temp=0.0,
    ):
        output += token.text
        num_tokens += 1
        if "</doctag>" in output or "<|end_of_text|>" in output:
            break

    elapsed = time.time() - t0
    return page_no, num_tokens, elapsed


def main():
    import pypdfium2 as pdfium

    # Rasterize all pages
    t_raster = time.time()
    pdf = pdfium.PdfDocument(pdf_path)
    page_paths = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=2.0)
        img = bitmap.to_pil()
        p = os.path.join(tempfile.gettempdir(), f"page_{i}.png")
        img.save(p)
        page_paths.append((p, i + 1))
    t_raster = time.time() - t_raster
    num_pages = len(page_paths)
    print(f"Rasterized {num_pages} pages in {t_raster:.2f}s")

    # Parallel VLM inference — one process per page
    num_workers = min(4, num_pages)  # 4 concurrent MLX processes
    print(f"Running {num_workers} parallel workers...")

    t_infer = time.time()
    with mp.Pool(num_workers, initializer=init_worker) as pool:
        results = pool.map(process_page, page_paths)
    t_infer = time.time() - t_infer

    total = t_raster + t_infer

    print(f"\n{'=' * 50}")
    print("BENCHMARK: Parallel MLX (multiprocessing)")
    print(f"{'=' * 50}")
    print(f"Workers:         {num_workers}")
    print(f"Pages:           {num_pages}")
    print(f"Rasterize:       {t_raster:.2f}s")
    print(f"Inference:       {t_infer:.2f}s")
    print(f"Total:           {total:.2f}s")
    print(f"Pages/sec:       {num_pages / t_infer:.2f}")
    print(f"Sec/page:        {t_infer / num_pages:.2f}")

    print(f"\n{'─' * 50}")
    print("PER-PAGE DETAILS")
    print(f"{'─' * 50}")
    for page_no, tokens, elapsed in sorted(results):
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  Page {page_no}: {tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
