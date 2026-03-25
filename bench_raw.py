# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "pypdfium2",
#     "mlx-vlm",
#     "pillow",
# ]
# ///

"""Profile rasterization vs VLM inference separately."""

import time
import pypdfium2 as pdfium
from PIL import Image

pdf_path = "/var/folders/09/d7r37q9d7k9757wsh2k8s8280000gn/T/benchmark.pdf"

# --- Phase 1: Rasterize pages to images ---
t0 = time.time()
pdf = pdfium.PdfDocument(pdf_path)
images = []
for i in range(len(pdf)):
    page = pdf[i]
    bitmap = page.render(scale=2.0)
    img = bitmap.to_pil()
    images.append(img)
t_raster = time.time() - t0
print(f"Rasterization: {t_raster:.2f}s for {len(images)} pages ({t_raster/len(images):.3f}s/page)")
print(f"Image sizes: {images[0].size}")

# --- Phase 2: Run MLX model directly ---
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "ibm-granite/granite-docling-258M-mlx"

t1 = time.time()
model, processor = load(model_path)
config = load_config(model_path)
t_load = time.time() - t1
print(f"\nModel load: {t_load:.2f}s")

prompt = "Convert this page to docling."

page_times = []
total_tokens = 0
for i, img in enumerate(images):
    # Save temp image for mlx_vlm
    tmp_path = f"/tmp/page_{i}.png"
    img.save(tmp_path)

    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    t_page = time.time()
    output = generate(
        model, processor, formatted,
        image=[tmp_path],
        max_tokens=8192,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.time() - t_page
    tokens = len(output.split()) if isinstance(output, str) else 0
    page_times.append(elapsed)
    total_tokens += tokens
    print(f"  Page {i+1}: {elapsed:.2f}s (~{tokens} words)")

t_inference = sum(page_times)
print(f"\nVLM inference: {t_inference:.2f}s total ({t_inference/len(images):.2f}s/page)")
print(f"Total: {t_raster + t_load + t_inference:.2f}s")
print(f"Pages/sec (inference only): {len(images)/t_inference:.2f}")
