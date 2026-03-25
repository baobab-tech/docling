# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "docling[vlm]>=2.64.0",
# ]
# ///

import logging
import os
import tempfile
import time
import urllib.request

logging.basicConfig(level=logging.WARNING)
logging.getLogger("docling.models").setLevel(logging.DEBUG)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Download PDF first
pdf_url = "https://arxiv.org/pdf/2501.17887"
pdf_path = os.path.join(tempfile.gettempdir(), "benchmark.pdf")
if not os.path.exists(pdf_path):
    urllib.request.urlretrieve(pdf_url, pdf_path)
source = pdf_path

# --- Setup: point at local vllm-mlx server ---
t_setup_start = time.time()

vlm_options = ApiVlmOptions(
    url="http://localhost:8000/v1/chat/completions",
    params={"model": "ibm-granite/granite-docling-258M-mlx", "max_tokens": 8192},
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    timeout=120,
    concurrency=4,  # fire 4 pages at once
    stop_strings=["</doctag>", "<|end_of_text|>"],
)

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_options,
    enable_remote_services=True,
    generate_page_images=True,
    generate_picture_images=False,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

t_setup = time.time() - t_setup_start

# --- Conversion ---
t_convert_start = time.time()
result = converter.convert(source=source)
t_convert = time.time() - t_convert_start

doc = result.document
num_pages = len(result.pages)

print("=" * 50)
print("BENCHMARK: GraniteDocling via vllm-mlx server")
print("=" * 50)
print(f"Source:          {source}")
print(f"Pages:           {num_pages}")
print(f"Setup time:      {t_setup:.2f}s")
print(f"Conversion time: {t_convert:.2f}s")
print(f"Total time:      {t_setup + t_convert:.2f}s")
if num_pages > 0:
    print(f"Pages/sec:       {num_pages / t_convert:.2f}")
    print(f"Sec/page:        {t_convert / num_pages:.2f}")

# --- Per-page VLM details ---
print(f"\n{'─' * 50}")
print("PER-PAGE VLM DETAILS")
print(f"{'─' * 50}")
for p in result.pages:
    vlm = p.predictions.vlm_response
    if vlm:
        tps = vlm.num_tokens / vlm.generation_time if vlm.generation_time > 0 else 0
        print(f"  Page {p.page_no}: {vlm.num_tokens} tokens in {vlm.generation_time:.2f}s ({tps:.1f} tok/s)")
