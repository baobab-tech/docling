"""Patch granite-docling's processor to have a chat template, then start vllm-mlx."""

import sys
import mlx_vlm

# Monkey-patch mlx_vlm.load to copy chat_template from tokenizer to processor
_original_load = mlx_vlm.load

def _patched_load(*args, **kwargs):
    model, processor = _original_load(*args, **kwargs)
    if not getattr(processor, "chat_template", None):
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer and getattr(tokenizer, "chat_template", None):
            processor.chat_template = tokenizer.chat_template
            print(f"[PATCH] Copied chat_template from tokenizer to processor")
    return model, processor

mlx_vlm.load = _patched_load
# Also patch the import location vllm-mlx uses
import vllm_mlx.models.mllm as mllm_module
if hasattr(mllm_module, 'load'):
    mllm_module.load = _patched_load

# Now start vllm-mlx as normal
# Patch prefill step size before engine loads
import vllm_mlx.mllm_scheduler as sched
_orig_init = sched.MLLMSchedulerConfig.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.prefill_step_size = 4096
sched.MLLMSchedulerConfig.__init__ = _patched_init

sys.argv = [
    "vllm-mlx",
    "--model", "ibm-granite/granite-docling-258M-mlx",
    "--port", "8000",
    "--mllm",
    "--continuous-batching",
]

from vllm_mlx.server import main
main()
