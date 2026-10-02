"""Probe for ``test_plugin_registration_creates_no_cuda_context``.

Run as a script in a fresh interpreter (a CUDA context is process-global, so the
check is only meaningful in a clean process). Exits non-zero if loading the Pearl
plugin creates a CUDA context before any worker has bound its device.
"""

import torch
import vllm_miner

vllm_miner.register_pearl_miner_layer()

assert not torch.cuda.is_initialized(), (
    "plugin registration created a CUDA context before device binding"
)
