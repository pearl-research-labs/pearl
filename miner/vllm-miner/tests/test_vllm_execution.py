"""Greedy generations under ``--quantization pearl`` match recorded references.

Each model serves once through the mined pipeline and once with the
``MINER_NO_MINING`` kill switch. Regenerate ``reference_outputs.json`` with
``REGENERATE_VLLM_REFERENCES=true``.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

MODELS = {
    "qwen3_dense": "Qwen/Qwen3-0.6B",
    "olmoe": "allenai/OLMoE-1B-7B-0125-Instruct",
}
PROMPTS = {
    "translate": [
        {
            "role": "system",
            "content": "You are a French translator. Reply with ONLY the French translation "
            "of the user's sentence. No preamble, no quotes, no explanation.",
        },
        {"role": "user", "content": "The cat is on the table."},
    ],
    "capital": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
}
REFERENCES_FILE = Path(__file__).with_name("reference_outputs.json")
REGENERATE = os.environ.get("REGENERATE_VLLM_REFERENCES", "").lower() in ("1", "true", "yes")


@pytest.fixture(scope="module")
def references():
    refs = json.loads(REFERENCES_FILE.read_text()) if REFERENCES_FILE.exists() else {}
    yield refs
    if REGENERATE:
        REFERENCES_FILE.write_text(json.dumps(refs, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mining", [True, False], ids=["mining", "no_mining"])
@pytest.mark.parametrize("model_id", MODELS)
def test_generation_matches_reference(model_id, mining, references, monkeypatch):
    monkeypatch.setenv("MINER_NO_MINING", str(not mining).lower())
    monkeypatch.setenv("MINER_NO_GATEWAY", "true")
    # Mine the short reference prompts' prefills too.
    monkeypatch.setenv("PEARL_MIN_MINING_TOKENS", "4")
    # Lets collective_rpc ship the counter probe below to the worker.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams

    def credited_hashes(_worker):
        from vllm_miner.mining_state import get_async_manager

        return get_async_manager().credited_hashes

    llm = LLM(
        MODELS[model_id],
        quantization="pearl",
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        attention_config={"backend": "TRITON_ATTN"},
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=32)
        mismatches = {}
        for prompt_id, messages in PROMPTS.items():
            key = f"{model_id}_{prompt_id}_{'mining' if mining else 'no_mining'}"
            output = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
            text = output[0].outputs[0].text
            if REGENERATE:
                references[key] = text
            elif references.get(key) != text:
                mismatches[key] = {"expected": references.get(key), "actual": text}
        assert not mismatches, json.dumps(mismatches, indent=2, ensure_ascii=False)
        assert (llm.collective_rpc(credited_hashes)[0] > 0) == mining
    finally:
        llm.llm_engine.engine_core.shutdown()
