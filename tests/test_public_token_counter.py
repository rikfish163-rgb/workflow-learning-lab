"""The public counter keeps local execution while native callers can import it."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import verification_token_counter as counter


def test_missing_tokenizer_configuration_stops_before_child_launch(monkeypatch):
    monkeypatch.setattr(counter, "MODEL_PATH", "")

    def unexpected(*args, **kwargs):
        pytest.fail("unconfigured counter started a child")

    monkeypatch.setattr(counter.subprocess, "run", unexpected)
    with pytest.raises(counter.TokenCountError, match="tokenizer_path_required"):
        counter.count_input_tokens([{"role": "user", "content": "synthetic"}])


def test_configured_counter_uses_local_cpu_child_and_validates_proof(monkeypatch):
    monkeypatch.setattr(counter, "MODEL_PATH", "/synthetic/tokenizer")
    messages = [{"role": "user", "content": "synthetic"}]

    def run(argv, **kwargs):
        assert argv == [sys.executable, "-I", "-B", "-c", counter.CPU_CODE]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        payload = json.loads(kwargs["input"])
        assert payload["messages"] == messages
        assert payload["model_path"] == "/synthetic/tokenizer"
        proof = {"status": "measured_CPU_tokenization",
                 "messages_sha256": payload["messages_sha256"], "input_tokens": 11,
                 "model_path": payload["model_path"],
                 "tokenizer_files_sha256": payload["tokenizer_files_sha256"],
                 "CUDA_VISIBLE_DEVICES": "", "cuda_initialized": False,
                 "model_weights_loaded": False, "GPU_jobs": 0, "model_calls": 0}
        return SimpleNamespace(returncode=0, stdout=json.dumps(proof).encode())

    monkeypatch.setattr(counter.subprocess, "run", run)
    assert counter.count_input_tokens(messages) == 11
    assert counter.PYTHON_PATH == sys.executable
