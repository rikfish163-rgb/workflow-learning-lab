"""One admitted synthetic generation with no AppWorld or scorer operations."""

import argparse
import hashlib
import os
import re
import sys
import sysconfig
import time
from functools import cache
from pathlib import Path

from scripts import native_appworld_model_client_004 as model
from scripts import run_native_state_recovery_source_002 as source
from scripts.native_appworld_worker_client import WORKSPACE, _encode
from scripts.native_source_registration import PRIVATE_STATE_ROOT, _absolute_hash, _same, _sha

ID = "qwen38-native-state-recovery-generation-probe-001"
PLAN = f"artifacts/{ID}-plan.json"
MODULE = "scripts.probe_native_state_recovery_generation_001"
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + "/native-state-recovery-generation-probe-001"
SOURCE_FILE = "scripts/probe_native_state_recovery_generation_001.py"
FAILURE = "artifacts/native-state-recovery-source-startup-terminal-002.json"
FAILURE_SHA256 = "c2c76ee4ed1a78097e0310349eba0b60a7bc38df9b47928bbf32d73068921904"
SOURCE_SHA256 = "c081d26ea5c32a58ebe1f6e0f7fd22f31e6ac9cdc05f06373a30123e65bee6b9"
MESSAGES = [{"role": "user", "content": "Respond with exactly the word ready."}]
SEED = 17
MAX_TOKENS = 16


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_generation_probe_invalid")


def header_binding():
    include = Path(sysconfig.get_path("include"))
    _require(include == Path(PRIVATE_STATE_ROOT) / "ubuntu-python312-runtime-001/include/python3.12")
    rows = {path.relative_to(include).as_posix(): _absolute_hash(str(path))
            for path in include.rglob("*.h")}
    _require(len(rows) > 200 and "Python.h" in rows and "x86_64-linux-gnu/python3.12/pyconfig.h" in rows)
    return {"header_file_count": len(rows), "headers_sha256": hashlib.sha256(_encode(rows)).hexdigest()}


@cache
def bindings():
    reg, queue, episode, runner = source.bindings.__wrapped__()
    for component in (reg, queue, episode, runner):
        component.ID, component.PLAN = ID, PLAN
    reg.PRIVATE_RUN_ROOT = PRIVATE_RUN_ROOT
    queue.GUARD_NAME = "native-state-recovery-generation-probe-operations-001.lock"
    reg.FIXED_INPUTS = reg.FIXED_INPUTS | {FAILURE: FAILURE_SHA256, source.PLAN: SOURCE_SHA256}
    reg.NEW_SOURCES = reg.NEW_SOURCES | {SOURCE_FILE}
    build_source = reg.build

    def build(root):
        data = build_source(root)
        failure = reg.public._read(root, FAILURE)
        _require(failure["source_plan_sha256"] == SOURCE_SHA256 and failure["release_verified"] is True
                 and failure["compiler_error_category"] == "missing_python_3_12_development_header")
        data.update(scope="one_synthetic_generation_no_task", generation_messages_sha256=hashlib.sha256(
                    _encode(MESSAGES)).hexdigest(), generation_seed=SEED, generation_max_tokens=MAX_TOKENS,
                    total_gpu_seconds=7200, compiled_header_binding=header_binding())
        return data

    reg.build = build
    make_job = queue.make_job

    def job(index, owner, sha):
        _require(type(index) is int and index == 0 and _sha(sha))
        result = make_job(index, owner, sha)
        result["argv"][3] = MODULE
        return result

    queue.make_job = job
    return reg, queue, runner


def core_for(runner):
    core = runner._core()
    command = core.unit_command

    def probe_command(kind, *args, **kwargs):
        _require(kind == "pilot")
        return [item.replace(source.MODULE, MODULE).replace(f"RuntimeMaxSec={runner.PARENT_SECONDS}",
                "RuntimeMaxSec=94200") for item in command(kind, *args, **kwargs)]

    core.unit_command = probe_command
    return core


def run(sha):
    reg, queue, runner = bindings()
    core = core_for(runner)
    runtime = core._Runtime(WORKSPACE)
    plan = runtime.load_plan(sha)
    runtime.begin(plan, sha, [])
    result = {"id": ID, "status": "unknown", "plan_sha256": sha, "source_tasks": 0, "actions": 0,
              "official_scores": 0, "generation_intents": 0, "generation_returns": 0,
              "costs": {"GPU_jobs": []}}
    try:
        job = queue.make_job(0, runtime.owner(), sha)
        with runner._lease(runtime, core, job, result) as (service, guard):
            guard()
            client = model.NativeModel(f"http://127.0.0.1:{reg.PORT}/v1", reg.MODEL_ID,
                expected_backend=service["backend"], max_tokens=MAX_TOKENS, context_tokens=16384,
                temperature=0.7, timeout=300, deadline=time.monotonic() + 420)
            client.check_health()
            tokens = client.count_input_tokens(MESSAGES)
            _require(type(tokens) is int and 1 <= tokens <= 15360)
            result["generation_intents"] = 1
            runtime.record({"event": "synthetic_generation_intent", "input_tokens": tokens, "seed": SEED})
            reply = client.generate(MESSAGES, seed=SEED)
            result["generation_returns"] = 1
            result.update(prompt_tokens=reply["prompt_tokens"], completion_tokens=reply["completion_tokens"],
                          output_sha256=hashlib.sha256(reply["text"].encode()).hexdigest())
            _require(reply["format_error"] is None and result["prompt_tokens"] == tokens
                     and type(result["completion_tokens"]) is int and 1 <= result["completion_tokens"] <= MAX_TOKENS)
            runtime.record({"event": "synthetic_generation_returned", "prompt_tokens": tokens,
                            "completion_tokens": result["completion_tokens"]})
        result["status"] = "passed"
    except BaseException as error:
        core._capture_initiating_exception(runtime, error)
    finally:
        runtime.record({"event": "probe_stop", "status": result["status"]})
        runtime.write("study-result.json", result)
        runtime.close()
    return int(result["status"] != "passed")


def main():
    if "--serve" in sys.argv[1:]:
        return bindings()[1].main()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and re.fullmatch(r"[a-f0-9]{64}", args.plan_sha256))
    if args.launch:
        print(core_for(bindings()[2]).launch_pilot(args.plan_sha256, root=WORKSPACE)["unit"])
        return 0
    return run(args.plan_sha256)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write("native_state_recovery_generation_probe_failed\n")
        raise SystemExit(2) from None
