"""Authored comparison with the established wire-format backend check."""

import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from scripts import run_native_state_recovery_authored_study_002 as previous
from scripts import run_native_state_recovery_source_002 as historical
from scripts.native_appworld_worker_client import RUNTIME_TMP, WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import public_read, static_backend

ID = "qwen38-native-state-recovery-authored-003"
PLAN = "artifacts/qwen38-native-state-recovery-authored-model-plan-003.json"
PRIVATE = RUNTIME_TMP / "authored-panel-model-003"
MODULE = "scripts.run_native_state_recovery_authored_study_003"
SOURCE = MODULE.replace(".", "/") + ".py"
ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_SHA = "d88ff7dfae59b93b5a1b0b20f65f5c20187655889aab91d19585f92a078b373d"
WIRE_SOURCE = "scripts/run_native_state_recovery_source_002.py"
WIRE_SHA = "44a7eaa6ab4961faa3287a97141890064a21e90a02e0ebf1f4df46da939ec326"
STARTUP = "artifacts/native-state-recovery-authored-startup-terminal-002.json"
STARTUP_SHA = "363cae6bbc56a8c41cce168514ce3e76d7f24a142a56fd6af547fdbf3888dba6"
_require = previous.first._require
_hash = previous.first._hash


def _proof():
    raw = public_read(WORKSPACE, STARTUP)
    _require(_hash(raw) == STARTUP_SHA)
    result = _decode(raw)
    _require(result["scope"] == "authored_backend_comparison_failed_before_actor"
             and result["release_verified"] is True and result["backend_ready"] is False
             and result["executed_branches"] == result["model_calls"] == 0
             and result["automatic_retry"] is False)
    return result


def wire_projection(backend):
    return static_backend(historical.wire_backend(backend))


def _checked_projection(backend):
    projection = wire_projection(backend)
    expected = previous.first.episode._prior()["expected_backend_static"]["B3"]
    differences = sum(not _same(projection.get(key), expected.get(key)) for key in projection.keys() | expected.keys())
    if differences:
        sys.stderr.write(f"authored_backend_projection_mismatch_fields={differences}\n")
    return projection


def _verify_service_import():
    base = lineage()
    core = base.bindings()
    spec = """
from scripts import run_native_state_recovery_authored_study_003 as s
from scripts import serve_qwen38_b3_runtime_relocation_001 as relocated
c = s.lineage().bindings()
q = c._queue()
assert q._get_core().make_job(0, 'a'*32, 'b'*64) == c._job(0, 'a'*32, 'b'*64)
assert s.wire_projection({"fixture": (1, True)}) == {"fixture": [1, True]}
prior = c.episode._prior()
relocated.preflight(prior["runtime_relocation"]["sha256"], prior["service_plan_sha256"], prior["model_id"])
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([core.PYTHON, "-s", "-B", "-c", spec], cwd=core.episode.DEPLOYMENT,
                            env=env, capture_output=True, timeout=180, check=False)
    _require(result.returncode == 0 and result.stdout == b"")


@cache
def lineage():
    raw = public_read(ROOT, previous.SOURCE)
    _require(_hash(raw) == PREVIOUS_SHA and _hash(public_read(ROOT, WIRE_SOURCE)) == WIRE_SHA
             and Path(historical.__file__) == ROOT / WIRE_SOURCE)
    module = ModuleType("scripts._authored_003_lineage")
    module.__file__ = str(ROOT / SOURCE)
    exec(compile(raw, str(ROOT / previous.SOURCE), "exec", dont_inherit=True), vars(module))
    original_proof = module._proof

    def predecessors():
        _proof()
        return original_proof()

    vars(module).update(ID=ID, PLAN=PLAN, PRIVATE=PRIVATE, MODULE=MODULE, SOURCE=SOURCE,
                        _proof=predecessors, static_backend=_checked_projection,
                        _verify_service_import=_verify_service_import)
    core = module.bindings()
    core.REQUIRED_SOURCES |= {previous.SOURCE, WIRE_SOURCE}
    return module


def main():
    base = lineage()
    if sys.argv[1:] == ["--prepare-and-register"]:
        from scripts import audit_native_state_recovery_authored_startup_002 as audit

        _require(_same(audit.report(), _proof()))
        print(_encode(base.prepare_and_register()).decode())
        return 0
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
