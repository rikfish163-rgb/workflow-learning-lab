"""Independent authored comparison with the verified relocated B3 service."""

import argparse
import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import native_state_recovery_gpu_queue_001 as historical_queue
from scripts import run_native_state_recovery_authored_study_001 as first
from scripts.native_appworld_worker_client import RUNTIME_TMP, WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import public_read, static_backend

ID = "qwen38-native-state-recovery-authored-002"
PLAN = "artifacts/qwen38-native-state-recovery-authored-model-plan-002.json"
PRIVATE = RUNTIME_TMP / "authored-panel-model-002"
MODULE = "scripts.run_native_state_recovery_authored_study_002"
SOURCE = MODULE.replace(".", "/") + ".py"
ROOT = Path(__file__).resolve().parents[1]
STARTUP = "artifacts/native-state-recovery-authored-startup-terminal-001.json"
STARTUP_SHA = "05c8b6a8a79eb6a9cec11efa198c68d092bb3cba820d007dafbd564333afc988"
SERVICE_ADAPTER = "scripts/native_state_recovery_gpu_queue_001.py"
RELOCATED = "scripts/serve_qwen38_b3_runtime_relocation_001.py"
STAGING = "scripts/prepare_native_state_recovery_authored_assets_001.py"
REGISTRAR = "scripts/register_native_state_recovery_authored_study_001.py"
REPORTER = "scripts/report_native_state_recovery_authored_study_001.py"
PINS = {
    first.SOURCE: "cdc1d7227ccc93a5fdf7c63c3ea54003a7f9aabad5b1751405de01c1c77dbb40",
    first.episode.EPISODE_SOURCE: "6de2ff47d3ea259a7bfa88c22bacfb4af6b9b07673ba7d30fb2893489b69f56c",
    SERVICE_ADAPTER: "334daa20a24621c1a53eab895ab232671649c9797c9da18350a22b507da50453",
    RELOCATED: "c74195277b35a27bcc55fc6d4d059a51d7966a8507b76b0a371d6a66b6aee9ad",
    STAGING: "07248d54fb08638c22055011852a7ccbe502a583fff0db7084d87a9911f7eb65",
    REGISTRAR: "e743531a06388028d168ec2edfda51e893571ff1591dd66acd49ca18e6f51264",
    REPORTER: "f6082b3d82b29ccd630ba7d62a6008f00a7662ed1faec01c1d142c399e11825b",
}


def _proof():
    raw = public_read(WORKSPACE, STARTUP)
    first._require(first._hash(raw) == STARTUP_SHA)
    result = _decode(raw)
    first._require(result["release_verified"] is True and result["executed_branches"] == result["model_calls"] == 0
                   and result["backend_failure"] == "legacy_runtime_digest_mismatch"
                   and result["ticket_consumed"] is result["automatic_retry"] is False)
    return result


def _verify_source(path, root):
    raw = public_read(root, path)
    first._require(first._hash(raw) == PINS[path])
    return raw


def _clone(path, *, root=ROOT, **bindings):
    raw = _verify_source(path, root)
    module = ModuleType("scripts._authored_002_" + Path(path).stem)
    module.__file__ = str(root / path)
    exec(compile(raw, module.__file__, "exec", dont_inherit=True), vars(module))
    vars(module).update(bindings)
    return module


@cache
def bindings():
    core = _clone(first.SOURCE, ID=ID, PLAN=PLAN, PRIVATE=PRIVATE, MODULE=MODULE, SOURCE=SOURCE)
    core.episode = _clone(first.episode.EPISODE_SOURCE, ID=ID)
    core.REQUIRED_SOURCES |= {SOURCE, SERVICE_ADAPTER, RELOCATED}
    base_load = core.load_public

    def load_public(root, sha):
        _proof()
        for path in (first.SOURCE, first.episode.EPISODE_SOURCE, SERVICE_ADAPTER, RELOCATED):
            _verify_source(path, root)
        plan = base_load(root, sha)
        prior = core.episode._prior()
        return {**plan, "runtime_relocation": prior["runtime_relocation"],
                "service_plan_sha256": prior["service_plan_sha256"], "model_id": prior["model_id"]}

    core.load_public = load_public
    base_queue = core._queue

    @cache
    def queue():
        q = base_queue()
        _verify_source(SERVICE_ADAPTER, ROOT)
        first._require(Path(historical_queue.__file__) == ROOT / SERVICE_ADAPTER)
        prior = core.episode._prior()
        registry = SimpleNamespace(load_registration=load_public, static_backend=static_backend,
                                   _same=_same, PORT=prior["port"])
        q.serve = core._bind(historical_queue.serve, _core=lambda: q, registration=registry,
                             ROOT=ROOT, REMOTE=core.episode.DEPLOYMENT)
        return q

    core._queue = queue
    return core


def _verify_service_import():
    core = bindings()
    spec = """
from scripts import run_native_state_recovery_authored_study_002 as s
from scripts import serve_qwen38_b3_runtime_relocation_001 as relocated
c = s.bindings()
q = c._queue()
assert q._get_core().make_job(0, 'a'*32, 'b'*64) == c._job(0, 'a'*32, 'b'*64)
prior = c.episode._prior()
relocated.preflight(prior["runtime_relocation"]["sha256"], prior["service_plan_sha256"], prior["model_id"])
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([core.PYTHON, "-s", "-B", "-c", spec], cwd=core.episode.DEPLOYMENT,
                            env=env, capture_output=True, timeout=180, check=False)
    first._require(result.returncode == 0 and result.stdout == b"")


def prepare_and_register():
    from scripts import audit_native_state_recovery_authored_startup_001 as audit

    first._require(_same(audit.report(), _proof()))
    core = bindings()
    staging = _clone(STAGING, root=WORKSPACE, study=core, episode=core.episode)
    registrar = _clone(REGISTRAR, root=WORKSPACE, study=core, episode=core.episode, staging=staging,
                       _verify_service_import=_verify_service_import)
    staging.prepare()
    return registrar.register()


def main():
    if sys.argv[1:] == ["--prepare-and-register"]:
        print(_encode(prepare_and_register()).decode())
        return 0
    if sys.argv[1:2] == ["--audit"]:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--audit", action="store_true", required=True)
        parser.add_argument("--plan-sha256", required=True)
        parser.add_argument("--result-sha256", required=True)
        args = parser.parse_args()
        reporter = _clone(REPORTER, root=WORKSPACE, study=bindings())
        print(_encode(reporter.report(args.plan_sha256, args.result_sha256)).decode())
        return 0
    return bindings().main()


if __name__ == "__main__":
    raise SystemExit(main())
