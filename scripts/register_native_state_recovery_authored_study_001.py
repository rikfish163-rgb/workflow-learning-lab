"""Freeze the selected authored study in both remote roots without enqueuing it."""

import argparse
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

from scripts import native_state_recovery_authored_episode_001 as episode
from scripts import native_state_recovery_authored_runtime_001 as assets
from scripts import prepare_native_state_recovery_authored_assets_001 as staging
from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts import run_native_state_recovery_authored_study_001 as study
from scripts.native_appworld_worker_client import (
    WORKSPACE,
    _encode,
    _open_directory,
    _private_tree,
    _same,
)
from scripts.native_source_registration import public_read


def _require(value):
    if not value:
        raise ValueError("native_authored_study_registration_invalid") from None


def _publish(path, raw, mode):
    parent_fd = _open_directory(path.parent)
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".wll-authored-registration-") as file:
            os.fchmod(file.fileno(), mode)
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
            try:
                os.link(Path(file.name).name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                        follow_symlinks=False)
            except FileExistsError:
                _require(not path.is_symlink() and path.is_file() and path.read_bytes() == raw)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _verify_service_import():
    spec = (
        "from scripts import run_native_state_recovery_authored_study_001 as s;"
        "q=s._queue();"
        "assert q._get_core().make_job(0,'a'*32,'b'*64)==s._job(0,'a'*32,'b'*64)"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([study.PYTHON, "-s", "-B", "-c", spec], cwd=episode.DEPLOYMENT,
                            env=env, capture_output=True, timeout=45, check=False)
    _require(result.returncode == 0 and result.stdout == b"")


def register():
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and Path.cwd() == WORKSPACE and episode.DEPLOYMENT.resolve(strict=True) == episode.DEPLOYMENT
             and shutil.disk_usage(study.PRIVATE).free >= 5 * 1024**3
             and shutil.disk_usage(episode.DEPLOYMENT).free >= 1024**3)
    os.umask(0o077)
    root = study.PRIVATE
    draft = comparison.drafts()["authored"]
    private_draft, prepared = staging._documents(root, draft)
    assets.verify_assets(root, prepared["initial_state_manifest_sha256"])
    _require(_same(episode._read(root / staging.PRIVATE_DRAFT, prepared["private_draft_sha256"]), private_draft))
    for name in ("configs", "runs", "scores"):
        fd = _private_tree(root / name)
        try:
            _require(not os.listdir(fd))
        finally:
            os.close(fd)
    _require(not (root / "parent-launch-intent.json").exists())
    prior = episode._prior()
    runtime = study.Runtime(WORKSPACE)
    try:
        runtime.service.plan = prior
        runtime.monitor(empty_port=True)
    finally:
        runtime.close()
    sources = {}
    for name in sorted(study.REQUIRED_SOURCES):
        raw = public_read(WORKSPACE, name)
        _publish(episode.DEPLOYMENT / name, raw, 0o644)
        _require(public_read(episode.DEPLOYMENT, name) == raw)
        sources[name] = study._hash(raw)
    _require(sources[study.QUEUE_SOURCE] == study.QUEUE_SHA and sources[study.LOOP_SOURCE] == study.LOOP_SHA)
    _verify_service_import()
    private_plan = {**private_draft, "status": "registered", "utility_authorized": True}
    private_bytes = _encode(private_plan) + b"\n"
    public_plan = {
        "id": study.ID, "status": "registered", "utility_authorized": True,
        "private_sha256": study._hash(private_bytes), "draft_summary_sha256": episode.DRAFT_SUMMARY_SHA256,
        "source_sha256": sources,
    }
    public_bytes = _encode(public_plan) + b"\n"
    _publish(root / episode.REGISTRATION, private_bytes, 0o600)
    _publish(episode.DEPLOYMENT / study.PLAN, public_bytes, 0o644)
    _publish(WORKSPACE / study.PLAN, public_bytes, 0o644)
    sha = study._hash(public_bytes)
    plan, _bundle = study._preflight(WORKSPACE, sha)
    _require(plan["draft"]["split"] == "authored" and plan["dev_or_test_authorized"] is False)
    return {"id": study.ID, "plan_sha256": sha, "pairs": 27, "batches": 14, "official_test_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-authored", action="store_true", required=True)
    parser.parse_args()
    try:
        print(_encode(register()).decode())
    except Exception:
        raise SystemExit("native_authored_study_registration_invalid") from None
