"""Prepare the authored inputs without creating an execution registration."""

import ctypes
import os
import socket
import stat
import tempfile
from pathlib import Path

from scripts import native_state_recovery_authored_episode_001 as episode
from scripts import native_state_recovery_authored_runtime_001 as assets
from scripts import prepare_native_state_recovery_comparison_001 as comparison
from scripts import run_native_state_recovery_authored_study_001 as study
from scripts.native_appworld_worker_client import (
    WORKSPACE,
    _decode,
    _encode,
    _exclusive,
    _private_tree,
    _same,
)
from scripts.native_source_registration import _public_hash, public_read
from scripts.native_state_recovery_authored_panel_001 import PANEL
from scripts.probe_native_state_recovery_authored_panel_001 import _prepare
from scripts.probe_native_state_recovery_authored_world_001 import SCRATCH

PREPARED = "assets-prepared.json"
PRIVATE_DRAFT = "study-registration-draft.json"
RENAME_NOREPLACE = 1


def _require(condition):
    if not condition:
        raise ValueError("native_authored_asset_staging_invalid") from None


def _absent(path):
    return not path.exists() and not path.is_symlink()


def _require_unregistered(root):
    _require(_absent(root / episode.REGISTRATION)
             and _absent(WORKSPACE / study.PLAN)
             and _absent(episode.DEPLOYMENT / study.PLAN))


def _write(root, name, value):
    fd = _private_tree(root)
    try:
        with _exclusive(fd, name) as file:
            file.write(_encode(value) + b"\n")
            file.flush()
            os.fsync(file.fileno())
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(staging, final):
    _require(staging.parent == final.parent == SCRATCH)
    fd = _private_tree(staging)
    os.close(fd)
    parent_fd = _private_tree(staging.parent)
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(parent_fd, os.fsencode(staging.name), parent_fd, os.fsencode(final.name), RENAME_NOREPLACE):
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _documents(final, draft):
    raw = public_read(WORKSPACE, episode.DRAFT_SUMMARY)
    _require(episode._hash(raw) == episode.DRAFT_SUMMARY_SHA256)
    selected = _decode(raw)["options"]["authored"]
    _require(episode._hash(_encode(draft)) == selected["draft_sha256"]
             and draft["model_execution_authorized"] is False)
    private_draft = {
        "id": study.ID, "status": "draft_no_execution", "utility_authorized": False,
        "dev_or_test_authorized": False, "automatic_retry": False, "asset_root": str(final),
        "draft": draft, "episode_source_sha256": _public_hash(WORKSPACE, episode.EPISODE_SOURCE),
    }
    prepared = {
        "scope": "authored_inputs_prepared_no_execution", "scenarios": 9, "task_variants": 27,
        "draft_sha256": selected["draft_sha256"],
        "private_draft_sha256": episode._hash(_encode(private_draft) + b"\n"),
        "initial_state_manifest_sha256": draft["initial_state_manifest_sha256"],
        "api_schema_sha256": assets.API_SCHEMA_SHA256, "utility_authorized": False,
        "official_task_bodies_opened": 0, "model_calls": 0, "gpu_jobs": 0,
    }
    return private_draft, prepared


def _verify(root, draft, prepared):
    assets.verify_assets(root, prepared["initial_state_manifest_sha256"])
    _require(_same(episode._read(root / PRIVATE_DRAFT, prepared["private_draft_sha256"]), draft))
    _require(_same(episode._read(root / PREPARED, episode._hash(_encode(prepared) + b"\n")), prepared))
    for name in ("configs", "runs", "scores"):
        path = root / name
        fd = _private_tree(path)
        try:
            _require(not os.listdir(fd))
        finally:
            os.close(fd)
    _require_unregistered(root)
    return prepared


def prepare():
    _require(socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
             and os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    info = SCRATCH.stat(follow_symlinks=False)
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
             and info.st_uid == os.geteuid())
    os.umask(0o077)
    final = study.PRIVATE
    _require_unregistered(final)
    draft, prepared = _documents(final, comparison.drafts()["authored"])
    if not _absent(final):
        return _verify(final, draft, prepared)
    with tempfile.TemporaryDirectory(prefix="authored-panel-staging-", dir=SCRATCH) as directory:
        staging = Path(directory)
        _prepare(staging, PANEL.values())
        for name in ("configs", "scores"):
            (staging / name).mkdir(mode=0o700)
        _write(staging, PRIVATE_DRAFT, draft)
        _write(staging, PREPARED, prepared)
        _verify(staging, draft, prepared)
        _require_unregistered(final)
        try:
            _publish(staging, final)
        except FileExistsError:
            # Another identical preparation may already have published.
            return _verify(final, draft, prepared)
    return _verify(final, draft, prepared)


if __name__ == "__main__":
    try:
        print(_encode(prepare()).decode())
    except Exception:
        raise SystemExit("native_authored_asset_staging_invalid") from None
