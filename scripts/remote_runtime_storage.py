"""Check the required local SSHFS views before a new utility study uses them.

The result records current mount and directory observations; it does not reserve
the filesystems or make later writes atomic with this check.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

REMOTE_WORKSPACE_ROOT = '/home/hetaisheng/projects/workflow-learning-lab-remote'
REMOTE_PRIVATE_ROOT = '/home/hetaisheng/.local/share/workflow-learning-lab'
REMOTE_TMP_DIR = '/home/hetaisheng/.local/share/workflow-learning-lab/runtime-tmp'
REMOTE_WORKSPACE_SOURCE = '4090:/home/data4/hts/workflow-learning-lab/runtime-remote-001/workspace'
REMOTE_PRIVATE_SOURCE = '4090:/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state'
MIN_FREE_BYTES = 5 * 1024**3
_ERROR = 'remote_runtime_storage_invalid'


def _require(condition):
    if not condition:
        raise ValueError(_ERROR)


def _mount(path, source, target):
    result = subprocess.run(
        ['findmnt', '--json', '--output', 'SOURCE,FSTYPE,TARGET', '--target', path],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=True, timeout=5)
    _require(type(result.stdout) is str and len(result.stdout) <= 8192)
    value = json.loads(result.stdout)
    _require(type(value) is dict and set(value) == {'filesystems'})
    filesystems = value['filesystems']
    _require(type(filesystems) is list and len(filesystems) == 1)
    expected = {'source': source, 'fstype': 'fuse.sshfs', 'target': target}
    _require(type(filesystems[0]) is dict and filesystems[0] == expected)
    return expected


def _directory(path, *, mode=None):
    parsed = Path(path)
    _require(parsed.is_absolute() and parsed.resolve(strict=True) == parsed)
    info = parsed.lstat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
             and (mode is None or stat.S_IMODE(info.st_mode) == mode))


def _free_bytes(path):
    info = os.statvfs(path)
    _require(type(info.f_bavail) is int and info.f_bavail >= 0
             and type(info.f_frsize) is int and info.f_frsize > 0)
    free = info.f_bavail * info.f_frsize
    _require(free >= MIN_FREE_BYTES)
    return free


def verify_storage(root, tmp_dir):
    """Fail closed on a local fallback, wrong temporary path or unsafe directory.

    Only public mount, path and capacity metadata is returned. No SSH session,
    private input read, directory creation or fallback write is performed.
    """
    try:
        _require(isinstance(root, (str, Path)) and str(root) == REMOTE_WORKSPACE_ROOT
                 and type(tmp_dir) is str and tmp_dir == REMOTE_TMP_DIR
                 and os.environ.get('TMPDIR') == REMOTE_TMP_DIR)
        workspace = _mount(REMOTE_WORKSPACE_ROOT, REMOTE_WORKSPACE_SOURCE, REMOTE_WORKSPACE_ROOT)
        private = _mount(REMOTE_PRIVATE_ROOT, REMOTE_PRIVATE_SOURCE, REMOTE_PRIVATE_ROOT)
        _require(_mount(REMOTE_TMP_DIR, REMOTE_PRIVATE_SOURCE, REMOTE_PRIVATE_ROOT) == private)
        _directory(REMOTE_WORKSPACE_ROOT)
        _directory(REMOTE_PRIVATE_ROOT, mode=0o700)
        _directory(REMOTE_TMP_DIR, mode=0o700)
        return {'status': 'verified_remote_storage',
                'workspace': {**workspace, 'free_bytes': _free_bytes(REMOTE_WORKSPACE_ROOT)},
                'private_state': {**private, 'free_bytes': _free_bytes(REMOTE_PRIVATE_ROOT)},
                'tmp_dir': REMOTE_TMP_DIR, 'minimum_free_bytes': MIN_FREE_BYTES}
    except (OSError, TypeError, ValueError, subprocess.SubprocessError):
        raise ValueError(_ERROR) from None
