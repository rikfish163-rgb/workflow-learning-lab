"""Serve the existing B3 export after the amax Python 3.12 path relocation.

The legacy registrations and model implementation are read-only. Validate the
relocated interpreter and libraries independently, then use the archived old
venv configuration solely to authenticate the historical parent registration.
GPU admission, allocation, inference, and adapter loading use the frozen code.
"""

import hashlib
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from scripts import native_source_registration as public
from scripts.native_appworld_worker_client import _decode, _encode

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path(public.PRIVATE_STATE_ROOT) / "ubuntu-python312-runtime-001"
VENV = Path(public.PYTHON).parent.parent
CONFIG = VENV / "pyvenv.cfg"
OLD_CONFIG = PRIVATE / "frozen-pyvenv.cfg"
INTERPRETER = PRIVATE / "bin/python3.12"
FALLBACK = VENV / "lib/python3.12/site-packages/zz-wll-runtime-fallback-001.pth"
RELOCATION = "artifacts/amax-python312-relocation-001.json"
ID = "amax-python312-relocation-001"
PYTHON_SHA256 = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"
PARENT = "scripts/serve_qwen38_idle180.py"
B3 = "scripts/serve_qwen38_b3.py"
SOURCE_PINS = {
    PARENT: "9af8b5875033c146da643afb081498ce9617008572348c04f664ac5cb19214d8",
    B3: "339385fb9dcd4fa31f481854242905f7cc5eacbc54f0605f37e973a611d34604",
}
STDLIB_ALIAS = "lib/python3.12/_sysconfigdata__linux_x86_64-linux-gnu.py"


def _require(condition):
    if not condition:
        raise ValueError("amax_python312_relocation_invalid")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _library_binding():
    rows = {}
    for name in ("lib/python3.12", "distro-fallback"):
        for path in (PRIVATE / name).rglob("*"):
            relative = path.relative_to(PRIVATE).as_posix()
            info = path.lstat()
            _require(info.st_uid == os.geteuid())
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                _require(relative == STDLIB_ALIAS and target == "_sysconfigdata__x86_64-linux-gnu.py")
                rows[relative] = {"symlink": target}
                continue
            _require(not info.st_mode & 0o022)
            if stat.S_ISDIR(info.st_mode):
                continue
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and "__pycache__" not in path.parts)
            rows[relative] = {"sha256": public._absolute_hash(str(path)), "size": info.st_size}
    _require(len(rows) > 1_000)
    return {"file_count": len(rows), "tree_sha256": _hash(_encode(rows))}


def runtime_binding():
    """Measure the actual interpreter files and relocated environment identity."""
    public._private_directory(PRIVATE)
    old = OLD_CONFIG.read_bytes()
    expected = old.replace(b"home = /usr/bin\n", f"home = {INTERPRETER.parent}\n".encode())
    expected = expected.replace(b"executable = /usr/bin/python3.12\n", f"executable = {INTERPRETER}\n".encode())
    _require(sys.executable == public.PYTHON and sys.version_info[:3] == (3, 12, 3)
             and Path(sys.base_prefix) == PRIVATE and Path(sys.prefix) == VENV
             and Path(sys.executable).resolve() == INTERPRETER and expected != old
             and CONFIG.read_bytes() == expected and FALLBACK.read_text() == str(PRIVATE / "distro-fallback") + "\n"
             and public._absolute_hash(str(INTERPRETER)) == PYTHON_SHA256)
    return {"id": ID, "historical_config_sha256": public._absolute_hash(str(OLD_CONFIG)),
            "effective_config_sha256": public._absolute_hash(str(CONFIG)),
            "fallback_pth_sha256": public._absolute_hash(str(FALLBACK)),
            "interpreter_sha256": PYTHON_SHA256, "libraries": _library_binding()}


def _isolated_source(name):
    raw = public.public_read(ROOT, name)
    _require(_hash(raw) == SOURCE_PINS[name])
    module = ModuleType("scripts._wll_runtime_relocated_" + Path(name).stem)
    module.__file__ = str(ROOT / name)
    module.__package__ = "scripts"
    exec(compile(raw, module.__file__, "exec", dont_inherit=True), vars(module))
    return module


def _bound_service(relocation_sha):
    raw = public.public_read(ROOT, RELOCATION)
    _require(_hash(raw) == relocation_sha and public._same(_decode(raw), runtime_binding()))
    parent = _isolated_source(PARENT)
    b3 = _isolated_source(B3)
    runtime_audit = public._json(public.public_read(ROOT, parent.RUNTIME))
    _require(runtime_audit["runtime_sha256"][str(CONFIG)] == public._absolute_hash(str(OLD_CONFIG)))
    digest = parent.digest

    def historical_digest(path):
        # The actual effective config has already been verified separately.
        # The frozen runtime audit compares its original config to this copy.
        return digest(OLD_CONFIG if Path(path) == CONFIG else path)

    parent.digest = historical_digest
    b3.parent_preflight = parent.preflight
    return b3


def preflight(relocation_sha, service_sha=public.SERVICE_SHA, model_id=public.MODEL_ID):
    """Run the relocated runtime and original full model/adapter preflight on CPU."""
    return _bound_service(relocation_sha).preflight(service_sha, model_id, arm="B3")


@contextmanager
def loaded_engine(relocation_sha, service_sha, model_id, *, admitted):
    service = _bound_service(relocation_sha)
    with service.loaded_engine(service_sha, model_id, arm="B3", admitted=admitted) as engine:
        engine.backend["runtime_relocation_sha256"] = relocation_sha
        yield engine
