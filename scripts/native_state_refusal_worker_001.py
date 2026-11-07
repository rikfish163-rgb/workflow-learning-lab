"""Native CPU worker with one durable pre-dispatch state-write refusal.

The original lifecycle, resource limits, observations and action journal are
unchanged. A separate private receipt is created only when the fault fires.
This module does not run a model, load ground truth or establish task success.
"""

import ast
import functools
import hashlib
import importlib
import importlib.util
import os
import re
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "scripts/native_appworld_worker.py": "8fe81c0e4cd8b95f0486cd2a711d3b0d6ae2004581d14468daef93079ebfc421",
    "scripts/native_appworld_tools.py": "c3afd29dc376150dca7d680e5d8e6114736ca1c9a9db1732c96ccde6cdeafa96",
    "scripts/native_state_refusal_dispatcher_001.py": "b02c8348b381e0486a4b5d2f4e647293452cbb238f8e03f543b3e22069448a7d",
}
_ERROR = "state_refusal_worker_source_invalid"


def verify_sources():
    for name, digest in SOURCE_SHA256.items():
        if hashlib.sha256(public_read(ROOT, name)).hexdigest() != digest:
            raise ValueError(_ERROR)


verify_sources()
dispatcher_module = importlib.import_module("scripts.native_state_refusal_dispatcher_001")
original_tools = importlib.import_module("scripts.native_appworld_tools")
if (Path(dispatcher_module.__file__).resolve() != ROOT / "scripts/native_state_refusal_dispatcher_001.py"
        or Path(original_tools.__file__).resolve() != ROOT / "scripts/native_appworld_tools.py"):
    raise ValueError(_ERROR)


def _build_refusal_dispatcher(apis, record):
    """Run the verified original schema builder with the bound dispatcher."""
    verify_sources()
    path = ROOT / "scripts/native_appworld_tools.py"
    raw = public_read(ROOT, "scripts/native_appworld_tools.py")
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256["scripts/native_appworld_tools.py"]:
        raise ValueError(_ERROR)
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    builders = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "build_appworld_dispatcher"]
    if len(builders) != 1 or builders[0].decorator_list:
        raise ValueError(_ERROR)
    builder = ast.get_source_segment(source, builders[0])
    if type(builder) is not str:
        raise ValueError(_ERROR)
    scope = dict(vars(original_tools))
    scope["ToolDispatcher"] = functools.partial(dispatcher_module.StateRefusalDispatcher, record=record)
    exec(compile(builder, str(path), "exec", dont_inherit=True), scope)
    return scope["build_appworld_dispatcher"](apis)


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(_ERROR)
    return source.replace(old, new, 1)


def _isolated(*, raw=None):
    """Load verified worker bytes under a non-main name, with three local hooks."""
    verify_sources()
    name = "scripts/native_appworld_worker.py"
    raw = public_read(ROOT, name) if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[name]:
        raise ValueError(_ERROR)
    source = raw.decode("utf-8")
    source = _replace_once(
        source, "        self._status_file = self._journal_file = self._runtime_file = None\n",
        "        self._status_file = self._journal_file = self._runtime_file = None\n"
        "        self._fault_dir_fd = None\n        self._fault_written = False\n",
    )
    source = _replace_once(
        source, '                self._runtime_file = _exclusive_file(run_fd, "runtime.log")\n',
        '                self._runtime_file = _exclusive_file(run_fd, "runtime.log")\n'
        "                self._fault_dir_fd = os.dup(run_fd)\n",
    )
    source = _replace_once(
        source, "            self._dispatcher = self._private_call(build_appworld_dispatcher, self._world.apis)\n",
        "            self._dispatcher = self._private_call(_build_refusal_dispatcher, self._world.apis, self._record_fault)\n",
    )
    path = ROOT / name
    spec = importlib.util.spec_from_file_location("scripts._native_state_refusal_worker_001_core", path)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    core = importlib.util.module_from_spec(spec)
    core.__dict__["_build_refusal_dispatcher"] = _build_refusal_dispatcher
    exec(compile(source, str(path), "exec", dont_inherit=True), core.__dict__)
    return core


_CORE = _isolated()
WorkerError = _CORE.WorkerError
FatalWorkerError = _CORE.FatalWorkerError
serve_stdio = _CORE.serve_stdio


class NativeAppWorldWorker(_CORE.NativeAppWorldWorker):
    def __init__(self, *args, **kwargs):
        verify_sources()
        super().__init__(*args, **kwargs)

    def _record_fault(self, receipt):
        if (self._fault_written or self._fault_dir_fd is None or type(receipt) is not dict
                or set(receipt) != {"policy", "fault_index", "action_sha256", "injected_count",
                                    "backend_dispatches_at_refusal"}
                or receipt["policy"] != dispatcher_module.POLICY
                or type(receipt["fault_index"]) is not int or receipt["fault_index"] < 0
                or type(receipt["action_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", receipt["action_sha256"], re.ASCII) is None
                or type(receipt["injected_count"]) is not int or receipt["injected_count"] != 1
                or type(receipt["backend_dispatches_at_refusal"]) is not int
                or receipt["backend_dispatches_at_refusal"] != 0):
            raise ValueError("Invalid fault provenance.")
        with _CORE._exclusive_file(self._fault_dir_fd, "fault-provenance.json") as file:
            file.write(_CORE._json_text(receipt) + "\n")
            _CORE._flush(file)
        os.fsync(self._fault_dir_fd)
        self._fault_written = True

    def close(self):
        if self._closed:
            return
        failed = False
        if self._fault_dir_fd is not None:
            try:
                os.close(self._fault_dir_fd)
            except OSError:
                failed = True
                self._mark_unknown()
            self._fault_dir_fd = None
        super().close()
        if failed:
            raise FatalWorkerError("Worker cleanup or private logging failed.")


# The base main resolves this name in its isolated globals. Install the final
# class before exposing main and before the module's own CLI can be invoked.
_CORE.NativeAppWorldWorker = NativeAppWorldWorker
main = _CORE.main


if __name__ == "__main__":
    raise SystemExit(main())
