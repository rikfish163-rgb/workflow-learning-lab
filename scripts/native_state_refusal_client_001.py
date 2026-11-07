"""Client 006 controls with a source-bound state-refusal CPU worker launch.

All RPC, timeout, PID/unit, journal, freshness and snapshot checks are inherited.
Fault provenance is a separate private file for the episode collector to bind
to the complete action and worker histories after the worker exits.
"""

import ast
import hashlib
import importlib
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "scripts/native_appworld_worker_client.py": "16eac615ae5d9f2e143bb86b43ade4435ad0114e3bb626c3a1c90533e7d978dc",
    "scripts/native_appworld_worker_client_006.py": "4db3a03cce104a71a14d547ca2cd23f4c02757a8f0d4713b53073c4a6dca4980",
    "scripts/native_appworld_worker.py": "8fe81c0e4cd8b95f0486cd2a711d3b0d6ae2004581d14468daef93079ebfc421",
    "scripts/native_appworld_tools.py": "c3afd29dc376150dca7d680e5d8e6114736ca1c9a9db1732c96ccde6cdeafa96",
    "scripts/native_state_refusal_dispatcher_001.py": "b02c8348b381e0486a4b5d2f4e647293452cbb238f8e03f543b3e22069448a7d",
    "scripts/native_state_refusal_worker_001.py": "ce95fcd9873fe6b0c8929ccf33d54a6d6b938f7b78d7106acc28aadadbfa83c3",
}
_ERROR = "state_refusal_client_source_invalid"


def verify_sources():
    for name, digest in SOURCE_SHA256.items():
        if hashlib.sha256(public_read(ROOT, name)).hexdigest() != digest:
            raise ValueError(_ERROR)


verify_sources()
original = importlib.import_module("scripts.native_appworld_worker_client")
client6 = importlib.import_module("scripts.native_appworld_worker_client_006")
if (Path(original.__file__).resolve() != ROOT / "scripts/native_appworld_worker_client.py"
        or Path(client6.__file__).resolve() != ROOT / "scripts/native_appworld_worker_client_006.py"):
    raise ValueError(_ERROR)
NativeWorkerClientError = original.NativeWorkerClientError


def _isolated_init(*, raw=None):
    verify_sources()
    name = "scripts/native_appworld_worker_client.py"
    raw = public_read(ROOT, name) if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[name]:
        raise ValueError(_ERROR)
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeWorkerClient"]
    if len(classes) != 1:
        raise ValueError(_ERROR)
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError(_ERROR)
    initializer = ast.get_source_segment(source, methods[0])
    old = "from scripts.native_appworld_worker import main;raise SystemExit(main())"
    new = "from scripts.native_state_refusal_worker_001 import main;raise SystemExit(main())"
    if type(initializer) is not str or initializer.count(old) != 1:
        raise ValueError(_ERROR)
    initializer = initializer.replace(old, new, 1)
    scope = dict(vars(original))
    exec(compile(initializer, str(ROOT / name), "exec", dont_inherit=True), scope)
    return scope["__init__"], scope


_INITIALIZE, _INIT_GLOBALS = _isolated_init()


class NativeWorkerClient(client6.NativeWorkerClient):
    def __init__(self, task_id, experiment_name, run_dir, *, max_actions,
                 max_api_requests, rpc_timeout, wall_seconds):
        verify_sources()
        _INITIALIZE(self, task_id, experiment_name, run_dir, max_actions=max_actions,
                    max_api_requests=max_api_requests, rpc_timeout=rpc_timeout,
                    wall_seconds=wall_seconds)
