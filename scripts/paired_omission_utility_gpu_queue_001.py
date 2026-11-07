"""Fourteen isolated utility leases using the frozen native queue fence.

Import and job construction do not touch the queue, model, or GPU.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_experience_utility_gpu_queue_001.py"
BASE_SHA256 = "6bcac4be518771fab651ceb3697e1c31fe870e3b4a8b6ed8939f81c9c3733ff3"
ID = "qwen38-paired-omission-utility-001"
PLAN = f"artifacts/{ID}-plan.json"
GUARD_NAME = "paired-omission-utility-operations-001.lock"
REMOTE = Path("/home/hts/workflow-learning-lab")
PORT = 18997
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
N_BATCHES = 14
_CORE = None
_ERROR = "paired_omission_utility_queue_source_invalid"


def _isolated(*, raw=None):
    raw = public_read(ROOT, "scripts/native_experience_utility_gpu_queue_001.py") if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR) from None
    source = raw.decode("utf-8")
    for before, after, count in (
        ("qwen38-native-experience-utility-001", ID, 1),
        ("artifacts/qwen38-native-experience-utility-plan-001.json", PLAN, 1),
        ("native-experience-utility-operations-001.lock", GUARD_NAME, 1),
        ("native_experience_utility_gpu_queue_001", "paired_omission_utility_gpu_queue_001", 2),
        ("native_experience_utility_registration_001", "paired_omission_utility_registration_001", 1),
        ("scripts._native_experience_utility_queue_base", "scripts._paired_omission_utility_queue_base", 1),
        ("PORT = 18991", "PORT = 18997", 1),
        ("    core = parent._isolated()\n", "    core = parent._isolated()\n    core.PORT = PORT\n", 1),
    ):
        if source.count(before) != count:
            raise ValueError(_ERROR) from None
        source = source.replace(before, after)
    spec = importlib.util.spec_from_file_location("scripts._paired_omission_utility_queue_adapter", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR) from None
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), module.__dict__)
    return module._isolated()


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args(argv)
    core = _get_core()
    return core.serve(core.make_job(args.batch, args.owner, args.plan_sha256))


if __name__ == "__main__":
    main()
