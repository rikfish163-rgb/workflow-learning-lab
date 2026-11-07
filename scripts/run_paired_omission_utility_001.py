"""Run the full V27 paired-omission comparison on the frozen native runtime.

Fifty-four first attempts are planned, with a fresh worker and separate scorer
for each arm. Scores never stop the schedule. Execution stops if a lease,
cleanup, source verification, or other required infrastructure evidence is
unknown; no first attempt is retried, and all planned rows stay in the report.
The amax systemd parent remains independent of the initiating desktop session.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/run_native_experience_utility_001.py"
BASE_SHA256 = "aa5e8184098900f7575c81e3ab1b39218f4e9b8b5db0df62588789c87aa02876"
ID = "qwen38-paired-omission-utility-001"
PLAN = f"artifacts/{ID}-plan.json"
_MODULE = None
_ERROR = "paired_omission_utility_source_invalid"


def _isolated(*, raw=None):
    raw = public_read(ROOT, "scripts/run_native_experience_utility_001.py") if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR) from None
    source = raw.decode("utf-8")
    for before, after, count in (
        ("qwen38-native-experience-utility-001", ID, 1),
        ("artifacts/qwen38-native-experience-utility-plan-001.json", PLAN, 1),
        ("run_native_experience_utility_001", "run_paired_omission_utility_001", 2),
        ("native_experience_utility_registration_001", "paired_omission_utility_registration_001", 1),
        ("native_experience_utility_gpu_queue_001", "paired_omission_utility_gpu_queue_001", 1),
        ("native-experience-utility-001-", "paired-omission-utility-001-", 1),
        ("ordinary_control_recovery_gate", "paired_verified_write_omission_v1", 1),
        ("native-experience-001:", "paired-omission-experience-001:", 1),
    ):
        if source.count(before) != count:
            raise ValueError(_ERROR) from None
        source = source.replace(before, after)
    spec = importlib.util.spec_from_file_location("scripts._paired_omission_utility_adapter", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR) from None
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), module.__dict__)
    return module


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = _isolated()
    return _MODULE


def run_utility(plan_sha, root=ROOT, _runtime=None):
    return _get_module().run_utility(plan_sha, root=root, _runtime=_runtime)


def launch_utility(plan_sha, *, root=ROOT):
    return _get_module().launch_utility(plan_sha, root=root)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_module(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--config")
    parser.add_argument("--config-sha256")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--launch", action="store_true")
    choice.add_argument("--episode", action="store_true")
    parser.parse_args(argv)
    return _get_module().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
