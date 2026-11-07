"""A fresh fixture probe with per-case private directories created before launch."""

import hashlib
from pathlib import Path
from types import ModuleType

from scripts.native_appworld_worker_client import WORKSPACE
from scripts.native_source_registration import PRIVATE_STATE_ROOT, public_read

PRIOR = "scripts/probe_native_state_recovery_fixture_001.py"
PRIOR_SHA256 = "ef32d8fc77ea0cb39e6902df42c465a91cf5eee70c23b41d1686a1d5750dbec8"
ID = "native-state-recovery-fixture-CPU-002"


def main():
    raw = public_read(WORKSPACE, PRIOR)
    if hashlib.sha256(raw).hexdigest() != PRIOR_SHA256:
        raise ValueError("fixture_source_invalid")
    module = ModuleType("scripts._native_state_recovery_fixture_002")
    module.__file__ = __file__
    exec(compile(raw, str(WORKSPACE / PRIOR), "exec", dont_inherit=True), vars(module))
    module.ID = ID
    module.PRIVATE = Path(PRIVATE_STATE_ROOT) / ID
    module.PROOF = WORKSPACE / "artifacts" / f"{ID}.json"
    original = module._case

    def run_case(task, spec):
        (module.PRIVATE / spec[0]).mkdir(mode=0o700)
        return original(task, spec)

    module._case = run_case
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
