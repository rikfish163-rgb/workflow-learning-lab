"""Apply the predeclared V27 panel to a reviewed ordered-evidence lesson."""

from scripts import native_state_recovery_transition_review_002 as review
from scripts import report_native_state_recovery_utility_001 as reporter
from scripts import run_native_state_recovery_utility_001 as base

LINEAGE = base.UtilityLineage(
    "native-state-recovery-transition-utility-001",
    "scripts.run_native_state_recovery_transition_utility_001",
    review,
    4,
)


def build_registration(root, proof_sha, *, private=False):
    return base.build_registration(root, proof_sha, private=private, lineage=LINEAGE)


def _verification_payload(spec):
    return base.bindings(LINEAGE)[2]._verification_payload(spec)


def report(plan_sha, result_sha):
    return reporter.report(plan_sha, result_sha, lineage=LINEAGE)


if __name__ == "__main__":
    raise SystemExit(base.main(lineage=LINEAGE))
