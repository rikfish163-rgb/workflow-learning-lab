"""Audit the ordered-evidence writer registered after the pre-entry failure."""

from dataclasses import replace

from scripts import native_state_recovery_transition_review_001 as base
from scripts import run_native_state_recovery_transition_writer_002 as writer

SOURCE_GATE_KIND = base.SOURCE_GATE_KIND
BUNDLE_PREFIX = "native-state-recovery-transition-experience-002:"
MANIFEST = base.MANIFEST
LINEAGE = replace(base.LINEAGE, writer=writer, bundle_prefix=BUNDLE_PREFIX,
                  manifest_id="native-state-recovery-transition-review-manifest-002")
SOURCE_RESULT = LINEAGE.source_result


def report(writer_plan_sha, writer_result_sha):
    return base._auditor().report(writer_plan_sha, writer_result_sha, lineage=LINEAGE)


def prepare_manifest(writer_plan_sha, writer_result_sha):
    return base._auditor().prepare_manifest(writer_plan_sha, writer_result_sha, lineage=LINEAGE)


def load_reviewed_bundle(writer_plan_sha, writer_result_sha, review_manifest_sha):
    return base._auditor().load_reviewed_bundle(writer_plan_sha, writer_result_sha, review_manifest_sha, lineage=LINEAGE)
