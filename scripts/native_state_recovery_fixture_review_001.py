"""Bind reviewed experience to an authored workflow source and private writer journal."""

from scripts import native_state_recovery_review_001 as review
from scripts import run_native_state_recovery_fixture_writer_001 as writer

BUNDLE_PREFIX = "native-state-recovery-fixture-experience-001:"
SOURCE_GATE_KIND = "verified_authored_fixture_state_transition_v1"
MANIFEST = review.MANIFEST
LINEAGE = review.ReviewLineage(
    writer, SOURCE_GATE_KIND, BUNDLE_PREFIX, "native-state-recovery-fixture-review-manifest-001",
)
SOURCE_RESULT = LINEAGE.source_result


def report(writer_plan_sha256, writer_result_sha256):
    return review.report(writer_plan_sha256, writer_result_sha256, lineage=LINEAGE)


def prepare_manifest(writer_plan_sha256, writer_result_sha256):
    return review.prepare_manifest(writer_plan_sha256, writer_result_sha256, lineage=LINEAGE)


def load_reviewed_bundle(writer_plan_sha256, writer_result_sha256, review_manifest_sha256):
    return review.load_reviewed_bundle(
        writer_plan_sha256, writer_result_sha256, review_manifest_sha256, lineage=LINEAGE,
    )
