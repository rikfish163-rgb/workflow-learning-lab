"""Authenticate the ordered-evidence writer and its exact reviewed insertion."""

from functools import cache
from types import FunctionType, SimpleNamespace

from scripts import native_state_recovery_review_001 as base
from scripts import native_state_recovery_transition_stage_001 as stage
from scripts import run_native_state_recovery_transition_writer_001 as writer
from scripts.native_source_registration import _public_hash

SOURCE_GATE_KIND = "verified_ordered_fixture_state_transition_v1"
BUNDLE_PREFIX = "native-state-recovery-transition-experience-001:"
MANIFEST = base.MANIFEST
LINEAGE = base.ReviewLineage(writer, SOURCE_GATE_KIND, BUNDLE_PREFIX, "native-state-recovery-transition-review-manifest-001")
SOURCE_RESULT = LINEAGE.source_result
BASE_PATH = "scripts/native_state_recovery_review_001.py"
BASE_SHA = "8645b462af378382b55f29faeeb7c0ec9cd0bb6850a9541cd9d43c3edc04c0da"


@cache
def _auditor():
    base._require(_public_hash(stage.old_stage.ROOT, BASE_PATH) == BASE_SHA)
    scope = {**vars(base), "evidence": stage,
             "stage": SimpleNamespace(**{**vars(base.stage), "run": stage.run})}
    for name in ("replay", "_terminal_writer", "report", "prepare_manifest", "load_reviewed_bundle"):
        function = getattr(base, name)
        scope[name] = FunctionType(function.__code__, scope, name)
    return SimpleNamespace(**scope)


def report(writer_plan_sha, writer_result_sha):
    return _auditor().report(writer_plan_sha, writer_result_sha, lineage=LINEAGE)


def prepare_manifest(writer_plan_sha, writer_result_sha):
    return _auditor().prepare_manifest(writer_plan_sha, writer_result_sha, lineage=LINEAGE)


def load_reviewed_bundle(writer_plan_sha, writer_result_sha, review_manifest_sha):
    return _auditor().load_reviewed_bundle(writer_plan_sha, writer_result_sha, review_manifest_sha, lineage=LINEAGE)
