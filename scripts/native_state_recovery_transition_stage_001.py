"""Use ordered state observations with the frozen writer and review protocol."""

from functools import cache
from types import FunctionType, SimpleNamespace

from scripts import native_state_recovery_writer_evidence_001 as old_evidence
from scripts import native_state_recovery_writer_stage_001 as old_stage
from scripts.native_appworld_worker_client import _encode, _same
from scripts.native_source_registration import _public_hash

STAGE_SHA256 = "dbaf1e28bbefa1628e8c880b09116950e4e3a710e77fec24d9ea885a3db28884"
STAGE_PATH = "scripts/native_state_recovery_writer_stage_001.py"
WRITER_SEED, REVIEW_SEED = old_stage.WRITER_SEED, old_stage.REVIEW_SEED
WRITER_MAX_TOKENS, REVIEW_MAX_TOKENS = old_stage.WRITER_MAX_TOKENS, old_stage.REVIEW_MAX_TOKENS


def _validate_cases(cases):
    old_evidence._require(type(cases) is list and len(cases) == 4)
    families, initial_values = set(), set()
    for case in cases:
        old_evidence._require(type(case) is dict and set(case) == {"operation_family", "resource", "events"}
                              and case["operation_family"] in ("A", "B") and case["resource"] == "target")
        events = case["events"]
        old_evidence._require(type(events) is list and len(events) == 4 and type(events[0]) is dict)
        initial = events[0].get("observed_value")
        old_evidence._require(type(initial) is bool)
        desired = not initial
        expected = [
            {"event": "read", "observed_value": initial},
            {"event": "write", "requested_value": desired, "outcome": "explicitly_not_executed"},
            {"event": "write", "requested_value": desired, "same_call": True, "outcome": "acknowledged"},
            {"event": "read", "observed_value": desired},
        ]
        old_evidence._require(_same(events, expected))
        families.add(case["operation_family"])
        initial_values.add(initial)
    old_evidence._require(families == {"A", "B"} and initial_values == {True, False})


def writer_messages(cases):
    _validate_cases(cases)
    return [{"role": "system", "content": old_evidence.WRITER_SYSTEM},
            {"role": "user", "content": _encode({"observed_state_transition_cases": cases}).decode("ascii")}]


def review_messages(cases, candidate):
    _validate_cases(cases)
    candidate = old_stage.content._candidate(candidate)
    old_evidence._require(not old_stage.screening._screen_flags(candidate))
    payload = {"observed_state_transition_cases": cases, "candidate": candidate,
               "insertion": old_stage.content.insertion(candidate)}
    return [{"role": "system", "content": old_evidence.REVIEW_SYSTEM},
            {"role": "user", "content": _encode(payload).decode("ascii")}]


@cache
def _engine():
    old_evidence._require(_public_hash(old_stage.ROOT, STAGE_PATH) == STAGE_SHA256)
    scope = {**vars(old_stage), "evidence": SimpleNamespace(
        writer_messages=writer_messages, review_messages=review_messages, _validate_cases=_validate_cases,
        REVIEW_SYSTEM=old_evidence.REVIEW_SYSTEM, parse_review=old_evidence.parse_review,
    )}
    scope["_review_once"] = cache(FunctionType(old_stage._review_once.__wrapped__.__code__, scope))
    return FunctionType(old_stage.run.__code__, scope, old_stage.run.__name__)


def run(cases, *, model_factory, guard, record):
    return _engine()(cases, model_factory=model_factory, guard=guard, record=record)
