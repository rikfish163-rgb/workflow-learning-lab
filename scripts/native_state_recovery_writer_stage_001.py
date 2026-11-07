"""One extraction and, if eligible, one review of the exact proposed insertion."""

import hashlib
from functools import cache
from pathlib import Path

from scripts import native_experience_content_001 as content
from scripts import native_experience_writer_001 as writer
from scripts import native_state_recovery_writer_evidence_001 as evidence
from scripts import paired_omission_writer_evidence_001 as screening
from scripts.native_source_registration import public_read
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
CONTENT = "scripts/native_experience_content_001.py"
CONTENT_SHA256 = "d08b48932b5e82368b6674501b918881bfa710e2e189e1d798ad2abe8ac56334"
WRITER_SEED, REVIEW_SEED = 17, 18
WRITER_MAX_TOKENS, REVIEW_MAX_TOKENS = 128, 256


@cache
def _review_once():
    raw = public_read(ROOT, CONTENT)
    if hashlib.sha256(raw).hexdigest() != CONTENT_SHA256:
        raise ValueError("native_state_recovery_review_source_invalid")
    text = raw.decode("utf-8")
    edits = (
        ("set(payload) == {\"history\", \"candidate\", \"insertion\"}",
         "set(payload) == {\"observed_state_transition_cases\", \"candidate\", \"insertion\"}"),
        ('        _history([{"role": "system", "content": WRITER_SYSTEM},\n'
         '                  {"role": "user", "content": _json({"history": payload["history"]})}])',
         '        _validate_cases(payload["observed_state_transition_cases"])'),
        ('        history_length = len(payload["history"])',
         '        case_count = len(payload["observed_state_transition_cases"])'),
        ("verdict = parse_review(text, history_length=history_length)",
         "verdict = parse_review(text, case_count=case_count)"),
    )
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError("native_state_recovery_review_source_invalid")
        text = text.replace(before, after)
    namespace = {"__name__": "scripts._native_state_recovery_review_001"}
    exec(compile(text, str(ROOT / CONTENT), "exec", dont_inherit=True), namespace)
    namespace.update(_REVIEW_SYSTEM=evidence.REVIEW_SYSTEM, _validate_cases=evidence._validate_cases,
                     parse_review=evidence.parse_review)
    return namespace["review_once"]


def run(cases, *, model_factory, guard, record):
    """Run the prequalified private input through the admitted model client.

    The coordinator owns admission, source qualification and release. All
    model-bearing journal records and the returned candidate stay private.
    """
    writer_messages = evidence.writer_messages(cases)
    review_once = _review_once()

    def record_writer(event):
        if event["event"] == "generation_intent":
            guard()
        record({"stage": "writer", **event})

    def record_review(event):
        if event["event"] == "generation_intent":
            guard()
        record({"stage": "review", **event})

    guard()
    draft = writer.write_once(writer_messages, model_factory(WRITER_MAX_TOKENS),
                              seed=WRITER_SEED, record=record_writer)
    result = {"status": draft["status"], "writer": draft, "review": None,
              "screen": None, "insertion_sha256": None, "training_authorized": False,
              "utility_authorized": False}
    if draft["status"] != "candidate_unvalidated":
        return result
    candidate = draft["candidate"]
    flags = screening._screen_flags(candidate)
    result["screen"] = {"passed": not flags, "reason_codes": flags}
    record({"event": "candidate_screened", **result["screen"]})
    if flags:
        result["status"] = "structurally_rejected"
        return result
    review_messages = evidence.review_messages(cases, candidate)
    insertion_sha = hashlib.sha256(content.insertion(candidate).encode()).hexdigest()
    result["insertion_sha256"] = insertion_sha
    record({"event": "review_binding", "insertion_sha256": insertion_sha,
            "messages_sha256": _message_payload(review_messages)[1]})
    guard()
    review = review_once(review_messages, model_factory(REVIEW_MAX_TOKENS),
                         seed=REVIEW_SEED, record=record_review)
    result.update(status=review["status"], review=review)
    return result
