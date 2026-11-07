"""Pure structural reconciliation of private actor-004 generation journals.

Only a returned usage record or a validated not-started record closes the one
pending intent. Missing cancellation writes remain unknown; outcome counters
cannot erase them. This validates the persisted event contract, not the origin
of a proof, an exception class, or an actor receipt. Returned token totals are
lower bounds only. No request, message or response content leaves this helper.
"""

import math
import re

from scripts.verification_token_counter import TokenCountError, _message_payload

_ERROR = "native_generation_accounting_invalid"
_INTENT_FIELDS = frozenset({"event", "messages", "seed", "prompt_tokens", "max_new_tokens"})
_CANCEL_FIELDS = frozenset({
    "event", "proof_source", "reason", "seed", "prompt_tokens", "messages_sha256",
    "request_sha256", "intent_monotonic", "observed_monotonic", "validated_monotonic",
    "model_deadline", "actor_deadline", "request_window_seconds", "journal_reserve_seconds",
    "transport_started", "cost_counter_decremented",
})
_NUMERIC_PROOF_FIELDS = (
    "intent_monotonic", "observed_monotonic", "validated_monotonic", "model_deadline",
    "actor_deadline", "request_window_seconds", "journal_reserve_seconds",
)
_HASH = re.compile(r"[a-f0-9]{64}\Z")


def _require(condition):
    if not condition:
        raise ValueError(_ERROR) from None


def _positive_integer(value):
    return type(value) is int and value > 0


def _nonnegative_finite(value):
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _intent(row):
    _require(set(row) == _INTENT_FIELDS
             and type(row["seed"]) is int and 0 <= row["seed"] < 2**31
             and _positive_integer(row["prompt_tokens"])
             and _positive_integer(row["max_new_tokens"]) and row["max_new_tokens"] <= 1024)
    try:
        digest = _message_payload(row["messages"])[1]
    except TokenCountError:
        raise ValueError(_ERROR) from None
    return row["seed"], row["prompt_tokens"], row["max_new_tokens"], digest


def _returned(row, pending):
    _require(pending is not None and set(row) == {"event", "response"})
    response = row["response"]
    _require(type(response) is dict
             and _positive_integer(response.get("prompt_tokens"))
             and _positive_integer(response.get("completion_tokens"))
             and response["prompt_tokens"] == pending[1]
             and response["completion_tokens"] <= pending[2])
    return response["prompt_tokens"], response["completion_tokens"]


def _cancelled(row, pending):
    _require(pending is not None and set(row) == _CANCEL_FIELDS)
    seed, prompt_tokens, _, digest = pending
    _require(type(row["seed"]) is int and row["seed"] == seed
             and type(row["prompt_tokens"]) is int and row["prompt_tokens"] == prompt_tokens
             and type(row["messages_sha256"]) is str and row["messages_sha256"] == digest
             and all(_nonnegative_finite(row[name]) for name in _NUMERIC_PROOF_FIELDS)
             and row["intent_monotonic"] <= row["observed_monotonic"] <= row["validated_monotonic"]
             and 0 < row["model_deadline"] <= row["actor_deadline"]
             and row["request_window_seconds"] == 120.0
             and row["journal_reserve_seconds"] == 5.0
             and row["transport_started"] is False)
    if row["proof_source"] == "actor_before_model_call":
        _require(row["reason"] == "wall_seconds" and row["request_sha256"] is None
                 and row["cost_counter_decremented"] is False
                 and row["observed_monotonic"] == row["validated_monotonic"]
                 and row["observed_monotonic"] >= row["actor_deadline"])
    else:
        _require(row["proof_source"] == "client_before_transport"
                 and row["reason"] == "request_window_seconds"
                 and type(row["request_sha256"]) is str
                 and _HASH.fullmatch(row["request_sha256"]) is not None
                 and row["cost_counter_decremented"] is True
                 and row["model_deadline"] - row["observed_monotonic"] < 120.0)


def generation_accounting(journal):
    """Return content-free counts, rejecting malformed or ambiguous ordering.

    A generation identity is its seed and complete-message fingerprint. It may
    occur only once, so replaying closed intent/result or intent/cancel blocks
    cannot forge counts. Ordinary named events are ignored, including actor
    outcome counters. A final unmatched intent contributes one to the unknown
    upper bound; its token usage is not invented or subtracted.
    """
    _require(type(journal) in (list, tuple))
    intents = results = cancelled = prompt_tokens = completion_tokens = 0
    pending = None
    seen = set()
    for row in journal:
        _require(type(row) is dict and type(row.get("event")) is str)
        event = row["event"]
        if event == "generation_intent":
            _require(pending is None)
            pending = _intent(row)
            identity = pending[0], pending[3]
            _require(identity not in seen)
            seen.add(identity)
            intents += 1
        elif event == "generation_result":
            prompt, completion = _returned(row, pending)
            prompt_tokens += prompt
            completion_tokens += completion
            results += 1
            pending = None
        elif event == "generation_not_started":
            _cancelled(row, pending)
            cancelled += 1
            pending = None
    return {
        "generation_intents": intents, "generation_results": results,
        "confirmed_not_started": cancelled,
        "unreturned_generation_intents_upper_bound": intents - results - cancelled,
        "journal_returned_prompt_tokens_lower_bound": prompt_tokens,
        "journal_returned_completion_tokens_lower_bound": completion_tokens,
        "generation_attempts_after_cancellations": intents - cancelled,
    }
