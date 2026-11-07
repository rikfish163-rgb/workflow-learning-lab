"""Classify one authenticated native, actor-visible recovery sequence.

This pure checker performs no API calls, worker operations, scoring or model
execution. The caller must independently authenticate the complete actor/worker
journals, their projection into these records, and the fault receipt. A
qualified local pattern is not authentication, task success, durable-state proof,
training authorization, or evidence that a lesson transfers.
"""

import re

from scripts import verification_readonly_registry_v3 as readonly
from scripts.native_appworld_replay import _observation
from scripts.native_appworld_worker_client import _decode, _same
from wll.protocol import fingerprint

POLICY = "native_state_recovery_predispatch_v1"
FAULT_MESSAGE = "Controlled transient tool refusal: request was not executed."
MAX_RECORDS = 1024
MAX_ACTION_BYTES = 65536
_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z", re.ASCII)
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_FAULT_FIELDS = frozenset({"policy", "fault_index", "action_sha256", "injected_count",
                           "backend_dispatches_at_refusal"})
_PAIRS = {
    "todoist__update_task": ("todoist__show_task", "task_id", "is_completed"),
    "simple_note__update_note": ("simple_note__show_note", "note_id", "pinned"),
}
_COUNT_FIELDS = frozenset({"records", "faults", "write_attempts_after_fault", "matching_reads_before",
                           "matching_reads_after"})
_REASONS = frozenset({
    "verified_transition", "invalid_records", "invalid_provenance", "no_fault", "fault_action_mismatch",
    "fault_observation_mismatch", "unsupported_fault_action", "missing_before_read", "invalid_before_read",
    "before_read_failed", "before_already_satisfied", "missing_correction", "changed_correction",
    "repeated_write", "correction_failed", "invalid_correction_result", "missing_after_read",
    "invalid_after_read", "after_read_failed", "postcondition_not_met", "ambiguous_intervening_action",
})


def _result(status, reason, counts, private=None):
    return {"status": status, "reason": reason, "counts": dict(counts), "private_evidence": private}


def _decode_records(records):
    if type(records) is not list or len(records) > MAX_RECORDS:
        raise ValueError
    decoded = []
    for row in records:
        if type(row) is not dict or set(row) != {"action", "observation"} or type(row["action"]) is not str:
            raise ValueError
        observation = row["observation"]
        _observation(observation)
        raw = row["action"].encode("utf-8")
        try:
            if len(raw) > MAX_ACTION_BYTES:
                raise ValueError
            value = _decode(raw)
            if (type(value) is not dict or set(value) != {"name", "arguments"}
                    or type(value["name"]) is not str or value["name"].count("__") != 1
                    or type(value["arguments"]) is not dict):
                raise ValueError
            app, api = value["name"].split("__")
            if _IDENTIFIER.fullmatch(app) is None or _IDENTIFIER.fullmatch(api) is None:
                raise ValueError
        except (TypeError, ValueError, RuntimeError, UnicodeError, RecursionError, OverflowError):
            # The caller must authenticate native worker counters: this exact
            # actor-visible error is then known to precede backend dispatch.
            if observation.get("error_type") != "ToolDispatchError":
                raise ValueError from None
            value = None
        decoded.append((value, observation))
    return decoded


def _valid_fault(fault):
    if (type(fault) is not dict or set(fault) != _FAULT_FIELDS or type(fault["policy"]) is not str
            or fault["policy"] != POLICY):
        return False
    count = fault["injected_count"]
    if type(count) is not int or count not in (0, 1):
        return False
    if count == 0:
        return (fault["fault_index"] is None and fault["action_sha256"] is None
                and fault["backend_dispatches_at_refusal"] is None)
    return (type(fault["fault_index"]) is int and fault["fault_index"] >= 0
            and type(fault["action_sha256"]) is str and _SHA.fullmatch(fault["action_sha256"]) is not None
            and type(fault["backend_dispatches_at_refusal"]) is int
            and fault["backend_dispatches_at_refusal"] == 0)


def _target_write(value):
    definition = _PAIRS.get(value["name"])
    if definition is None:
        return None
    _read, identifier, field = definition
    args = value["arguments"]
    if (set(args) != {"access_token", identifier, field} or type(args["access_token"]) is not str
            or not args["access_token"] or type(args[identifier]) is not int or args[identifier] <= 0
            or type(args[field]) is not bool):
        return None
    return definition


def _matching_read(value, name, identifier, arguments):
    return (value is not None and value["name"] == name
            and set(value["arguments"]) == {"access_token", identifier}
            and _same(value["arguments"]["access_token"], arguments["access_token"])
            and _same(value["arguments"][identifier], arguments[identifier]))


def _read_value(observation, identifier, field, arguments):
    if not observation["ok"]:
        return "failed", None
    value = observation["result"]
    if (type(value) is not dict or type(value.get(identifier)) is not int
            or not _same(value[identifier], arguments[identifier]) or type(value.get(field)) is not bool):
        return "invalid", None
    return "known", value[field]


def _safe_intervening_read(value, observation):
    if observation.get("error_type") == "ToolDispatchError":
        return True
    if value is None:
        return False
    app, api = value["name"].split("__")
    if (app, api) not in readonly.ALLOWED_APIS:
        return False
    try:
        readonly.validate_inspection_arguments(app, api, value["arguments"])
        if observation["ok"]:
            readonly.validate_inspection_response(app, api, observation["result"])
        return True
    except (TypeError, ValueError, KeyError, ArithmeticError, RecursionError):
        return False


def classify(records, fault_provenance):
    """Return a private classification of the complete projected actor journal.

    Each record contains exactly an action JSON string and its actor-visible
    observation. The fault binding is a separately authenticated receipt with
    zero-based ``fault_index`` and ``action_sha256`` equal to the project's
    ``fingerprint`` of the full decoded action. Matching correction actions
    compare all decoded fields with exact JSON types, including the token.
    """
    counts = {"records": len(records) if type(records) is list else 0, "faults": 0,
              "write_attempts_after_fault": 0, "matching_reads_before": 0, "matching_reads_after": 0}
    try:
        decoded = _decode_records(records)
    except (TypeError, ValueError, RuntimeError, UnicodeError, RecursionError, OverflowError):
        return _result("unknown", "invalid_records", counts)
    if not _valid_fault(fault_provenance):
        return _result("unknown", "invalid_provenance", counts)
    if fault_provenance["injected_count"] == 0:
        return _result("ineligible", "no_fault", counts)
    counts["faults"] = 1
    fault_index = fault_provenance["fault_index"]
    if fault_index >= len(decoded):
        return _result("unknown", "fault_action_mismatch", counts)
    blocked, fault_observation = decoded[fault_index]
    if blocked is None or fingerprint(blocked) != fault_provenance["action_sha256"]:
        return _result("unknown", "fault_action_mismatch", counts)
    if not _same(fault_observation, {"ok": False, "error_type": "ToolDispatchError", "message": FAULT_MESSAGE}):
        return _result("unknown", "fault_observation_mismatch", counts)
    definition = _target_write(blocked)
    if definition is None:
        return _result("ineligible", "unsupported_fault_action", counts)
    read_name, identifier, field = definition
    args = blocked["arguments"]
    before = [i for i, (value, _obs) in enumerate(decoded[:fault_index])
              if _matching_read(value, read_name, identifier, args)]
    counts["matching_reads_before"] = len(before)
    if not before:
        return _result("ineligible", "missing_before_read", counts)
    before_index = before[-1]
    state, before_value = _read_value(decoded[before_index][1], identifier, field, args)
    if state != "known":
        return _result("unknown" if state == "invalid" else "ineligible", state + "_before_read"
                       if state == "invalid" else "before_read_failed", counts)
    if before_value is args[field]:
        return _result("ineligible", "before_already_satisfied", counts)
    writes = [i for i in range(fault_index + 1, len(decoded))
              if decoded[i][0] is not None and decoded[i][0]["name"] == blocked["name"]]
    counts["write_attempts_after_fault"] = len(writes)
    if not writes:
        return _result("ineligible", "missing_correction", counts)
    if len(writes) != 1:
        return _result("ineligible", "repeated_write", counts)
    correction_index = writes[0]
    correction, correction_observation = decoded[correction_index]
    if not _same(correction, blocked):
        return _result("ineligible", "changed_correction", counts)
    if not correction_observation["ok"]:
        return _result("ineligible", "correction_failed", counts)
    result = correction_observation["result"]
    if type(result) is not dict or type(result.get("message")) is not str:
        return _result("unknown", "invalid_correction_result", counts)
    after = [i for i in range(correction_index + 1, len(decoded))
             if _matching_read(decoded[i][0], read_name, identifier, args)]
    counts["matching_reads_after"] = len(after)
    if not after:
        return _result("ineligible", "missing_after_read", counts)
    after_index = after[0]
    for index in after:
        state, after_value = _read_value(decoded[index][1], identifier, field, args)
        if state != "known":
            return _result("unknown" if state == "invalid" else "ineligible", state + "_after_read"
                           if state == "invalid" else "after_read_failed", counts)
        if after_value is not args[field]:
            return _result("ineligible", "postcondition_not_met", counts)
    for i in range(before_index + 1, after_index):
        if i not in (fault_index, correction_index) and not _safe_intervening_read(*decoded[i]):
            return _result("ineligible", "ambiguous_intervening_action", counts)
    app = blocked["name"].split("__")[0]
    for i in range(after_index + 1, len(decoded)):
        value, observation = decoded[i]
        if (value is not None and value["name"].startswith(app + "__")
                and _same(value["arguments"].get(identifier), args[identifier])
                and not _safe_intervening_read(value, observation)):
            return _result("ineligible", "ambiguous_intervening_action", counts)
    indices = {"before": before_index, "fault": fault_index, "correction": correction_index, "after": after_index}
    private = {"indices": indices, "record_sha256": {key: fingerprint(records[i]) for key, i in indices.items()},
               "decoded_action_sha256": fingerprint(blocked), "fault_provenance_sha256": fingerprint(fault_provenance)}
    return _result("qualified", "verified_transition", counts, private)


def public_projection(result):
    """Expose only fixed category labels and integer counts, never private hashes or values."""
    if (type(result) is not dict or set(result) != {"status", "reason", "counts", "private_evidence"}
            or type(result["status"]) is not str or result["status"] not in {"qualified", "ineligible", "unknown"}
            or type(result["reason"]) is not str or result["reason"] not in _REASONS
            or type(result["counts"]) is not dict or set(result["counts"]) != _COUNT_FIELDS
            or any(type(value) is not int or value < 0 for value in result["counts"].values())):
        raise ValueError("native_state_recovery_public_projection_invalid") from None
    return {"status": result["status"], "reason": result["reason"], "counts": dict(result["counts"])}
