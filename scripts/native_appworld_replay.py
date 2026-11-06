"""Trusted, private prefix replay against an already fresh native JSON worker.

The caller creates a distinct constrained process and private namespace for
each paired branch. This helper checks zero-cost fresh metadata, replays only
the protected action strings, and compares complete actor-visible observations.
It does not create, save, close, score, or reset a worker, generate a prompt, or
import task ground truth. The caller always owns cleanup and the common wall
deadline, and must configure the worker with these same explicit parent limits.

``record`` must synchronously append and fsync each complete private event or
raise. Returned metadata and matched prefix records are also private; only a
fully matched result may be passed to a suffix actor. Confirmed worker totals
remain separate from RPC attempts whose effects or costs may be unknown.
"""

import copy
import json
import math
import time
from typing import Any

if __package__:
    from .native_appworld_tools import (
        _check_depth,
        _check_json_values,
        _reject_constant,
        _unique_object,
    )
else:
    from native_appworld_tools import (
        _check_depth,
        _check_json_values,
        _reject_constant,
        _unique_object,
    )

MAX_ACTION_BYTES = 65536
_MAX_JSON_DEPTH = 32
_MAX_RPC_BYTES = 6 * MAX_ACTION_BYTES + 4096
_COST_KEYS = {"action_attempts", "api_invocations", "upstream_requests"}
_FRESH_KEYS = {
    "last_successful_save_through_attempt", "current_state_saved", "score_eligible",
    "outcome_unknown", "unknown_cost",
}


class _JournalError(Exception):
    pass


class _ClockError(Exception):
    pass


class _ReplyError(Exception):
    pass


def _canonical(value: Any) -> str:
    """Validate JSON without coercion, then compare its full canonical text."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"))
    if len(text) > _MAX_RPC_BYTES:
        raise ValueError
    _check_depth(text, _MAX_JSON_DEPTH)
    _check_json_values(value)
    pending = [value]
    while pending:
        current = pending.pop()
        if type(current) is dict:
            if any(type(key) is not str for key in current):
                raise ValueError
            pending.extend(current.values())
        elif type(current) is list:
            pending.extend(current)
    return text


def _metadata(value: Any) -> None:
    if (type(value) is not dict or
            set(value) != {"instruction", "supervisor", "app_descriptions"}):
        raise ValueError
    supervisor, descriptions = value["supervisor"], value["app_descriptions"]
    if (type(value["instruction"]) is not str or type(supervisor) is not dict or
            set(supervisor) - {"first_name", "last_name", "email", "phone_number"} or
            any(type(item) is not str for item in supervisor.values()) or
            type(descriptions) is not dict or
            any(type(key) is not str or type(item) is not str
                for key, item in descriptions.items())):
        raise ValueError
    _canonical(value)


def _observation(value: Any) -> None:
    if type(value) is not dict or type(value.get("ok")) is not bool:
        raise ValueError
    if value["ok"]:
        if set(value) != {"ok", "result"}:
            raise ValueError
    else:
        fields = {"ok", "error_type", "message"}
        if value.get("error_type") == "ObservedAPIError":
            fields.add("status_code")
            if type(value.get("status_code")) is not int or not 400 <= value["status_code"] < 500:
                raise ValueError
        elif value.get("error_type") != "ToolDispatchError":
            raise ValueError
        if set(value) != fields or type(value["message"]) is not str:
            raise ValueError
    _canonical(value)


def _inputs(source_metadata, prefix_records, record, deadline, max_actions, max_api_requests, clock):
    try:
        if any(type(limit) is not int or limit < 1 for limit in (max_actions, max_api_requests)):
            raise ValueError
        if (type(deadline) not in (int, float) or not math.isfinite(deadline) or
                not callable(record) or not callable(clock)):
            raise ValueError
        _metadata(source_metadata)
        if type(prefix_records) is not list or len(prefix_records) > max_actions:
            raise ValueError
        source_invocations = 0
        for item in prefix_records:
            if type(item) is not dict or set(item) != {"action", "observation"}:
                raise ValueError
            action, expected = item["action"], item["observation"]
            if (type(action) is not str or len(action) > MAX_ACTION_BYTES or
                    len(action.encode("utf-8")) > MAX_ACTION_BYTES):
                raise ValueError
            _observation(expected)
            source_invocations += expected.get("error_type") != "ToolDispatchError"
            if source_invocations > max_api_requests:
                raise ValueError
            try:
                _check_depth(action, _MAX_JSON_DEPTH)
                parsed = json.loads(action, object_pairs_hook=_unique_object,
                                    parse_constant=_reject_constant)
                _check_json_values(parsed)
                if (type(parsed) is not dict or set(parsed) != {"name", "arguments"} or
                        type(parsed["name"]) is not str or type(parsed["arguments"]) is not dict):
                    raise ValueError
            except (ValueError, RecursionError):
                # Rejected source text is still an observed dispatch attempt.
                # Preserve it verbatim for the same fixed worker boundary;
                # never repair, normalize, evaluate, or execute it here.
                if expected.get("error_type") != "ToolDispatchError":
                    raise ValueError from None
        return copy.deepcopy(source_metadata), copy.deepcopy(prefix_records)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("Invalid trusted prefix replay input.") from None


def _reply(reply, previous, *, metadata, max_actions, max_api_requests):
    """Validate full replies, including cumulative costs from test doubles."""
    try:
        _canonical(reply)
        if type(reply) is not dict or type(reply.get("ok")) is not bool:
            raise ValueError
        fields = {"ok", "cost", "freshness"}
        fields |= {"result"} if reply["ok"] else {"error_type", "message"}
        if not reply["ok"] and reply.get("error_type") == "ObservedAPIError":
            fields.add("status_code")
        if set(reply) != fields:
            raise ValueError
        cost, fresh = reply["cost"], reply["freshness"]
        if (type(cost) is not dict or set(cost) != _COST_KEYS or
                any(type(value) is not int or value < previous[key]
                    for key, value in cost.items())):
            raise ValueError
        if (cost["action_attempts"] != previous["action_attempts"] + (not metadata) or
                cost["action_attempts"] > max_actions or
                cost["api_invocations"] > cost["action_attempts"] or
                cost["api_invocations"] > max_api_requests or
                cost["upstream_requests"] > max_api_requests):
            raise ValueError
        if (type(fresh) is not dict or set(fresh) != _FRESH_KEYS or
                fresh["last_successful_save_through_attempt"] is not None or
                any(type(fresh[key]) is not bool or fresh[key]
                    for key in _FRESH_KEYS - {"last_successful_save_through_attempt"})):
            raise ValueError
        if metadata:
            if not reply["ok"] or cost != previous:
                raise ValueError
            _metadata(reply["result"])
            return reply["result"]
        if reply["ok"]:
            value = {"ok": True, "result": reply["result"]}
        else:
            value = {key: reply[key] for key in ("ok", "error_type", "message")}
            if "status_code" in reply:
                value["status_code"] = reply["status_code"]
        # The recognized errors retain their actual invocation/request costs.
        # A public closure can make multiple upstream requests; do not replace
        # the observed request delta with a one-request assumption.
        invocations = cost["api_invocations"] - previous["api_invocations"]
        requests = cost["upstream_requests"] - previous["upstream_requests"]
        if value.get("error_type") == "ToolDispatchError":
            if invocations != 0 or requests != 0:
                raise ValueError
        elif invocations != 1 or requests < 1:
            raise ValueError
        _observation(value)
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise _ReplyError from None


def replay_prefix(worker, *, source_metadata, prefix_records, record, deadline,
                  max_actions, max_api_requests, clock=time.monotonic) -> dict:
    """Replay protected JSON history once, returning a fully private summary.

    Trusted source/configuration mistakes raise a content-free ``ValueError``
    before any request. Runtime failures and unexpected replies return unknown
    and never retry. ``worker_cost`` contains only validated cumulative totals;
    ``replay_actions_attempted`` additionally counts uncertain dispatch RPCs.
    No returned history may authorize continuation unless status is matched.
    """
    source_metadata, prefix_records = _inputs(
        source_metadata, prefix_records, record, deadline, max_actions, max_api_requests, clock,
    )
    cost = dict.fromkeys(_COST_KEYS, 0)
    actual_metadata = None
    verified = []
    attempted = 0
    outcome_unknown = unknown_cost = False
    last_time = None
    phase = None

    def write(event):
        try:
            record(copy.deepcopy(event))
        except Exception:
            raise _JournalError from None

    def now():
        nonlocal last_time
        try:
            value = clock()
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    (last_time is not None and value < last_time)):
                raise ValueError
        except Exception:
            raise _ClockError from None
        last_time = value
        return value

    def summary(status, reason):
        return {
            "status": status, "stop_reason": reason,
            "prefix_records": copy.deepcopy(verified) if status == "matched" else [],
            "metadata": copy.deepcopy(actual_metadata), "worker_cost": cost.copy(),
            "replay_actions_attempted": attempted,
            "replay_api_invocations": cost["api_invocations"],
            "replay_upstream_requests": cost["upstream_requests"],
            "replay_generations": 0, "replay_prompt_tokens": 0, "replay_completion_tokens": 0,
            "outcome_unknown": outcome_unknown, "unknown_cost": unknown_cost,
        }

    def finish(status, reason):
        result = summary(status, reason)
        write({"event": "replay_stop", "outcome": result})
        return result

    def exception_event(event, error):
        try:
            message = str(error)
        except Exception:
            message = "Exception message unavailable."
        write({"event": event, "error_type": type(error).__name__, "message": message})

    try:
        now()
        write({"event": "replay_start", "source_metadata": source_metadata,
               "prefix_length": len(prefix_records), "deadline": deadline,
               "max_actions": max_actions, "max_api_requests": max_api_requests})
        if now() >= deadline:
            return finish("unknown", "wall_seconds")
        write({"event": "replay_metadata_intent", "request": {"op": "metadata"}})
        remaining = deadline - now()
        if remaining <= 0:
            return finish("unknown", "wall_seconds")
        phase = "metadata"
        try:
            reply = worker.request({"op": "metadata"}, timeout=remaining)
        except Exception as error:
            outcome_unknown = unknown_cost = True
            exception_event("replay_metadata_failure", error)
            phase = None
            return finish("unknown", "worker_failure")
        write({"event": "replay_metadata_result", "reply": reply})
        try:
            actual_metadata = copy.deepcopy(_reply(
                reply, cost, metadata=True, max_actions=max_actions,
                max_api_requests=max_api_requests,
            ))
        except _ReplyError:
            outcome_unknown = unknown_cost = True
            return finish("unknown", "invalid_worker_reply")
        phase = None
        same_metadata = _canonical(actual_metadata) == _canonical(source_metadata)
        write({"event": "replay_metadata_comparison", "matched": same_metadata})
        if not same_metadata:
            return finish("observation_mismatch", "metadata_mismatch")
        if now() >= deadline:
            return finish("unknown", "wall_seconds")
        for index, item in enumerate(prefix_records):
            if now() >= deadline:
                return finish("unknown", "wall_seconds")
            if cost["action_attempts"] >= max_actions:
                return finish("unknown", "max_actions")
            if (cost["api_invocations"] >= max_api_requests or
                    cost["upstream_requests"] >= max_api_requests):
                return finish("unknown", "max_api_requests")
            write({"event": "replay_dispatch_intent", "index": index,
                   "action": item["action"], "expected_observation": item["observation"]})
            remaining = deadline - now()
            if remaining <= 0:
                return finish("unknown", "wall_seconds")
            attempted += 1
            phase = "dispatch"
            try:
                reply = worker.request({"op": "dispatch", "action": item["action"]},
                                       timeout=remaining)
            except Exception as error:
                outcome_unknown = unknown_cost = True
                exception_event("replay_dispatch_failure", error)
                phase = None
                return finish("unknown", "worker_failure")
            write({"event": "replay_dispatch_result", "index": index, "reply": reply})
            if (type(reply) is dict and reply.get("ok") is False and
                    reply.get("error_type") not in ("ToolDispatchError", "ObservedAPIError")):
                outcome_unknown = unknown_cost = True
                return finish("unknown", "worker_failure")
            try:
                actual = _reply(reply, cost, metadata=False, max_actions=max_actions,
                                max_api_requests=max_api_requests)
            except _ReplyError:
                outcome_unknown = unknown_cost = True
                return finish("unknown", "invalid_worker_reply")
            cost = reply["cost"].copy()
            phase = None
            matches = _canonical(actual) == _canonical(item["observation"])
            write({"event": "replay_observation_comparison", "index": index,
                   "observation": actual, "matched": matches})
            if not matches:
                return finish("observation_mismatch", "observation_mismatch")
            verified.append({"action": item["action"], "observation": copy.deepcopy(actual)})
            if now() >= deadline:
                return finish("unknown", "wall_seconds")
        return finish("matched", "prefix_matched")
    except _JournalError:
        if phase is not None:
            outcome_unknown = unknown_cost = True
        return summary("unknown", "journal_failure")
    except _ClockError:
        if phase is not None:
            outcome_unknown = unknown_cost = True
        try:
            return finish("unknown", "clock_failure")
        except _JournalError:
            return summary("unknown", "journal_failure")
