"""Bounded private parent loop for the native AppWorld JSON worker.

The caller supplies an already initialized, fresh worker, private metadata, an
exact CPU token counter, a configured model, and a durable private journal
callback. The callback must synchronously append and fsync every event or
raise. Metadata, messages, model responses and worker replies are private.
This loop neither saves nor closes the worker and never scores a task.

The caller also owns service context-window/timeout enforcement and must use
the same API-request limit for the worker's upstream RequestTracker. Returned
costs are confirmed totals, except dispatch attempts include an RPC attempted
even if its outcome is unknown. Unknown flags qualify any incomplete totals.
"""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

SYSTEM_PROMPT = """You interact with the available apps using one JSON action at a time.
Respond with only a JSON object of the form {"name":"app__api","arguments":{...}}.
Use the public APIs, not Python, URLs, or administrative or internal controls.
Discover apps with {"name":"api_docs__show_app_descriptions","arguments":{}}.
List an app's APIs with {"name":"api_docs__show_api_descriptions","arguments":{"app_name":"APP"}}.
Get a public API's parameters and details with {"name":"api_docs__show_api_doc","arguments":{"app_name":"APP","api_name":"API"}}.
Use the returned public documentation to choose API names and arguments.
Call supervisor__complete_task when you have finished the task.
"""


@dataclass(frozen=True)
class ActorLimits:
    max_generations: int
    max_actions: int
    max_api_requests: int
    max_input_tokens: int
    max_prompt_tokens_total: int
    max_completion_tokens_total: int
    max_new_tokens: int
    wall_seconds: float


class _JournalError(Exception):
    pass


class _ClockError(Exception):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _validate_inputs(metadata: Any, limits: Any, seed: Any, model: Any) -> str:
    if not isinstance(limits, ActorLimits):
        raise ValueError("Expected ActorLimits.")
    for name in (
        "max_generations", "max_actions", "max_api_requests", "max_input_tokens",
        "max_prompt_tokens_total", "max_completion_tokens_total", "max_new_tokens",
    ):
        value = getattr(limits, name)
        if type(value) is not int or value < 1:
            raise ValueError("Actor integer limits must be positive integers.")
    if (type(limits.wall_seconds) not in (int, float) or
            not math.isfinite(limits.wall_seconds) or limits.wall_seconds <= 0):
        raise ValueError("Wall limit must be finite and positive.")
    if (type(seed) is not int or seed < 0 or
            seed + limits.max_generations - 1 > 2**31 - 1):
        raise ValueError("Generation seeds must fit the nonnegative signed 32-bit range.")
    if type(getattr(model, "max_tokens", None)) is not int or model.max_tokens != limits.max_new_tokens:
        raise ValueError("Model completion limit must equal the reserved maximum.")
    return _metadata_text(metadata)


def _metadata_text(metadata: Any) -> str:
    if type(metadata) is not dict or set(metadata) != {"instruction", "supervisor", "app_descriptions"}:
        raise ValueError("Invalid private metadata.")
    supervisor, descriptions = metadata["supervisor"], metadata["app_descriptions"]
    if (type(metadata["instruction"]) is not str or type(supervisor) is not dict or
            set(supervisor) - {"first_name", "last_name", "email", "phone_number"} or
            any(type(value) is not str for value in supervisor.values()) or
            type(descriptions) is not dict or
            any(type(key) is not str or type(value) is not str for key, value in descriptions.items())):
        raise ValueError("Invalid private metadata.")
    try:
        text = _json(metadata)
        text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("Invalid private metadata.") from None
    return text


def _validate_review_hint(review_hint):
    try:
        if review_hint is not None and (type(review_hint) is not str or
                not 1 <= len(review_hint.encode("utf-8")) <= 65536):
            raise ValueError
    except (ValueError, UnicodeError):
        raise ValueError("Invalid fixed review hint.") from None


def build_actor_messages(metadata, *, prefix_records=None, review_hint=None):
    """Build the full private context, appending one fixed hint after its prefix.

    Student projection removes only index ``2 + 2 * len(prefix_records)``;
    identical text elsewhere in the history is preserved. Replay equivalence
    and cumulative costs remain the controller's obligation.
    """
    text = _metadata_text(metadata)
    _validate_review_hint(review_hint)
    prefix = [] if prefix_records is None else prefix_records
    _prefix_api_count(prefix)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
    for item in prefix:
        messages.extend([{"role": "assistant", "content": item["action"]},
                         {"role": "user", "content": _json(item["observation"])}])
    if review_hint is not None:
        messages.append({"role": "user", "content": review_hint})
    return messages


def _validate_reply(reply: Any, previous: dict[str, int], attempted: int, limits: ActorLimits) -> bool:
    if type(reply) is not dict or type(reply.get("ok")) is not bool:
        return False
    cost, freshness = reply.get("cost"), reply.get("freshness")
    if type(cost) is not dict or set(cost) != {"action_attempts", "api_invocations", "upstream_requests"}:
        return False
    if any(type(value) is not int or value < 0 for value in cost.values()):
        return False
    if (cost["action_attempts"] != attempted or attempted > limits.max_actions or
            cost["api_invocations"] < previous["api_invocations"] or
            cost["api_invocations"] > previous["api_invocations"] + 1 or
            cost["api_invocations"] > attempted or
            cost["api_invocations"] > limits.max_api_requests or
            cost["upstream_requests"] < previous["upstream_requests"] or
            cost["upstream_requests"] > limits.max_api_requests):
        return False
    required = {"last_successful_save_through_attempt", "current_state_saved", "score_eligible", "outcome_unknown", "unknown_cost"}
    if type(freshness) is not dict or set(freshness) != required:
        return False
    if any(type(freshness[key]) is not bool for key in required - {"last_successful_save_through_attempt"}):
        return False
    saved_attempt = freshness["last_successful_save_through_attempt"]
    if saved_attempt is not None and (type(saved_attempt) is not int or not 0 <= saved_attempt <= attempted):
        return False
    if freshness["current_state_saved"] or freshness["score_eligible"]:
        return False
    if reply["ok"]:
        return (set(reply) == {"ok", "result", "cost", "freshness"} and
                cost["api_invocations"] == previous["api_invocations"] + 1 and
                cost["upstream_requests"] >= previous["upstream_requests"] + 1)
    valid_error = (
        set(reply) in ({"ok", "error_type", "message", "cost", "freshness"},
                       {"ok", "error_type", "message", "status_code", "cost", "freshness"})
        and type(reply.get("error_type")) is str and type(reply.get("message")) is str
    )
    if not valid_error:
        return False
    if reply["error_type"] == "ToolDispatchError":
        return (cost["api_invocations"] == previous["api_invocations"] and
                cost["upstream_requests"] == previous["upstream_requests"])
    if reply["error_type"] == "ObservedAPIError":
        return (cost["api_invocations"] == previous["api_invocations"] + 1 and
                cost["upstream_requests"] >= previous["upstream_requests"] + 1)
    return True


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key.")
        result[key] = value
    return result


def _is_completion(action: str) -> bool:
    def reject_constant(_value: str) -> None:
        raise ValueError("Non-finite number.")

    try:
        parsed = json.loads(action, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return False
    return (type(parsed) is dict and set(parsed) == {"name", "arguments"} and
            parsed["name"] == "supervisor__complete_task" and type(parsed["arguments"]) is dict)


def _prefix_api_count(prefix_records):
    from scripts.native_appworld_replay import MAX_ACTION_BYTES, _canonical, _observation

    try:
        if type(prefix_records) is not list:
            raise ValueError
        api_count = 0
        for item in prefix_records:
            if type(item) is not dict or set(item) != {"action", "observation"}:
                raise ValueError
            action, observation = item["action"], item["observation"]
            if type(action) is not str or len(action.encode("utf-8")) > MAX_ACTION_BYTES:
                raise ValueError
            _observation(observation)
            if observation["ok"] or observation["error_type"] == "ObservedAPIError":
                parsed = json.loads(action, object_pairs_hook=_unique_pairs,
                                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                _canonical(parsed)
                if (type(parsed) is not dict or set(parsed) != {"name", "arguments"}
                        or type(parsed["name"]) is not str or type(parsed["arguments"]) is not dict
                        or (observation["ok"] and _is_completion(action))):
                    raise ValueError
                api_count += 1
    except (TypeError, ValueError, KeyError, UnicodeError, RecursionError):
        raise ValueError("Invalid verified replay history.") from None
    return api_count


def _validated_prefix(prefix_records, initial_worker_cost, limits):
    zero = {"action_attempts": 0, "api_invocations": 0, "upstream_requests": 0}
    if prefix_records is None and initial_worker_cost is None:
        return [], zero
    try:
        if (type(prefix_records) is not list or len(prefix_records) > limits.max_actions
                or type(initial_worker_cost) is not dict or set(initial_worker_cost) != set(zero)
                or any(type(v) is not int or v < 0 for v in initial_worker_cost.values())):
            raise ValueError
        api_count = _prefix_api_count(prefix_records)
        if (initial_worker_cost["action_attempts"] != len(prefix_records)
                or initial_worker_cost["api_invocations"] != api_count
                or not api_count <= initial_worker_cost["upstream_requests"] <= limits.max_api_requests
                or api_count > limits.max_api_requests):
            raise ValueError
        if api_count == 0 and initial_worker_cost["upstream_requests"] != 0:
            raise ValueError
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("Invalid verified replay history or cumulative worker cost.") from None
    return copy.deepcopy(prefix_records), initial_worker_cost.copy()


def run_actor(worker, model, token_counter, *, metadata, limits, seed, record,
              clock=time.monotonic, prefix_records=None, initial_worker_cost=None,
              review_hint=None) -> dict:
    """Run the private actor until a budget, self-report, or unknown outcome.

    The worker starts with zero counters or a trusted, fully verified replay
    history and its actual cumulative costs. It accepts only dispatch requests
    from this loop. ``record`` receives complete private event dictionaries;
    successful return must mean their contents are durable. No exceptions or
    private text are exposed in the returned, allowlisted summary.
    """
    _validate_inputs(metadata, limits, seed, model)
    prefix, replay_cost = _validated_prefix(prefix_records, initial_worker_cost, limits)
    messages = build_actor_messages(metadata, prefix_records=prefix, review_hint=review_hint)
    cost = {"generations": 0, "prompt_tokens": 0, "completion_tokens": 0,
            **replay_cost}
    outcome_unknown = unknown_cost = False
    phase = None
    last_time = None

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
        return {"status": status, "stop_reason": reason, "cost": cost.copy(),
                "outcome_unknown": outcome_unknown, "unknown_cost": unknown_cost}

    def finish(status, reason):
        result = summary(status, reason)
        write({"event": "actor_stop", "outcome": result})
        return result

    def exception_event(event, error):
        try:
            message = str(error)
        except Exception:
            message = "Exception message unavailable."
        write({"event": event, "error_type": type(error).__name__, "message": message})

    try:
        started = now()
        deadline = started + limits.wall_seconds
        if not math.isfinite(deadline):
            raise _ClockError
        write({"event": "actor_start", "messages": messages, "limits": asdict(limits), "seed": seed,
               "review_hint_index": 2 + 2 * len(prefix) if review_hint is not None else None})
        while True:
            if now() >= deadline:
                return finish("incomplete", "wall_seconds")
            if cost["generations"] >= limits.max_generations:
                return finish("incomplete", "max_generations")
            if cost["action_attempts"] >= limits.max_actions:
                return finish("incomplete", "max_actions")
            if (cost["api_invocations"] >= limits.max_api_requests or
                    cost["upstream_requests"] >= limits.max_api_requests):
                return finish("incomplete", "max_api_requests")
            try:
                prompt_tokens = token_counter(copy.deepcopy(messages))
                if type(prompt_tokens) is not int or prompt_tokens < 1:
                    raise ValueError("Invalid token count.")
            except Exception as error:
                exception_event("token_count_failure", error)
                return finish("unknown", "token_count_failure")
            if now() >= deadline:
                return finish("incomplete", "wall_seconds")
            if prompt_tokens > limits.max_input_tokens:
                return finish("incomplete", "max_input_tokens")
            if cost["prompt_tokens"] + prompt_tokens > limits.max_prompt_tokens_total:
                return finish("incomplete", "max_prompt_tokens_total")
            if cost["completion_tokens"] + limits.max_new_tokens > limits.max_completion_tokens_total:
                return finish("incomplete", "max_completion_tokens_total")
            generation_seed = seed + cost["generations"]
            write({"event": "generation_intent", "messages": messages, "seed": generation_seed,
                   "prompt_tokens": prompt_tokens, "max_new_tokens": limits.max_new_tokens})
            if now() >= deadline:
                return finish("incomplete", "wall_seconds")
            phase = "generation"
            cost["generations"] += 1
            try:
                response = model.generate(copy.deepcopy(messages), seed=generation_seed)
            except Exception as error:
                unknown_cost = True
                exception_event("generation_failure", error)
                phase = None
                return finish("unknown", "generation_failure")
            write({"event": "generation_result", "response": response})
            phase = None
            if (type(response) is not dict or type(response.get("text")) is not str or
                    type(response.get("prompt_tokens")) is not int or
                    response["prompt_tokens"] != prompt_tokens or
                    type(response.get("completion_tokens")) is not int or
                    not 1 <= response["completion_tokens"] <= limits.max_new_tokens):
                unknown_cost = True
                return finish("unknown", "invalid_model_result")
            try:
                response["text"].encode("utf-8")
            except UnicodeError:
                unknown_cost = True
                return finish("unknown", "invalid_model_result")
            cost["prompt_tokens"] += prompt_tokens
            cost["completion_tokens"] += response["completion_tokens"]
            if now() >= deadline:
                return finish("incomplete", "wall_seconds")
            action = response["text"]
            write({"event": "dispatch_intent", "action": action})
            remaining = deadline - now()
            if remaining <= 0:
                return finish("incomplete", "wall_seconds")
            previous = cost.copy()
            phase = "dispatch"
            cost["action_attempts"] += 1
            try:
                reply = worker.request({"op": "dispatch", "action": action}, timeout=remaining)
            except Exception as error:
                outcome_unknown = unknown_cost = True
                exception_event("dispatch_failure", error)
                phase = None
                return finish("unknown", "worker_failure")
            write({"event": "dispatch_result", "reply": reply})
            phase = None
            if not _validate_reply(reply, previous, cost["action_attempts"], limits):
                outcome_unknown = unknown_cost = True
                return finish("unknown", "invalid_worker_reply")
            cost.update(reply["cost"])
            timed_out = now() >= deadline
            if reply["freshness"]["outcome_unknown"] or reply["freshness"]["unknown_cost"]:
                outcome_unknown = reply["freshness"]["outcome_unknown"]
                unknown_cost = reply["freshness"]["unknown_cost"]
                return finish("unknown", "worker_failure")
            if reply["ok"]:
                observation = {"ok": True, "result": reply["result"]}
            elif reply["error_type"] == "ToolDispatchError" and "status_code" not in reply:
                observation = {"ok": False, "error_type": reply["error_type"], "message": reply["message"]}
            elif (reply["error_type"] == "ObservedAPIError" and type(reply.get("status_code")) is int and
                  400 <= reply["status_code"] < 500):
                observation = {"ok": False, "error_type": reply["error_type"], "message": reply["message"],
                               "status_code": reply["status_code"]}
            else:
                outcome_unknown = unknown_cost = True
                return finish("unknown", "worker_failure")
            try:
                observation_text = _json(observation)
            except (TypeError, ValueError, UnicodeError, RecursionError):
                outcome_unknown = unknown_cost = True
                return finish("unknown", "invalid_worker_reply")
            messages.extend([{"role": "assistant", "content": action},
                             {"role": "user", "content": observation_text}])
            if timed_out:
                return finish("incomplete", "wall_seconds")
            if reply["ok"] and _is_completion(action):
                return finish("model_reported_complete", "supervisor_complete")
    except _JournalError:
        if phase == "generation":
            unknown_cost = True
        elif phase == "dispatch":
            outcome_unknown = unknown_cost = True
        return summary("unknown", "journal_failure")
    except _ClockError:
        if phase == "generation":
            unknown_cost = True
        elif phase == "dispatch":
            outcome_unknown = unknown_cost = True
        try:
            return finish("unknown", "clock_failure")
        except _JournalError:
            return summary("unknown", "journal_failure")
