"""Aggregate observable rollout symptoms; these labels do not infer root causes."""
import ast
import json
from collections import Counter

from .protocol import fingerprint


def error_kind(action, observation):
    if not observation.startswith("Execution failed"):
        return None
    try:
        ast.parse(action)
    except SyntaxError:
        return "invalid_python"
    # AppWorld API exceptions end with a JSON message. Classify only observed
    # failures; matching wording in an ordinary successful record is not evidence.
    try:
        payload = json.loads(observation.splitlines()[-1])
    except (ValueError, IndexError):
        payload = None
    message = payload.get("message") if isinstance(payload, dict) else None
    if isinstance(message, str):
        if message == "Invalid credentials" or (
                message.startswith("You are either not authorized to access this ") and
                message.endswith("access token is missing, invalid or expired.")):
            return "authentication_error"
        if message.startswith("You are not authorized to "):
            return "authorization_error"
    for label, patterns in [
        ("unbound_variable", ("NameError:", "UnboundLocalError:")),
        ("unknown_api", ("No API named", "No APIs with name")),
        ("invalid_arguments", ("Unexpected parameter", "Missing required", "TypeError:")),
        ("schema_or_index", ("KeyError:", "IndexError:", "AttributeError:")),
        ("policy_rejection", ("Policy restriction:", "CodePolicyError", "SafetyError", "not allowed", "forbidden")),
    ]:
        if any(pattern in observation for pattern in patterns):
            return label
    return "other_execution_error"


def action_key(action):
    try:
        return ast.dump(ast.parse(action), include_attributes=False)
    except SyntaxError:
        return action.strip()


def explicit_failure(action):
    """Only a standalone, literal status='fail' declaration is unambiguous here."""
    try:
        body = ast.parse(action).body
    except SyntaxError:
        return False
    if len(body) != 1 or not isinstance(body[0], ast.Expr):
        return False
    call = body[0].value
    return isinstance(call, ast.Call) and ast.unparse(call.func) == (
        "apis.supervisor.complete_task") and any(
            k.arg == "status" and isinstance(k.value, ast.Constant) and k.value.value == "fail"
            for k in call.keywords)


def diagnose(episodes):
    """No instructions, code, observations, credentials or answers in this report.

    Count repeated failed actions within an episode only. Repeating a successful
    action may be intentional pagination and is not classified as an error.
    Completed episodes alone cannot account for interrupted/unrecorded generation.
    """
    seen = set()
    errors, episode_errors, terminations, configurations = (Counter() for _ in range(4))
    repeats = replay_steps = generated_steps = prompt_tokens = completion_tokens = 0
    format_errors = declarations = failure_declarations = successes = 0
    scenarios = set()
    for episode in episodes:
        config = fingerprint({"model": episode.model_id, "metadata": episode.metadata})
        key = (episode.task.task_id, episode.seed, episode.branch, episode.experience_id, config)
        if key in seen:
            raise ValueError("duplicate rollout in diagnostic inputs")
        seen.add(key)
        configurations[config] += 1
        scenarios.add(episode.task.scenario_id)
        terminations[episode.termination] += 1
        successes += episode.success
        declarations += episode.termination == "completed" and not episode.success
        failure_declarations += bool(episode.termination == "completed" and episode.steps and
                                     explicit_failure(episode.steps[-1].action))
        failed_actions, kinds = set(), set()
        for index, step in enumerate(episode.steps):
            if index < episode.replayed_steps:
                replay_steps += 1
                continue
            generated_steps += 1
            prompt_tokens += step.prompt_tokens
            completion_tokens += step.completion_tokens
            if step.observation.startswith("Generation failed:"):
                format_errors += 1
            kind = error_kind(step.action, step.observation)
            if kind:
                errors[kind] += 1
                kinds.add(kind)
                action = action_key(step.action)
                repeats += action in failed_actions
                failed_actions.add(action)
        episode_errors.update(kinds)
    return {
        "error_taxonomy_version": "observable_errors_v2_auth",
        "episodes": len(seen), "scenarios": len(scenarios), "successes": successes,
        "terminations": dict(terminations), "error_steps": dict(errors),
        "episodes_with_error": dict(episode_errors),
        "repeated_failed_action_steps": repeats, "generation_format_errors": format_errors,
        "terminal_declaration_without_success": declarations,
        "explicit_failure_declarations": failure_declarations,
        "configuration_counts": dict(configurations),
        "cost": {"generated_prompt_tokens": prompt_tokens,
                 "generated_completion_tokens": completion_tokens,
                 "generated_steps": generated_steps, "replay_steps": replay_steps},
        "interpretation": "Descriptive symptoms on supplied completed episodes, not causal diagnoses "
                          "or a benchmark score across mixed configurations. Interrupted/unrecorded "
                          "generations require separate journal accounting.",
    }
