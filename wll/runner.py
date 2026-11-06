"""Single bounded rollout; a fresh world and exact prefix replay for every branch."""
from __future__ import annotations

import copy
import re
import time

from .model import ContextBudgetExceeded
from .prompts import DEFAULT_VERSION, initial_messages
from .protocol import Episode, Step


class ReplayMismatch(RuntimeError):
    pass


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, flags=re.S)
    # One unambiguous code block is common for instruct models. Preserve raw text
    # in the trace; multiple blocks remain a visible syntax/format failure.
    return blocks[0].strip() if len(blocks) == 1 else text


def run_episode(task, world_factory, model, *, max_steps: int, seed: int,
                prefix=(), experience=None, experience_id=None, branch="first", on_step=None,
                prompt_version=DEFAULT_VERSION):
    if max_steps <= 0 or len(prefix) > max_steps:
        raise ValueError("invalid step budget or replay prefix")
    if branch == "first" and prefix:
        raise ValueError("a replayed suffix is not a first attempt")
    messages = initial_messages(task, prompt_version)
    world = world_factory()
    started = time.monotonic()
    steps = []
    backend_seen, backend = False, None
    try:
        for old in prefix:
            replay_started = time.monotonic()
            observation = world.execute(old.action)
            if observation != old.observation:
                if on_step:
                    actual = copy.deepcopy(old)
                    actual.observation = observation
                    actual.elapsed_seconds = time.monotonic() - replay_started
                    on_step(len(steps), actual, True)
                raise ReplayMismatch(f"observation differs at replay step {len(steps)}")
            steps.append(copy.deepcopy(old))
            if on_step:
                on_step(len(steps)-1, steps[-1], True)
            messages.extend([{"role": "assistant", "content": old.raw_response or old.action},
                             {"role": "user", "content": observation}])
        if experience:
            messages.append({"role": "user", "content":
                             "A previous training experience may apply; check its conditions:\n" + experience})
        done = world.completed()
        termination = "step_budget"
        for step_id in range(len(prefix), max_steps):
            if done:
                break
            before = time.monotonic()
            try:
                response = model.generate(messages, seed=seed + step_id)
            except ContextBudgetExceeded:
                termination = "context_budget"
                break
            response_backend = response.get("backend")
            backend_changed = backend_seen and response_backend != backend
            if not backend_seen:
                backend, backend_seen = copy.deepcopy(response_backend), True
            format_error = response.get("format_error")
            action = "" if format_error or backend_changed else extract_code(response["text"])
            observation = ("Infrastructure error: backend identity changed" if backend_changed else
                           "Generation failed: " + format_error if format_error else world.execute(action))
            steps.append(Step(action, observation, response["prompt_tokens"],
                              response["completion_tokens"], time.monotonic()-before,
                              response["text"]))
            if on_step:
                on_step(len(steps)-1, steps[-1], False)
            if backend_changed:
                raise RuntimeError("backend identity changed during rollout")
            if format_error:
                termination = "generation_format_error"
                break
            messages.extend([{"role": "assistant", "content": response["text"]},
                             {"role": "user", "content": observation}])
            done = world.completed()
        # Boolean ground truth stays outside messages. No assertions are exposed.
        success = world.evaluate()
        return Episode(task, model.model_id, seed, steps, success,
                       "completed" if done else termination, experience_id, branch,
                       len(prefix), time.monotonic()-started,
                       {"max_steps": max_steps, "prompt_version": prompt_version,
                        "max_tokens": getattr(model, "max_tokens", None),
                        "temperature": getattr(model, "temperature", None),
                        **({"backend": backend} if backend is not None else {})})
    finally:
        world.close()
