import pytest

from wll.protocol import Task
from wll.runner import ReplayMismatch, extract_code, run_episode


class World:
    """Deliberately tiny deterministic test double, not a benchmark."""
    def __init__(self):
        self.value = 0
        self.closed = False

    def execute(self, code):
        if code == "inc":
            self.value += 1
        return str(self.value)

    def completed(self):
        return self.value == 2

    def evaluate(self):
        return self.value == 2

    def close(self):
        self.closed = True


class Model:
    model_id = "test-double"

    def generate(self, messages, seed):
        return {"text": "inc", "prompt_tokens": 2, "completion_tokens": 1}


TASK = Task("a_1", "a", "train", "Increment twice")


def test_single_python_block_is_extracted_even_with_surrounding_explanation():
    assert extract_code("Here is the code:\n```python\nprint(1)\n```\nDone.") == "print(1)"


def test_multiple_code_blocks_are_not_silently_combined_or_dropped():
    response = "```python\nfirst()\n```\n```python\nsecond()\n```"
    assert extract_code(response) == response


def test_replay_reconstructs_state_and_is_charged_to_budget():
    baseline = run_episode(TASK, World, Model(), max_steps=2, seed=1)
    replayed = run_episode(TASK, World, Model(), max_steps=2, seed=1,
                           prefix=baseline.steps[:1], branch="recovery")
    assert replayed.success
    assert replayed.replayed_steps == 1
    assert len(replayed.steps) == 2
    assert replayed.generated_tokens == 1


def test_prefix_observation_mismatch_aborts_pair():
    baseline = run_episode(TASK, World, Model(), max_steps=2, seed=1)
    baseline.steps[0].observation = "nondeterministic"
    with pytest.raises(ReplayMismatch):
        run_episode(TASK, World, Model(), max_steps=2, seed=1,
                    prefix=baseline.steps[:1], branch="recovery")


def test_mismatched_replay_is_recorded_with_actual_observation_before_abort():
    baseline = run_episode(TASK, World, Model(), max_steps=2, seed=1)
    baseline.steps[0].observation = "expected-but-not-observed"
    journal = []
    with pytest.raises(ReplayMismatch):
        run_episode(TASK, World, Model(), max_steps=2, seed=1,
                    prefix=baseline.steps[:1], branch="recovery",
                    on_step=lambda index, step, replayed: journal.append((index, step, replayed)))
    assert len(journal) == 1
    assert journal[0][0] == 0 and journal[0][2] is True
    assert journal[0][1].observation == "1"
    assert baseline.steps[0].observation == "expected-but-not-observed"


def test_budget_exhaustion_is_failure_not_silent_missing_row():
    result = run_episode(TASK, World, Model(), max_steps=1, seed=1)
    assert not result.success
    assert result.termination == "step_budget"


def test_world_is_closed_when_model_raises():
    world = World()

    class Broken(Model):
        def generate(self, messages, seed):
            raise RuntimeError("transport")

    with pytest.raises(RuntimeError, match="transport"):
        run_episode(TASK, lambda: world, Broken(), max_steps=2, seed=1)
    assert world.closed


def test_unfinished_reasoning_is_costed_without_executing_it():
    class Truncated(Model):
        def generate(self, messages, seed):
            return {"text": "", "prompt_tokens": 100, "completion_tokens": 4096,
                    "format_error": "reasoning budget exhausted before final answer"}

    world = World()
    result = run_episode(TASK, lambda: world, Truncated(), max_steps=2, seed=1)
    assert result.termination == "generation_format_error"
    assert not result.success and world.value == 0
    assert result.generated_tokens == 4096


def test_serving_identity_is_preserved_and_mid_run_changes_abort_after_accounting():
    class Attested(Model):
        def generate(self, messages, seed):
            return {**super().generate(messages, seed), "backend": {"server_sha256": "one"}}

    result = run_episode(TASK, World, Attested(), max_steps=2, seed=1)
    assert result.metadata["backend"] == {"server_sha256": "one"}

    class Changing(Model):
        def generate(self, messages, seed):
            return {**super().generate(messages, seed), "backend": {"revision": seed}}

    world, journal = World(), []
    with pytest.raises(RuntimeError, match="backend identity changed"):
        run_episode(TASK, lambda: world, Changing(), max_steps=2, seed=1,
                    on_step=lambda i, step, replayed: journal.append(step))
    assert world.closed and world.value == 1
    assert sum(step.completion_tokens for step in journal) == 2
