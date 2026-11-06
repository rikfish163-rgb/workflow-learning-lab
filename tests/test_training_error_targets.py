from test_token_labels import Tokenizer

from wll.protocol import Episode, Step, Task
from wll.training import encode_target, export_first_attempt_examples


def test_recovery_context_is_kept_but_failed_action_is_not_a_positive_target():
    task = Task("synthetic_1", "synthetic", "train", "Count the records")
    failed = Step("print(records['items'])", "Execution failed. TypeError: list indices must be integers")
    fixed = Step("print(len(records))", "3")
    episode = Episode(task, "test-double", 17, [failed, fixed], True, "completed")
    rows = export_first_attempt_examples([episode])
    assert len(rows) == 1
    assert rows[0]["step_index"] == 1
    messages = rows[0]["messages"]
    assert messages[-3:] == [
        {"role": "assistant", "content": failed.action},
        {"role": "user", "content": failed.observation},
        {"role": "assistant", "content": fixed.action},
    ]
    encoded = encode_target(Tokenizer(), messages, 10000)
    targets = bytes(x for x in encoded["labels"] if x != -100).decode()
    assert targets == fixed.action + "</end>"
    assert rows[0]["target_selection"] == "exclude_observed_errors_v2"


def test_policy_rejection_and_generation_failure_cannot_be_sft_targets():
    steps = [Step("import os", "Execution failed. Policy restriction: import not permitted"),
             Step("", "Generation failed: missing final answer"),
             Step("print(1)", "1")]
    episode = Episode(Task("synthetic_1", "synthetic", "train", "Print one"),
                      "test-double", 17, steps, True, "completed")
    rows = export_first_attempt_examples([episode])
    assert [r["step_index"] for r in rows] == [2]
    assert len(rows[0]["messages"]) == 7  # system, task, two old action/observation pairs, target


def test_error_mention_in_an_ordinary_observation_does_not_discard_valid_action():
    episode = Episode(Task("synthetic_1", "synthetic", "train", "Read an error log"),
                      "test-double", 17,
                      [Step("print(log)", "The log contains the phrase Execution failed.")],
                      True, "completed")
    assert len(export_first_attempt_examples([episode])) == 1
