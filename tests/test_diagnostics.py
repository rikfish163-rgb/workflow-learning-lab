import json

import pytest

from wll.diagnostics import diagnose
from wll.protocol import Episode, Step, Task


def episode(steps, *, success=False, termination="step_budget", seed=17):
    return Episode(Task("a_1", "a", "train", "private task text"), "model", seed,
                   steps, success, termination)


def test_repeated_failures_ignore_comments_but_not_action_values():
    error = "Execution failed. Traceback:\nNameError: name 'token' is not defined"
    result = diagnose([episode([
        Step("# first try\nprint(token)", error, 10, 2),
        Step("print( token ) # repeat", error, 20, 3),
        Step("print(other_token)", error, 30, 4),
    ])])
    assert result["error_steps"] == {"unbound_variable": 3}
    assert result["repeated_failed_action_steps"] == 1
    assert result["cost"]["generated_completion_tokens"] == 9


def test_error_words_in_successful_data_are_not_classified_as_exceptions():
    result = diagnose([episode([Step("print(notes)", 'A note says "NameError"')],
                               success=True, termination="completed")])
    assert result["error_steps"] == {}
    assert result["successes"] == 1


def test_completion_is_not_success_and_prose_is_a_format_failure():
    result = diagnose([episode([
        Step("Please open the app.", "Execution failed. Traceback:\nSyntax error in line"),
        Step("apis.supervisor.complete_task()", "Execution successful."),
    ], termination="completed")])
    assert result["terminal_declaration_without_success"] == 1
    assert result["error_steps"] == {"invalid_python": 1}
    assert "private task text" not in str(result)
    assert "Please open" not in str(result)


def test_explicit_failure_declaration_is_distinguished_from_claiming_completion():
    result = diagnose([episode([
        Step("apis.supervisor.complete_task(status='fail')", "Execution successful."),
    ], termination="completed")])
    assert result["terminal_declaration_without_success"] == 1
    assert result["explicit_failure_declarations"] == 1


def test_replayed_tokens_are_reported_separately_not_counted_as_new_generation():
    e = episode([Step("x=1", "ok", 10, 5), Step("print(x)", "1", 20, 7)])
    e.branch, e.replayed_steps = "recovery", 1
    result = diagnose([e])
    assert result["cost"]["replay_steps"] == 1
    assert result["cost"]["generated_completion_tokens"] == 7
    assert result["cost"]["generated_prompt_tokens"] == 20


def test_duplicate_episode_is_rejected_before_aggregation():
    e = episode([])
    with pytest.raises(ValueError, match="duplicate"):
        diagnose([e, e])


def test_generation_format_failure_does_not_become_an_environment_error():
    result = diagnose([episode([
        Step("", "Generation failed: reasoning budget exhausted", 100, 4096),
    ], termination="generation_format_error")])
    assert result["generation_format_errors"] == 1
    assert result["error_steps"] == {}


@pytest.mark.parametrize(("message", "kind"), [
    ("Invalid credentials", "authentication_error"),
    ("You are either not authorized to access this simple_note API endpoint or your access "
     "token is missing, invalid or expired.", "authentication_error"),
    ("You are not authorized to deny this payment request.", "authorization_error"),
    ("You are not authorized to approve this payment request.", "authorization_error"),
])
def test_structured_app_errors_distinguish_authentication_from_permissions(message, kind):
    observation = "Execution failed. Traceback:\n" + json.dumps({"message": message})
    report = diagnose([episode([Step("apis.app.operation()", observation)])])
    assert report["error_steps"] == {kind: 1}
    assert report["episodes_with_error"] == {kind: 1}
    assert message not in json.dumps(report)


def test_authentication_text_in_successful_app_output_is_not_an_error():
    observation = json.dumps({"message": "Invalid credentials"})
    report = diagnose([episode([Step("print(note)", observation)])])
    assert report["error_steps"] == {}
