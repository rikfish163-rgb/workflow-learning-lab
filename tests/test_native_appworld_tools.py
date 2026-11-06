import builtins
import json
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from scripts.native_appworld_tools import (
    ToolDispatcher,
    ToolDispatchError,
    build_appworld_dispatcher,
)


def public_schema(name="phone__show_profile", parameter_names=("metadata", "optional")):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {parameter: {} for parameter in parameter_names},
            },
        },
    }


def action(arguments=None, name="phone__show_profile"):
    return json.dumps({"name": name, "arguments": arguments or {}})


@pytest.fixture
def bound_tools():
    target = Mock(return_value={"domain_result": "unchanged"})
    forbidden = Mock(side_effect=AssertionError("Private entrypoint invoked"))

    class Collection(dict):
        request = forbidden
        requester = forbidden

        def __getattr__(self, name):
            raise AssertionError("Attribute lookup is not dispatch")

    apis = Collection(
        phone={"show_profile": target},
        admin={"debit": forbidden, "credit": forbidden},
        _requester=forbidden,
    )
    dispatcher = ToolDispatcher([public_schema()], apis)
    return dispatcher, target, forbidden, apis


def test_dispatch_preserves_null_free_dictionary_values_and_return_identity(bound_tools):
    dispatcher, target, forbidden, _ = bound_tools
    arguments = {"optional": None, "metadata": {"free.key": [True, None, 2, "文字"]}}
    result = dispatcher.dispatch(action(arguments).encode("utf-8"))
    target.assert_called_once_with(**arguments)
    assert result is target.return_value
    forbidden.assert_not_called()
    assert dispatcher.action_attempts == 1
    assert dispatcher.api_invocations == 1
    assert dispatcher.tool_names == ("phone__show_profile",)


def test_missing_required_values_are_left_to_the_original_api():
    schema = public_schema(parameter_names=("username",))
    schema["function"]["parameters"]["required"] = ["username"]
    target = Mock()
    dispatcher = ToolDispatcher([schema], {"phone": {"show_profile": target}})
    dispatcher.dispatch(action())
    target.assert_called_once_with()


@pytest.mark.parametrize(
    "parameter",
    ["client", "track", "show", "raise_on_failure", "_system_datetime", "_app_name", "_api_name"],
)
def test_reserved_parameters_rejected_before_keyword_expansion(bound_tools, parameter):
    dispatcher, target, forbidden, _ = bound_tools
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(action({parameter: False}))
    target.assert_not_called()
    forbidden.assert_not_called()
    assert dispatcher.action_attempts == 1
    assert dispatcher.api_invocations == 0


@pytest.mark.parametrize(
    "name",
    ["admin__debit", "admin__credit", "phone__unknown", "https://host/phone", "__class__", "phone.__dict__"],
)
def test_unknown_private_and_url_tools_cannot_reach_collection(bound_tools, name):
    dispatcher, target, forbidden, _ = bound_tools
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(action(name=name))
    target.assert_not_called()
    forbidden.assert_not_called()
    assert dispatcher.action_attempts == 1
    assert dispatcher.api_invocations == 0


def test_unknown_argument_cannot_reach_target(bound_tools):
    dispatcher, target, _, _ = bound_tools
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(action({"code": "print('not executed')"}))
    target.assert_not_called()


def test_bound_target_is_not_looked_up_again_after_construction(bound_tools):
    dispatcher, target, _, apis = bound_tools
    apis["phone"]["show_profile"] = Mock(side_effect=AssertionError("Rebound target invoked"))
    dispatcher.dispatch(action())
    target.assert_called_once_with()


def test_target_exception_propagates_and_still_counts_as_invocation():
    failure = RuntimeError("Synthetic target failure")
    target = Mock(side_effect=failure)
    dispatcher = ToolDispatcher([public_schema()], {"phone": {"show_profile": target}})
    with pytest.raises(RuntimeError) as caught:
        dispatcher.dispatch(action())
    assert caught.value is failure
    assert dispatcher.action_attempts == 1
    assert dispatcher.api_invocations == 1


@pytest.mark.parametrize("field", ["tool_names", "action_attempts", "api_invocations"])
def test_public_metadata_is_read_only(bound_tools, field):
    dispatcher, _, _, _ = bound_tools
    with pytest.raises(AttributeError):
        setattr(dispatcher, field, 0)


@pytest.mark.parametrize(
    "payload",
    [
        '{"name":"phone__show_profile","name":"phone__show_profile","arguments":{}}',
        '{"name":"phone__show_profile","arguments":{"metadata":{"x":1,"x":2}}}',
        '{"name":"phone__show_profile","arguments":{"optional":NaN}}',
        '{"name":"phone__show_profile","arguments":{"optional":Infinity}}',
        '{"name":"phone__show_profile","arguments":{"optional":-Infinity}}',
        '{"name":"phone__show_profile","arguments":{"optional":1e9999}}',
        '{"name":"phone__show_profile","arguments":{"optional":"\\ud800"}}',
        '{"name":"phone__show_profile","arguments":{"metadata":{"\\udfff":0}}}',
        '{"name":"phone__show_profile","arguments":{}} trailing',
        '{"name":"phone__show_profile","arguments":{},}',
        '\ufeff{"name":"phone__show_profile","arguments":{}}',
        b'{"name":"phone__show_profile","arguments":{"optional":"\xff"}}',
        '{"name":"phone__show_profile","arguments":{"optional":"\ud800"}}',
    ],
)
def test_invalid_json_and_utf8_rejected_without_invocation(bound_tools, payload):
    dispatcher, target, _, _ = bound_tools
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(payload)
    target.assert_not_called()
    assert dispatcher.action_attempts == 1
    assert dispatcher.api_invocations == 0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"name": "phone__show_profile", "arguments": {}},
        bytearray(b"{}"),
        "[]",
        "null",
        '{"name":"phone__show_profile"}',
        '{"arguments":{}}',
        '{"name":"phone__show_profile","arguments":{},"code":"ignored?"}',
        '{"name":7,"arguments":{}}',
        '{"name":"phone__show_profile","arguments":[]}',
        '{"name":"phone__show_profile","arguments":null}',
    ],
)
def test_only_exact_json_envelope_is_accepted(bound_tools, payload):
    dispatcher, target, _, _ = bound_tools
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(payload)
    target.assert_not_called()


def test_byte_limit_counts_utf8_bytes():
    text = json.dumps({"name": "phone__show_profile", "arguments": {"optional": "字"}}, ensure_ascii=False)
    target = Mock()
    dispatcher = ToolDispatcher(
        [public_schema()], {"phone": {"show_profile": target}},
        max_action_bytes=len(text.encode("utf-8")),
    )
    dispatcher.dispatch(text)
    dispatcher.dispatch(text.encode("utf-8"))
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(text + " ")
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch((text + " ").encode("utf-8"))
    assert target.call_count == 2
    assert dispatcher.action_attempts == 4
    assert dispatcher.api_invocations == 2


def test_depth_limit_ignores_brackets_and_escaped_quotes_in_strings():
    target = Mock()
    dispatcher = ToolDispatcher(
        [public_schema()], {"phone": {"show_profile": target}}, max_json_depth=3,
    )
    dispatcher.dispatch(action({"metadata": {"text": '[[[{\\"}]]'}}))
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch(action({"metadata": {"nested": {}}}))
    with pytest.raises(ToolDispatchError):
        dispatcher.dispatch("[" * 2000 + "]" * 2000)
    assert target.call_count == 1
    assert dispatcher.action_attempts == 3
    assert dispatcher.api_invocations == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
@pytest.mark.parametrize("field", ["max_action_bytes", "max_json_depth"])
def test_invalid_limits_rejected(limit, field):
    with pytest.raises(ValueError):
        ToolDispatcher([public_schema()], {"phone": {"show_profile": Mock()}}, **{field: limit})


def test_duplicate_tool_schemas_are_rejected():
    with pytest.raises(ValueError):
        ToolDispatcher([public_schema(), public_schema()], {"phone": {"show_profile": Mock()}})


@pytest.mark.parametrize(
    "name",
    ["admin__debit", "admin__credit", "__class__", "phone.__dict__", "phone__show__profile", "https://host"],
)
def test_constructor_rejects_invalid_or_admin_tool_names(name):
    with pytest.raises(ValueError):
        ToolDispatcher([public_schema(name=name)], {"admin": {"debit": Mock(), "credit": Mock()}})


@pytest.mark.parametrize(
    "parameter", ["client", "track", "show", "raise_on_failure", "_system_datetime", "_app_name", "_api_name"],
)
def test_constructor_rejects_reserved_parameter_collisions(parameter):
    with pytest.raises(ValueError):
        ToolDispatcher([public_schema(parameter_names=(parameter,))], {"phone": {"show_profile": Mock()}})


@pytest.mark.parametrize("schema", [{}, {"type": "other"}, {"type": "function", "function": {}}])
def test_constructor_rejects_malformed_schema(schema):
    with pytest.raises(ValueError):
        ToolDispatcher([schema], {})


def test_constructor_requires_callable_target():
    with pytest.raises(ValueError):
        ToolDispatcher([public_schema()], {"phone": {"show_profile": "not callable"}})


def test_dispatch_does_not_use_python_execution_builtins(bound_tools, monkeypatch):
    dispatcher, target, _, _ = bound_tools
    forbidden = Mock(side_effect=AssertionError("Python execution builtin invoked"))
    payload = action({"optional": "__import__('os').system('not executed')"})
    with monkeypatch.context() as patcher:
        for name in ("eval", "exec", "compile"):
            patcher.setattr(builtins, name, forbidden)
        dispatcher.dispatch(payload)
    forbidden.assert_not_called()
    target.assert_called_once_with(optional="__import__('os').system('not executed')")


def test_factory_builds_public_schemas_without_task_or_world_lifecycle(monkeypatch):
    app_names = ["api_docs", "supervisor", "phone"]
    get_all_apps = Mock(return_value=app_names)
    docs = Mock()
    docs.function_calling.return_value = [public_schema()]
    collection = Mock()
    collection.build.return_value = docs
    for name in ("appworld", "appworld.collections", "appworld.collections.api_docs", "appworld.apps"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["appworld.apps"], "get_all_apps", get_all_apps, raising=False)
    monkeypatch.setattr(sys.modules["appworld.collections.api_docs"], "ApiDocCollection", collection, raising=False)
    target = Mock()
    dispatcher = build_appworld_dispatcher({"phone": {"show_profile": target}})
    get_all_apps.assert_called_once_with(skip_admin=True)
    collection.build.assert_called_once_with(load_apps=tuple(app_names), include_response_schemas=False)
    docs.function_calling.assert_called_once_with()
    assert dispatcher.tool_names == ("phone__show_profile",)
    target.assert_not_called()
