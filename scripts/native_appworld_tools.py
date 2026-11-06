"""A JSON boundary for trusted, prebound public AppWorld API closures.

This module is not a task backend. It provides no task/database lifecycle,
scoring, process isolation, or experiment budget. Callers retain those duties
and must keep API results, credentials, and target exceptions private.

Only tool and parameter names are checked against the public source schemas.
Values, including null and free dictionaries, go to the original API's
FastAPI/Pydantic validation without a replacement schema-value validator.
API results and exceptions are returned/raised unchanged; nothing is logged.
"""

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", re.ASCII)
_RESERVED_ARGUMENTS = frozenset(
    {"client", "track", "show", "raise_on_failure", "_system_datetime", "_app_name", "_api_name"}
)


class ToolDispatchError(ValueError):
    """The action was rejected before invoking an API closure."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolDispatchError("Duplicate JSON object key.")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ToolDispatchError("Non-finite JSON number.")


def _check_depth(text: str, maximum: int) -> None:
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > maximum:
                raise ToolDispatchError("JSON nesting exceeds the depth limit.")
        elif char in "]}":
            depth -= 1


def _check_json_values(value: Any) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        kind = type(current)
        if kind is dict:
            pending.extend(current.keys())
            pending.extend(current.values())
        elif kind is list:
            pending.extend(current)
        elif kind is str:
            try:
                current.encode("utf-8")
            except UnicodeEncodeError:
                raise ToolDispatchError("JSON contains invalid Unicode.") from None
        elif kind is float:
            if not math.isfinite(current):
                raise ToolDispatchError("Non-finite JSON number.")
        elif current is not None and kind not in (int, bool):
            raise ToolDispatchError("Only JSON values are allowed.")


class ToolDispatcher:
    """Bind a trusted public schema list to fixed API closures once.

    Each dispatch accepts exactly a JSON object with ``name`` and ``arguments``.
    The byte limit applies to UTF-8 encoding; the depth limit counts object/array
    containers, including the envelope. Attempts count every dispatch, including
    validation failures. Invocations count calls into targets, including calls
    that raise. These counters are observations, not runtime budget enforcement.
    """

    def __init__(
        self,
        public_schemas: Iterable[Mapping[str, Any]],
        apis: Mapping[str, Mapping[str, Callable[..., Any]]],
        *,
        max_action_bytes: int = 65536,
        max_json_depth: int = 32,
    ) -> None:
        for limit in (max_action_bytes, max_json_depth):
            if type(limit) is not int or limit < 1:
                raise ValueError("JSON limits must be positive integers.")
        tools: dict[str, tuple[frozenset[str], Callable[..., Any]]] = {}
        for schema in public_schemas:
            if not isinstance(schema, Mapping) or schema.get("type") != "function":
                raise ValueError("Expected a public function schema.")
            function = schema.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("Expected a public function schema.")
            name = function.get("name")
            if type(name) is not str or name.count("__") != 1:
                raise ValueError("Invalid public tool name.")
            app_name, api_name = name.split("__")
            if (
                app_name == "admin"
                or not _IDENTIFIER.fullmatch(app_name)
                or not _IDENTIFIER.fullmatch(api_name)
            ):
                raise ValueError("Invalid public tool name.")
            if name in tools:
                raise ValueError("Duplicate public tool schema.")
            parameters = function.get("parameters")
            if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
                raise ValueError("Expected object parameters in the public schema.")
            properties = parameters.get("properties")
            if not isinstance(properties, Mapping):
                raise ValueError("Expected parameter properties in the public schema.")
            for parameter in properties:
                if (
                    type(parameter) is not str
                    or parameter in _RESERVED_ARGUMENTS
                    or not _IDENTIFIER.fullmatch(parameter)
                ):
                    raise ValueError("Invalid or reserved public parameter name.")
            try:
                target = apis[app_name][api_name]
            except (KeyError, TypeError):
                raise ValueError("The public API closure is missing.") from None
            if not callable(target):
                raise ValueError("The public API target must be callable.")
            tools[name] = (frozenset(properties), target)
        self._tools = tools
        self._tool_names = tuple(sorted(tools))
        self._max_action_bytes = max_action_bytes
        self._max_json_depth = max_json_depth
        self._action_attempts = 0
        self._api_invocations = 0

    @property
    def tool_names(self) -> tuple[str, ...]:
        return self._tool_names

    @property
    def action_attempts(self) -> int:
        return self._action_attempts

    @property
    def api_invocations(self) -> int:
        return self._api_invocations

    def dispatch(self, action: str | bytes) -> Any:
        self._action_attempts += 1
        if type(action) not in (str, bytes):
            raise ToolDispatchError("The action must be UTF-8 JSON text or bytes.")
        if len(action) > self._max_action_bytes:
            raise ToolDispatchError("The action exceeds the byte limit.")
        try:
            if type(action) is bytes:
                text = action.decode("utf-8")
            else:
                text = action
                if len(text.encode("utf-8")) > self._max_action_bytes:
                    raise ToolDispatchError("The action exceeds the byte limit.")
        except UnicodeError:
            raise ToolDispatchError("The action is not valid UTF-8.") from None
        _check_depth(text, self._max_json_depth)
        try:
            payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            raise ToolDispatchError("Invalid JSON action.") from None
        _check_json_values(payload)
        if type(payload) is not dict or set(payload) != {"name", "arguments"}:
            raise ToolDispatchError("Expected only name and arguments fields.")
        name, arguments = payload["name"], payload["arguments"]
        if type(name) is not str or type(arguments) is not dict:
            raise ToolDispatchError("Expected a tool name and an arguments object.")
        if name not in self._tools:
            raise ToolDispatchError("Unknown public tool.")
        allowed_parameters, target = self._tools[name]
        if _RESERVED_ARGUMENTS.intersection(arguments) or set(arguments) - allowed_parameters:
            raise ToolDispatchError("Unexpected or reserved arguments.")
        self._api_invocations += 1
        return target(**arguments)


def build_appworld_dispatcher(
    apis: Mapping[str, Mapping[str, Callable[..., Any]]],
    *,
    max_action_bytes: int = 65536,
    max_json_depth: int = 32,
) -> ToolDispatcher:
    """Bind full public source schemas without starting a task, world, or scorer.

    The caller supplies an already initialized, trusted API collection. AppWorld
    is imported lazily so the boundary and its unit tests need only the stdlib.
    """
    from appworld.apps import get_all_apps
    from appworld.collections.api_docs import ApiDocCollection

    schemas = ApiDocCollection.build(
        load_apps=tuple(get_all_apps(skip_admin=True)), include_response_schemas=False
    ).function_calling()
    return ToolDispatcher(
        schemas, apis, max_action_bytes=max_action_bytes, max_json_depth=max_json_depth
    )
