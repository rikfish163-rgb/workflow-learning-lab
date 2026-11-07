"""Closed inspection requests; execute only through a trusted worker dispatcher.

The caller must bind an independently reviewed read-only endpoint allowlist and
the real requester/counter, outside the actor's Python namespace. Syntax or a
GET method alone does not prove an API has no side effects. No eval/exec occurs.
"""

import json
import math
import re

MAX_REQUEST_BYTES = 32768
MAX_VALUE_NODES = 2048
MAX_VALUE_DEPTH = 16
_NAME = re.compile(r'[a-zA-Z][a-zA-Z0-9_]{0,63}\Z')
_RESERVED = frozenset({'apis', 'requester', 'client', 'track', 'show',
                       'raise_on_failure', 'eval', 'exec', 'globals', 'locals'})


class CheckProtocolError(ValueError):
    """Fixed diagnostic code only; never include actor values or credentials."""


def _require(condition, reason):
    if not condition:
        raise CheckProtocolError(reason)


def _name(value):
    return type(value) is str and _NAME.fullmatch(value) is not None


def _plain_copy(value):
    """Bounded deep copy without invoking user object hooks or decoding code."""
    remaining = MAX_VALUE_NODES

    def copy(node, depth):
        nonlocal remaining
        remaining -= 1
        _require(remaining >= 0 and depth <= MAX_VALUE_DEPTH, 'value_resource_limit')
        kind = type(node)
        if node is None or kind is bool:
            return node
        if kind is str:
            _require(len(node) <= MAX_REQUEST_BYTES, 'value_resource_limit')
            return node
        if kind is int:
            _require(node.bit_length() <= 256, 'integer_resource_limit')
            return node
        if kind is float:
            _require(math.isfinite(node), 'nonfinite_number')
            return node
        if kind is list:
            _require(len(node) <= MAX_VALUE_NODES, 'value_resource_limit')
            return [copy(child, depth + 1) for child in node]
        if kind is dict:
            _require(len(node) <= MAX_VALUE_NODES, 'value_resource_limit')
            _require(all(type(key) is str and len(key) <= 256 for key in node),
                     'invalid_object_key')
            return {key: copy(child, depth + 1) for key, child in node.items()}
        raise CheckProtocolError('non_plain_value')

    result = copy(value, 0)
    try:
        size = len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8'))
    except (UnicodeEncodeError, ValueError):
        raise CheckProtocolError('invalid_value_encoding') from None
    _require(size <= MAX_REQUEST_BYTES, 'value_resource_limit')
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def _reject_constant(_value):
    raise CheckProtocolError('nonfinite_number')


def parse_check_request(text):
    """Parse exactly one JSON object, with no markdown or repair/resampling."""
    _require(type(text) is str and len(text) <= MAX_REQUEST_BYTES, 'invalid_request_text')
    try:
        _require(len(text.encode('utf-8')) <= MAX_REQUEST_BYTES, 'request_resource_limit')
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except CheckProtocolError:
        raise
    except (ValueError, UnicodeEncodeError, RecursionError, MemoryError):
        raise CheckProtocolError('invalid_json') from None
    return _validate_request(value)


def _validate_request(value):
    value = _plain_copy(value)
    _require(type(value) is dict and set(value) == {'check', 'tool'}, 'invalid_envelope')
    _require(type(value['check']) is str and 0 < len(value['check'].strip()) <= 2048
             and len(value['check']) <= 2048, 'invalid_check_text')
    tool = value['tool']
    if tool is None:
        return value
    _require(type(tool) is dict and set(tool) == {'app', 'api', 'arguments'},
             'invalid_tool_fields')
    _require(_name(tool['app']) and _name(tool['api']), 'invalid_endpoint_name')
    _require(type(tool['arguments']) is dict and len(tool['arguments']) <= 32,
             'invalid_arguments')
    for key, item in tool['arguments'].items():
        _require(_name(key) and key not in _RESERVED, 'reserved_argument')
        _require(type(item) is dict, 'invalid_argument_value')
        if set(item) == {'literal'}:
            continue
        _require(set(item) in ({'variable'}, {'variable', 'path'}), 'invalid_value_reference')
        _require(_name(item['variable']) and item['variable'] not in _RESERVED,
                 'invalid_variable_name')
        path = item.get('path', [])
        _require(type(path) is list and len(path) <= MAX_VALUE_DEPTH, 'invalid_variable_path')
        _require(all((type(k) is str and not k.startswith('_') and len(k) <= 256)
                     or (type(k) is int and 0 <= k <= 1_000_000) for k in path),
                 'invalid_variable_path')
    return value


def resolve_check_request(request, namespace, allowed_apis):
    """Resolve plain existing values and deny unlisted endpoints before any call."""
    request = _validate_request(request)
    _require(type(namespace) is dict and all(type(key) is str for key in namespace),
             'invalid_namespace')
    _require(type(allowed_apis) is frozenset and all(
        type(pair) is tuple and len(pair) == 2 and all(_name(x) for x in pair)
        and pair[0] != 'admin' and pair != ('supervisor', 'complete_task')
        for pair in allowed_apis), 'invalid_allowlist')
    tool = request['tool']
    if tool is None:
        return None
    endpoint = (tool['app'], tool['api'])
    _require(endpoint in allowed_apis, 'endpoint_not_allowed')
    arguments = {}
    for key, item in tool['arguments'].items():
        if 'literal' in item:
            value = item['literal']
        else:
            name = item['variable']
            _require(name in namespace, 'variable_not_available')
            value = namespace[name]
            for part in item.get('path', []):
                if type(value) is dict and type(part) is str:
                    _require(all(type(key) is str for key in value), 'non_plain_variable_key')
                    _require(part in value, 'variable_path_not_available')
                    value = value[part]
                elif type(value) is list and type(part) is int:
                    _require(part < len(value), 'variable_path_not_available')
                    value = value[part]
                else:
                    raise CheckProtocolError('non_plain_variable_path')
        arguments[key] = _plain_copy(value)
    return (*endpoint, _plain_copy(arguments))


def dispatch_check(text, *, namespace, allowed_apis, request, requests_count):
    """One trusted endpoint call at most. Backend failures propagate without retry.

    This function does not own a world, deadline, events or cleanup. Its caller
    must persist intent before invocation and retain unknown costs on exception.
    Response text is actor-visible and must remain in private trace storage.
    """
    parsed = parse_check_request(text)
    resolved = resolve_check_request(parsed, namespace, allowed_apis)
    _require(callable(request) and callable(requests_count), 'invalid_backend')
    if resolved is None:
        return {'status': 'skipped', 'check': parsed['check'], 'api_calls': 0,
                'observation': 'No inspection API requested; no inspection evidence obtained.'}
    before = requests_count()
    _require(type(before) is int and before >= 0, 'invalid_api_counter')
    app, api, arguments = resolved
    response = request(app, api, track=True, raise_on_failure=False, **arguments)
    after = requests_count()
    _require(type(after) is int and after > before, 'invalid_api_counter')
    observation = json.dumps(response, ensure_ascii=False, allow_nan=False)
    _require(len(observation.encode('utf-8')) <= 1_000_000, 'observation_resource_limit')
    return {'status': 'observed', 'check': parsed['check'], 'api_calls': after - before,
            'observation': observation}
