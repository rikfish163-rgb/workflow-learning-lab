"""Separately versioned, source-bound public condition-query registry.

The original five endpoint policies are delegated to the frozen completion
registry. The only additional endpoint is the reviewed transaction read path;
authentication, framework caches, request accounting and logs still run.
"""

import hashlib
import json
import re
from datetime import date

from scripts import completion_readonly_registry as completion
from scripts.completion_check_protocol import CheckProtocolError

PROTOCOL = 'wll-condition-query-v1'
PUBLIC_APPS = completion.PUBLIC_APPS
ALLOWED_APIS = completion.ALLOWED_APIS | {('venmo', 'show_transactions')}

SOURCE_SHA256 = {
    **completion.SOURCE_SHA256,
    'apps/venmo/apis.py': '99771a7b32a76db37b8519024eadc3f3ac6cb40fd81b1e20ca04b0bf22982284',
    'apps/venmo/models.py': '4e07456bbc8b90c7c304d8ca64aa7d7bfd4b0a9cdca81d7fb7d2b6e0bc5f36f5',
    'apps/venmo/responses.py': '796759c24b10634129d03c51132a4cdd47ee2404c3c3421c1094cce2be7a8336',
    'apps/lib/models/filter_sort.py': '3734ba59dc99e1855a2ecbd5ab1ec11c4eeabe4a0405c95139c28832b370336d',
    'apps/lib/apis/pagination.py': '777fd4093140cfda6b63e2a224b6371c646de1ce08b9b80a03eb20a680161627',
    'apps/lib/apis/datetime.py': 'cf3e219b3ddebedc586b8beef49df89992d859c4906bdec7b2f1defca4357fd0',
    'apps/lib/responses/__init__.py': 'd57617093c71a52a0d47ee169188f2ce0c95258bdc4648fff46208ea0f678dae',
    'common/datetime.py': '3edd3cdf00c3437da7cd57e397e1d4047cc6383f090c1dc50127e289fa647e5c',
    'common/constants.py': '776a1c2a97e8f3d7cbda00523be3fdf38ed6165d61ac729a82290a843524138d',
}

_TRANSACTION_ARGUMENTS = frozenset({
    'access_token', 'user_email', 'direction', 'min_created_at', 'max_created_at',
    'page_index', 'page_limit',
})


def capabilities():
    registry = {
        'protocol': PROTOCOL,
        'allowed_apis': [list(pair) for pair in sorted(ALLOWED_APIS)],
        'public_apps': sorted(PUBLIC_APPS),
        'source_sha256': SOURCE_SHA256,
    }
    digest = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
    return {
        'protocol': PROTOCOL,
        'registry_sha256': digest,
        'allowed_apis': registry['allowed_apis'],
        'counter': completion.capabilities()['counter'],
    }


def verify_installed_sources(appworld_root):
    for name, expected in SOURCE_SHA256.items():
        if hashlib.sha256((appworld_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('readonly_source_mismatch')


def _valid_email(value):
    """Bounded ASCII dot-atom/DNS subset; AppWorld still validates EmailStr."""
    if type(value) is not str or not 0 < len(value) <= 254 or value.count('@') != 1:
        return False
    local, domain = value.split('@')
    if (not 0 < len(local) <= 64 or local.startswith('.') or local.endswith('.')
            or '..' in local or re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+", local) is None):
        return False
    labels = domain.split('.')
    return (len(labels) >= 2 and all(
        re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
        is not None for label in labels)
        and re.search(r'[A-Za-z]', labels[-1]) is not None)


def _valid_date(value):
    if (type(value) is not str or len(value) != 10
            or re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value) is None):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate_inspection_arguments(app, api, arguments):
    """Validate before requester dispatch; never inject or rewrite query defaults."""
    if (app, api) != ('venmo', 'show_transactions'):
        return completion.validate_inspection_arguments(app, api, arguments)
    valid = (type(arguments) is dict
             and {'access_token'} <= arguments.keys() <= _TRANSACTION_ARGUMENTS)
    if valid:
        token = arguments['access_token']
        valid = type(token) is str and 0 < len(token) <= 8192
    if valid and arguments.get('user_email') is not None:
        valid = _valid_email(arguments['user_email'])
    if valid and arguments.get('direction') is not None:
        direction = arguments['direction']
        # These are the public source Literal values, never a task-selected filter.
        valid = type(direction) is str and direction in ('sent', 'received')
    for name in ('min_created_at', 'max_created_at'):
        if valid and name in arguments:
            valid = _valid_date(arguments[name])
    if valid and {'min_created_at', 'max_created_at'} <= arguments.keys():
        valid = arguments['min_created_at'] <= arguments['max_created_at']
    if valid and 'page_index' in arguments:
        index = arguments['page_index']
        valid = type(index) is int and 0 <= index < 2**63
    if valid and 'page_limit' in arguments:
        limit = arguments['page_limit']
        valid = type(limit) is int and 1 <= limit <= 20
    if not valid:
        raise CheckProtocolError('inspection_arguments_not_allowed')


def validate_inspection_response(app, api, response):
    return completion.validate_inspection_response(app, api, response)
