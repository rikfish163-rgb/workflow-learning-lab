"""Source-bound checker read policies for twelve additional public endpoints.

Old six policies are delegated unchanged. Only the listed state-read paths are
added; downloads, player initialization, and optional reviewed extensions remain
denied. Authentication, connection caches, request accounting and logs still run.
The bound sources are a selected review scope, not a full dependency closure or
formal isolation proof. Shapes and bounds do not attest clause coverage or task
success. Dispatch caps, upstream safety, and world isolation belong to the worker.
"""

import hashlib
import json
import math

from scripts import condition_readonly_registry as previous
from scripts.completion_check_protocol import CheckProtocolError, _plain_copy

PROTOCOL = 'wll-verification-readonly-v3'
PUBLIC_APPS = previous.PUBLIC_APPS
SOURCE_SCOPE = 'selected_public_query_paths_and_schemas_not_full_dependency_closure'

# Plain metadata drives validation and lets the actor documentation use exactly
# the accepted fields. Defaults describe upstream behavior; they are never added
# to, removed from, or otherwise substituted into the dispatched arguments.
ARGUMENT_RULES = {
    'token': 'Exact str, 1..8192 printable ASCII characters, no whitespace.',
    'id': 'Exact int, 1..2**63-1; bool is not an ID.',
    'nullable_id': 'None or exact int, 1..2**63-1.',
    'section_id': 'None or exact int, 0..2**63-1; 0 means without a section.',
    'query': 'Exact str, 0..4096 characters; search is not exact-match proof.',
    'tags': 'None or exact list of at most 32 exact nonempty strings, each <=256 characters.',
    'bool': 'Exact bool.',
    'nullable_bool': 'None or exact bool.',
    'false': 'Exact False only; deleted=True is denied for collection coverage, not a write.',
    'email': 'None or bounded ASCII dot-atom/DNS email, at most 254 characters.',
    'priority': 'None or exact str high, medium, or low.',
    'date': 'Exact YYYY-MM-DD calendar date, years 0001..9999.',
    'amount': 'Exact int/float, finite and within 0..2**63-1; bool is not an amount.',
    'phone': 'None or exact nonempty str, at most 64 characters; no coercion.',
    'page_index': 'Exact int, 0..2**63-1; omitted upstream default is 0.',
    'page_limit': 'Exact int, 1..20; omitted upstream default is 5.',
    'notes_sort': 'None, empty str, or one +/- field: created_at, updated_at, pinned.',
    'tasks_sort': 'Empty str or one +/- field: order_index, due_date, priority; no None.',
    'playlists_sort': 'None, empty str, or one +/- field: created_at, title.',
    'orders_sort': 'None, empty str, or one +/- field: created_at, paid_amount.',
    'expenses_sort': 'None, empty str, or one +/- field: created_at, amount.',
    'messages_sort': 'None, empty str, or one +/- field: created_at.',
}

ENDPOINT_POLICIES = {
    ('simple_note', 'search_notes'): {
        'required': {'access_token': 'token'},
        'optional': {
            'query': 'query', 'tags': 'tags', 'pinned': 'nullable_bool',
            'dont_reorder_pinned': 'nullable_bool', 'page_index': 'page_index',
            'page_limit': 'page_limit', 'sort_by': 'notes_sort',
        },
        'defaults': {
            'query': '', 'tags': None, 'pinned': None, 'dont_reorder_pinned': None,
            'page_index': 0, 'page_limit': 5, 'sort_by': None,
        },
    },
    ('simple_note', 'show_note'): {
        'required': {'access_token': 'token', 'note_id': 'id'},
        'optional': {}, 'defaults': {},
    },
    ('todoist', 'show_tasks'): {
        'required': {'access_token': 'token', 'project_id': 'id'},
        'optional': {
            'section_id': 'section_id', 'assignee_email': 'email',
            'assigner_email': 'email', 'priority': 'priority',
            'is_completed': 'nullable_bool', 'due_today': 'bool', 'overdue': 'bool',
            'label_id': 'nullable_id', 'min_due_date': 'date', 'max_due_date': 'date',
            'sort_by': 'tasks_sort',
        },
        'defaults': {
            'section_id': None, 'assignee_email': None, 'assigner_email': None,
            'priority': None, 'is_completed': None, 'due_today': False, 'overdue': False,
            'label_id': None, 'min_due_date': '1500-01-01', 'max_due_date': '3000-01-01',
            'sort_by': '+order_index',
        },
    },
    ('todoist', 'show_task'): {
        'required': {'access_token': 'token', 'task_id': 'id'},
        'optional': {}, 'defaults': {},
    },
    ('spotify', 'show_playlist_library'): {
        'required': {'access_token': 'token'},
        'optional': {
            'query': 'query', 'is_public': 'nullable_bool', 'page_index': 'page_index',
            'page_limit': 'page_limit', 'sort_by': 'playlists_sort',
        },
        'defaults': {
            'query': '', 'is_public': None, 'page_index': 0, 'page_limit': 5, 'sort_by': None,
        },
    },
    ('spotify', 'show_playlist'): {
        'required': {'playlist_id': 'id'},
        'optional': {'access_token': 'token'}, 'defaults': {},
    },
    ('amazon', 'show_orders'): {
        'required': {'access_token': 'token'},
        'optional': {
            'query': 'query', 'page_index': 'page_index', 'page_limit': 'page_limit',
            'sort_by': 'orders_sort',
        },
        'defaults': {'query': '', 'page_index': 0, 'page_limit': 5, 'sort_by': None},
    },
    ('amazon', 'show_order'): {
        'required': {'access_token': 'token', 'order_id': 'id'},
        'optional': {}, 'defaults': {},
    },
    ('splitwise', 'show_group_expenses'): {
        'required': {'access_token': 'token', 'group_id': 'id'},
        'optional': {
            'query': 'query', 'participant_email': 'email', 'min_amount': 'amount',
            'max_amount': 'amount', 'min_created_at': 'date', 'max_created_at': 'date',
            'deleted': 'false', 'page_index': 'page_index', 'page_limit': 'page_limit',
            'sort_by': 'expenses_sort',
        },
        'defaults': {
            'query': '', 'participant_email': None, 'min_amount': 0,
            'max_amount': 2**63 - 1, 'min_created_at': '1500-01-01',
            'max_created_at': '3000-01-01', 'deleted': False, 'page_index': 0,
            'page_limit': 5, 'sort_by': None,
        },
    },
    ('splitwise', 'show_expense'): {
        'required': {'access_token': 'token', 'expense_id': 'id'},
        'optional': {}, 'defaults': {},
    },
    ('phone', 'search_text_messages'): {
        'required': {'access_token': 'token'},
        'optional': {
            'query': 'query', 'phone_number': 'phone', 'only_latest_per_contact': 'bool',
            'page_index': 'page_index', 'page_limit': 'page_limit', 'sort_by': 'messages_sort',
        },
        'defaults': {
            'query': '', 'phone_number': None, 'only_latest_per_contact': False,
            'page_index': 0, 'page_limit': 5, 'sort_by': None,
        },
    },
    ('phone', 'show_text_message'): {
        'required': {'access_token': 'token', 'text_message_id': 'id'},
        'optional': {}, 'defaults': {},
    },
}
ALLOWED_APIS = previous.ALLOWED_APIS | frozenset(ENDPOINT_POLICIES)

SOURCE_SHA256 = {
    **previous.SOURCE_SHA256,
    'apps/simple_note/apis.py': 'd26d2db3358ca14740bf4a82c93788156a12e02c8a7fbe6fff8f03221c7d1bf1',
    'apps/simple_note/models.py': 'effc919d4bbd9c7128ac8c1aa1bc37f43b3c86c7fb4908891e93293234eca835',
    'apps/todoist/apis.py': 'b3f62c7c5c5d0e26909e2aae987644d5ba68dc0f0ebcbb7b9bff6c4fdb1ec681',
    'apps/todoist/models.py': 'e926571044893bd1244ffe4316564ce48cda6054ada54a1ed0541ce3350a8104',
    'apps/spotify/apis.py': 'e5a7f6373df97fbe2d6c7b6b9712071ec0731af976bacb50d8395395f78ee779',
    'apps/spotify/models.py': 'a2ec6492df4ae2285a661852ab4212aa36e70af9b9a7a8a9b199800a6d0a298a',
    'apps/amazon/apis.py': '3ac6651b718426cb681cab86e2ba494f5712f542c58c5089f77df8103dd8a853',
    'apps/amazon/models.py': '26f914d079a74ea2e95bdb181a7f2829d826a2eea7330cdc34c3d08f09ffdf5e',
    'apps/splitwise/apis.py': '642ab1509e1ede9b5940cedc1ff15869740818c6326f33814f2f2a2e6e30d8d4',
    'apps/splitwise/models.py': '84a702b81d8f3939c4e875d83154397b406fbe57467a8719a01cd52a7d283146',
    'apps/phone/apis.py': 'da4e24e846845bbc8c56b6aac807372ea9612d9db16bbb0f8eaefd2bd169974b',
    'apps/phone/models.py': '66a3b52b8cd7c01c201ec02cfa7125745ade6f0aea057ac190f54fe8652ca5fb',
    'apps/simple_note/responses.py': '4ecc417d0f26bfe11915d12d715488bf73ed8b604ecf74b4e17453bb87cc2c6f',
    'apps/todoist/responses.py': '3ebc931dfac6294c30b628403bc95db1adfcc9111e7791604225437bd6853085',
    'apps/spotify/responses.py': 'ddbf7dc65f88c7861105db32181a6a05d9cdfe481d2f6aa8ba018577ab690cb6',
    'apps/amazon/responses.py': 'e7236e685042cd7e487f7ba4102760d3873997a760b9a3334631824f902c7992',
    'apps/splitwise/responses.py': 'f6bfd7949d25a470f4b25a4039b15b0977fd97914c8d9dc184a5074bbab4278a',
    'apps/phone/responses.py': '48743efd87bb1ce6dfc3db977239fb14c3742a1b8489f840cd73fcfc7106b6e3',
    'apps/todoist/constants.py': '7b9eacc1336401eb3588f9ef606be4b996b6e3d12ba2b050ebd50ee34f24cf20',
    'api_docs.py': '58671e9e4b7ba94060c9af7be0c5c2fc3ac936ea5129729d194d0dabbaa58e51',
    'common/collections.py': 'a35aa3ef4af05f1fc7a72387d2480a4f31043ebfad6dbbc6003b94d52b53e84a',
}


def capabilities():
    material = {
        'protocol': PROTOCOL,
        'allowed_apis': [list(pair) for pair in sorted(ALLOWED_APIS)],
        'public_apps': sorted(PUBLIC_APPS),
        'source_sha256': SOURCE_SHA256,
        'argument_rules': ARGUMENT_RULES,
        'endpoint_policies': [
            {'app': app, 'api': api, **policy}
            for (app, api), policy in sorted(ENDPOINT_POLICIES.items())
        ],
        'source_scope': SOURCE_SCOPE,
    }
    return {
        'protocol': PROTOCOL,
        'registry_sha256': hashlib.sha256(
            json.dumps(material, sort_keys=True).encode()).hexdigest(),
        'allowed_apis': material['allowed_apis'],
        'counter': previous.capabilities()['counter'],
    }


def verify_installed_sources(appworld_root):
    for name, expected in SOURCE_SHA256.items():
        if hashlib.sha256((appworld_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('readonly_source_mismatch')


def _integer(value, minimum):
    return type(value) is int and minimum <= value < 2**63


def _text(value, minimum, maximum):
    return type(value) is str and minimum <= len(value) <= maximum


def _sort(value, fields, nullable=True):
    return ((nullable and value is None) or
            (type(value) is str and (value == '' or
             (len(value) >= 2 and value[0] in '+-' and value[1:] in fields))))


def _valid(rule, value):
    if rule == 'token':
        return (_text(value, 1, 8192) and value.isascii()
                and all(32 < ord(char) < 127 for char in value))
    if rule in {'id', 'nullable_id'}:
        return (rule == 'nullable_id' and value is None) or _integer(value, 1)
    if rule in {'page_index', 'section_id'}:
        return (rule == 'section_id' and value is None) or _integer(value, 0)
    if rule == 'page_limit':
        return type(value) is int and 1 <= value <= 20
    if rule == 'query':
        return _text(value, 0, 4096)
    if rule == 'tags':
        return (value is None or (type(value) is list and len(value) <= 32
                                  and all(_text(tag, 1, 256) for tag in value)))
    if rule in {'bool', 'nullable_bool'}:
        return (rule == 'nullable_bool' and value is None) or type(value) is bool
    if rule == 'false':
        return value is False
    if rule == 'email':
        return value is None or previous._valid_email(value)
    if rule == 'priority':
        return value is None or (type(value) is str and value in ('high', 'medium', 'low'))
    if rule == 'date':
        return previous._valid_date(value)
    if rule == 'amount':
        return (type(value) in (int, float) and 0 <= value <= 2**63 - 1
                and math.isfinite(value))
    if rule == 'phone':
        return value is None or _text(value, 1, 64)
    sorts = {
        'notes_sort': ('created_at', 'updated_at', 'pinned'),
        'tasks_sort': ('order_index', 'due_date', 'priority'),
        'playlists_sort': ('created_at', 'title'),
        'orders_sort': ('created_at', 'paid_amount'),
        'expenses_sort': ('created_at', 'amount'),
        'messages_sort': ('created_at',),
    }
    return rule in sorts and _sort(value, sorts[rule], nullable=rule != 'tasks_sort')


def _cross_fields(pair, arguments, policy):
    defaults = policy['defaults']
    if pair == ('todoist', 'show_tasks'):
        low = arguments.get('min_due_date', defaults['min_due_date'])
        high = arguments.get('max_due_date', defaults['max_due_date'])
        today = arguments.get('due_today', False)
        overdue = arguments.get('overdue', False)
        return (low <= high and not (today and overdue)
                and not ((today or overdue)
                         and (low != defaults['min_due_date'] or high != defaults['max_due_date'])))
    if pair == ('splitwise', 'show_group_expenses'):
        return (arguments.get('min_created_at', defaults['min_created_at'])
                <= arguments.get('max_created_at', defaults['max_created_at'])
                and arguments.get('min_amount', defaults['min_amount'])
                <= arguments.get('max_amount', defaults['max_amount']))
    return True


def _new_policy(app, api, reason):
    if type(app) is not str or type(api) is not str:
        raise CheckProtocolError(reason)
    pair = app, api
    if pair not in ALLOWED_APIS or pair not in ENDPOINT_POLICIES:
        raise CheckProtocolError(reason)
    return pair, ENDPOINT_POLICIES[pair]


def validate_inspection_arguments(app, api, arguments):
    """Fail before dispatch; no defaults, coercion, login or request is performed."""
    if type(app) is str and type(api) is str and (app, api) in previous.ALLOWED_APIS:
        return previous.validate_inspection_arguments(app, api, arguments)
    reason = 'inspection_arguments_not_allowed'
    pair, policy = _new_policy(app, api, reason)
    required, optional = policy['required'], policy['optional']
    if (type(arguments) is not dict or len(arguments) > len(required) + len(optional)
            or any(type(key) is not str for key in arguments)):
        raise CheckProtocolError(reason)
    if not required.keys() <= arguments.keys() <= required.keys() | optional.keys():
        raise CheckProtocolError(reason)
    rules = required | optional
    if any(not _valid(rules[key], value) for key, value in arguments.items()):
        raise CheckProtocolError(reason)
    if not _cross_fields(pair, arguments, policy):
        raise CheckProtocolError(reason)
    try:
        _plain_copy(arguments)
    except CheckProtocolError:
        raise CheckProtocolError(reason) from None


def validate_inspection_response(app, api, response):
    """Check bounded JSON/top-level shape; do not assert predicate/task semantics."""
    if type(app) is str and type(api) is str and (app, api) in previous.ALLOWED_APIS:
        return previous.validate_inspection_response(app, api, response)
    reason = 'inspection_response_not_allowed'
    _pair, policy = _new_policy(app, api, reason)
    try:
        _plain_copy(response)
    except CheckProtocolError:
        raise CheckProtocolError(reason) from None
    collection = 'page_index' in policy['optional']
    if ((collection and (type(response) is not list
                         or any(type(item) is not dict for item in response)))
            or (not collection and type(response) is not dict)):
        raise CheckProtocolError(reason)

