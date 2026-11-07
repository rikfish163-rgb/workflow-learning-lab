"""Explicit endpoint registry for the installed, source-bound AppWorld version.

Read-only means no modeled application-state write on the reviewed query path;
framework authentication, query caches, request accounting and logs still run.
It is not inferred from GET or a name. All other endpoints are denied, including
known lazy initialization/download queries. Source review is not formal proof.
"""

import hashlib
import json
import re

from scripts.completion_check_protocol import CheckProtocolError

PUBLIC_APPS = frozenset({'api_docs', 'supervisor', 'amazon', 'phone', 'file_system',
                         'spotify', 'venmo', 'gmail', 'splitwise', 'simple_note', 'todoist'})

ALLOWED_APIS = frozenset({
    ('api_docs', 'show_app_descriptions'),
    ('api_docs', 'show_api_descriptions'),
    ('api_docs', 'show_api_doc'),
    ('gmail', 'show_email'),
    ('file_system', 'show_file'),
})

SOURCE_SHA256 = {
    'environment.py': 'f988f1e6f257041e73682725522662674d4d65e8a1673787614f54b89fe2baeb',
    'requester.py': '06f587ae9311e6749cc4f282cf943870f64e9bb04b1384d5faaacc59681a4a62',
    'collections/apis.py': '582feb30779991858e708627b0b3c17b9e6477135f2e73bafa19e4e05a6ce735',
    'apps/api_docs/apis.py': '4c3ebe37c82f6d11c23a75773ad1505e0d673ede846e6c5e36eaec4e7ec2fc62',
    'apps/api_docs/models.py': 'e66bd14421acf4171e36e9f42afd33930cd528e6faa40eba382eac2cbcfc5ef2',
    'apps/gmail/apis.py': 'fab69586994ff5471dc8a7b4e1e002844cfa74002e6b40c41e30b41e1fe70932',
    'apps/gmail/models.py': 'ebc722194744185bf7ee07cc4b7383134157ae9e6ee63ae778527ecfe03aa2de',
    'apps/file_system/apis.py': '06120abe1423111f15f0c0dbe813bf30db689631bdba7653fb12812fcd67a357',
    'apps/file_system/models.py': '6f0ad5a7cae6e6161f4aaa0ed55628d10bd71afd36117a0d1332ff2adddea5d3',
    'apps/lib/apis/authentication.py': 'bc7b04789c9d7d9823d2c9f2df85d24519841981ac763fbcfd436b2779c99ed8',
    'apps/lib/apis/local_remote.py': 'b5e16eeb13147992aa55ca973ed1f522f0f4829a5b2ab62048c6de4a31c4920d',
    'apps/lib/models/orm.py': 'df959a70b1cb39acd0f40a8fbec103f88f4b3481532d475cd079a551eb43abb9',
    'apps/lib/models/db.py': '641fd9b70f03db3acbd6d564dce762a097a380a3adc105d9294c286889f4b521',
    'apps/lib/models/users.py': '0ee6c97827832d757cc57e3d0b03a9c38cc4e7ad85e6d47f7b37e068b4f7a565',
    'apps/__init__.py': 'e3fb5431de0f6ecadba41fe6eb22fbdb8d1383ca4a745e646caf1dcdc15a9214',
    '../fastapi_login/fastapi_login.py': '7e6eebfbd05d5fab064063a1bdb25fc1230a74d315bb7ae9a2f22df44d6be1c7',
}


def capabilities():
    registry = {'allowed_apis': [list(pair) for pair in sorted(ALLOWED_APIS)],
                'public_apps': sorted(PUBLIC_APPS), 'source_sha256': SOURCE_SHA256}
    digest = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
    return {'protocol': 'wll-completion-inspection-v1', 'registry_sha256': digest,
            'allowed_apis': registry['allowed_apis'],
            'counter': 'public_requester_request_attempts_excluding_trusted_status_probe_v1'}


def verify_installed_sources(appworld_root):
    for name, expected in SOURCE_SHA256.items():
        if hashlib.sha256((appworld_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('readonly_source_mismatch')


def validate_inspection_arguments(app, api, arguments):
    """Additional semantic limits, before the counted requester is invoked."""
    valid = False
    if app == 'api_docs':
        if api == 'show_app_descriptions':
            valid = arguments == {}
        elif api in {'show_api_descriptions', 'show_api_doc'}:
            expected = {'app_name'} if api == 'show_api_descriptions' else {'app_name', 'api_name'}
            valid = (set(arguments) == expected and type(arguments.get('app_name')) is str
                     and arguments['app_name'] in PUBLIC_APPS)
            if valid and api == 'show_api_doc':
                valid = (type(arguments['api_name']) is str
                         and re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}', arguments['api_name'])
                         is not None)
    elif (app, api) == ('gmail', 'show_email'):
        valid = (set(arguments) == {'access_token', 'email_id'}
                 and type(arguments['email_id']) is int and 0 < arguments['email_id'] < 2**63)
    elif (app, api) == ('file_system', 'show_file'):
        valid = (set(arguments) == {'access_token', 'file_path'}
                 and type(arguments['file_path']) is str and 0 < len(arguments['file_path']) <= 4096)
    if valid and app in {'gmail', 'file_system'}:
        valid = (type(arguments['access_token']) is str
                 and 0 < len(arguments['access_token']) <= 8192)
    if not valid:
        raise CheckProtocolError('inspection_arguments_not_allowed')


def validate_inspection_response(app, api, response):
    """Fail after dispatch if metadata enumeration contains a private app name."""
    if (app, api) == ('api_docs', 'show_app_descriptions'):
        if (type(response) is not list or any(
                type(item) is not dict or type(item.get('name')) is not str
                or item['name'] not in PUBLIC_APPS for item in response)):
            raise CheckProtocolError('unexpected_public_app_enumeration')
