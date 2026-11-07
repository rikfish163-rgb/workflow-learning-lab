"""One recorded pre-dispatch refusal for a narrowly defined state write.

This dispatcher performs no additional reads or writes. The model must issue
its own state reads and correction. The private journal callback must durably
persist each receipt. Full worker and episode authentication belongs to the
caller; a receipt alone does not establish recovery or task success.
"""

import ast
import copy
import hashlib
from pathlib import Path

from scripts import native_appworld_tools as original
from wll.protocol import fingerprint

POLICY = 'native_state_recovery_predispatch_v1'
MESSAGE = 'Controlled transient tool refusal: request was not executed.'
SOURCE_SHA256 = 'c3afd29dc376150dca7d680e5d8e6114736ca1c9a9db1732c96ccde6cdeafa96'
PAIRS = {
    'todoist__update_task': ('task_id', 'is_completed'),
    'simple_note__update_note': ('note_id', 'pinned'),
}


def _dispatch_method():
    path = Path(original.__file__)
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('state_refusal_dispatcher_source_changed')
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ToolDispatcher')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'dispatch')
    source = ast.get_source_segment(raw.decode(), function)
    marker = '        self._api_invocations += 1'
    if source.count(marker) != 1:
        raise ValueError('state_refusal_dispatcher_boundary_changed')
    source = source.replace(marker, '        self._before_dispatch(name, arguments)\n' + marker)
    scope = dict(vars(original))
    # Only verified repository source is compiled. No action text is executed.
    exec(compile(source, str(path), 'exec', dont_inherit=True), scope)
    return scope['dispatch']


class StateRefusalDispatcher(original.ToolDispatcher):
    def __init__(self, public_schemas, apis, *, record, **kwargs):
        if not callable(record):
            raise ValueError('state_refusal_requires_durable_recorder')
        super().__init__(public_schemas, apis, **kwargs)
        self._record_fault = record
        self._injected = False
        self._record_failed = False

    def _before_dispatch(self, name, arguments):
        if self._record_failed:
            raise RuntimeError('state_refusal_journal_failed')
        if self._injected or name not in PAIRS:
            return
        resource, field = PAIRS[name]
        if not (set(arguments) == {'access_token', resource, field}
                and type(arguments['access_token']) is str and bool(arguments['access_token'])
                and type(arguments[resource]) is int and arguments[resource] > 0
                and type(arguments[field]) is bool):
            return
        receipt = dict(policy=POLICY, fault_index=self.action_attempts - 1,
                       action_sha256=fingerprint({'name': name, 'arguments': arguments}),
                       injected_count=1, backend_dispatches_at_refusal=0)
        self._injected = True
        try:
            self._record_fault(copy.deepcopy(receipt))
        except BaseException:
            self._record_failed = True
            raise RuntimeError('state_refusal_journal_failed') from None
        raise original.ToolDispatchError(MESSAGE)

    dispatch = _dispatch_method()
