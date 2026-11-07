"""Owned Docker transport for authenticated, human-authored worker modules.

Only the container compiles the trusted bootstrap bundle. Model Python is sent
later as an execute RPC and never evaluated by this host module.
"""
import hashlib
import json
import math
import os
import re
import selectors
import subprocess
import time
import uuid
from pathlib import Path

from scripts.owned_appworld_v3 import (
    OWNER_PID_LABEL,
    OWNER_TICKS_LABEL,
    OWNER_TOKEN_LABEL,
    OwnedAppWorldWorkerV3,
    _validate_intent,
)
from wll.appworld import StderrTail

BOOT_PROTOCOL = 'wll-codeact-trusted-bootstrap-v1'
MODULES = ('scripts.codeact_execution_observer_001', 'scripts.codeact_within_worker_state_001',
           'scripts.codeact_within_worker_001')
IMAGE_SOURCES = ('scripts/completion_actor_boundary.py', 'scripts/completion_check_protocol.py',
                 'scripts/completion_readonly_registry.py', 'scripts/completion_worker_state.py',
                 'scripts/condition_readonly_registry.py', 'scripts/verification_readonly_registry_v3.py',
                 'wll/code_policy.py')
MAX_FRAME = 4_000_000

# This fixed public program runs inside the pinned image, never on the host.
BOOTSTRAP = r'''
import hashlib, importlib, json, os, re, sys, types
from pathlib import Path
if (os.getuid() != 10001 or not Path('/.dockerenv').is_file()
        or os.environ.get('PYTHONHASHSEED') != '100'):
    raise RuntimeError('bootstrap_container_required')
pins = json.loads(sys.argv[1])
sys.path[:0] = ['/opt/completion', '/opt/deps']
for name, expected in pins['image_source_sha256'].items():
    raw = (Path('/opt/completion') / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('bootstrap_image_source_binding')
raw = sys.stdin.buffer.readline(2_000_001)
if len(raw) > 2_000_000 or not raw.endswith(b'\n'):
    raise RuntimeError('bootstrap_frame')
packet = json.loads(raw)
if (set(packet) != {'protocol', 'modules'}
        or packet['protocol'] != 'wll-codeact-trusted-bootstrap-v1'
        or set(packet['modules']) != set(pins['source_sha256'])):
    raise RuntimeError('bootstrap_schema')
package = importlib.import_module('scripts')
for name, expected in pins['source_sha256'].items():
    source = packet['modules'][name]
    if type(source) is not str or hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError('bootstrap_source_binding')
    module = types.ModuleType(name)
    module.__file__ = '/opt/wll-trusted/' + name.rsplit('.', 1)[1] + '.py'
    module.__package__ = 'scripts'
    sys.modules[name] = module
    setattr(package, name.rsplit('.', 1)[1], module)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
print(json.dumps({'ok': True, 'result': {
    'protocol': 'wll-codeact-trusted-bootstrap-v1',
    'source_sha256': pins['source_sha256'],
    'image_source_sha256': pins['image_source_sha256']}}), flush=True)
raise SystemExit(sys.modules['scripts.codeact_within_worker_001'].main())
'''


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def source_bundle(sources, expected):
    if (type(sources) is not dict or type(expected) is not dict
            or set(sources) != set(MODULES) or set(expected) != set(MODULES)):
        raise ValueError('bootstrap_module_binding')
    modules = {}
    for name in MODULES:
        path = Path(sources[name])
        if (path.is_symlink() or not path.is_file() or type(expected[name]) is not str
                or re.fullmatch('[0-9a-f]{64}', expected[name]) is None):
            raise ValueError('bootstrap_source_binding')
        raw = path.read_bytes()
        if len(raw) > 500_000 or sha256(raw) != expected[name]:
            raise ValueError('bootstrap_source_binding')
        modules[name] = raw.decode('utf-8', errors='strict')
    raw = json.dumps({'protocol': BOOT_PROTOCOL, 'modules': modules},
                     ensure_ascii=False, separators=(',', ':')).encode()
    if len(raw) + 1 > 2_000_000:
        raise ValueError('bootstrap_bundle_size')
    return raw


class CodeActOwnedWorker(OwnedAppWorldWorkerV3):
    def __init__(self, data_dir, image, deps_dir, *, sources, source_sha256,
                 image_source_sha256, on_container_intent, timeout=120):
        bundle = source_bundle(sources, source_sha256)
        if (not callable(on_container_intent) or type(image) is not str
                or re.fullmatch('sha256:[0-9a-f]{64}', image) is None
                or type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 0 < timeout <= 1500
                or type(image_source_sha256) is not dict
                or set(image_source_sha256) != set(IMAGE_SOURCES)
                or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None
                       for k, v in image_source_sha256.items())):
            raise ValueError('bootstrap_configuration')
        paths = [Path(value).resolve(strict=True) for value in (data_dir, deps_dir)]
        if any(not path.is_dir() or ',' in str(path) for path in paths):
            raise ValueError('bootstrap_mount')
        self.source_sha256 = {name: source_sha256[name] for name in MODULES}
        self.image_source_sha256 = dict(image_source_sha256)
        token = uuid.uuid4().hex
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        self._intent = {'container_name': 'wll-' + token[:16], 'image': image,
                        'owner_pid': os.getpid(), 'owner_start_ticks': ticks, 'ownership_token': token}
        _validate_intent(self._intent)
        self.name, self.timeout, self._closed = self._intent['container_name'], timeout, False
        self.proc = self.selector = self.error_log = None
        self._buffer = b''
        pins = {'source_sha256': self.source_sha256, 'image_source_sha256': self.image_source_sha256}
        command = ['docker', 'run', '--rm', '-i', '--name', self.name,
                   '--label', f'{OWNER_TOKEN_LABEL}={token}',
                   '--label', f'{OWNER_PID_LABEL}={os.getpid()}',
                   '--label', f'{OWNER_TICKS_LABEL}={ticks}', '--log-driver', 'none',
                   '--env', 'PYTHONHASHSEED=100', '--env', 'APPWORLD_ROOT=/workspace',
                   '--env', 'PYTHONPATH=/opt/completion:/opt/deps', '--env', 'CUDA_VISIBLE_DEVICES=',
                   '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges', '--pids-limit', '128',
                   '--memory', '3g', '--cpus', '2', '--user', '10001:10001',
                   '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m,uid=10001,gid=10001',
                   '--tmpfs', '/workspace/experiments:rw,nosuid,nodev,size=1g,uid=10001,gid=10001',
                   '--mount', f'type=bind,src={paths[0]},dst=/workspace/data,readonly',
                   '--mount', f'type=bind,src={paths[1]},dst=/opt/deps,readonly',
                   '--entrypoint', 'python', image, '-B', '-u', '-c', BOOTSTRAP,
                   json.dumps(pins, separators=(',', ':'))]
        on_container_intent(dict(self._intent))
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
            self.error_log = StderrTail(self.proc.stderr)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
            end = time.monotonic() + self.timeout
            self._send(bundle + b'\n', deadline=end)
            if self._receive(deadline=end) != {'protocol': BOOT_PROTOCOL, **pins}:
                raise RuntimeError('codeact_bootstrap_handshake')
        except BaseException:
            if self.proc is not None:
                self.close()
            raise

    def _send(self, raw, *, deadline=None):
        if type(raw) is not bytes or len(raw) > MAX_FRAME or not raw.endswith(b'\n'):
            raise ValueError('codeact_rpc_frame')
        end = time.monotonic() + self.timeout if deadline is None else deadline
        sent = 0
        fd = self.proc.stdin.fileno()
        os.set_blocking(fd, False)
        with selectors.DefaultSelector() as ready:
            ready.register(fd, selectors.EVENT_WRITE)
            while sent < len(raw):
                left = end - time.monotonic()
                if left <= 0 or not ready.select(left):
                    raise TimeoutError('codeact_rpc_send_deadline')
                try:
                    sent += os.write(fd, raw[sent:sent+65536])
                except BlockingIOError:
                    continue

    def _receive(self, *, deadline=None):
        end = time.monotonic() + self.timeout if deadline is None else deadline
        while b'\n' not in self._buffer:
            left = end - time.monotonic()
            if left <= 0 or not self.selector.select(left):
                raise TimeoutError('codeact_rpc_receive_deadline')
            chunk = os.read(self.proc.stdout.fileno(), min(65536, MAX_FRAME + 1 - len(self._buffer)))
            if not chunk:
                raise RuntimeError('codeact_rpc_eof')
            self._buffer += chunk
            if len(self._buffer) > MAX_FRAME:
                raise RuntimeError('codeact_rpc_size')
        line, self._buffer = self._buffer.split(b'\n', 1)
        reply = json.loads(line)
        if type(reply) is not dict or set(reply) not in ({'ok', 'result'}, {'ok', 'error'}):
            raise RuntimeError('codeact_rpc_schema')
        if reply.get('ok') is not True or set(reply) != {'ok', 'result'}:
            raise RuntimeError('codeact_rpc_worker_error')
        return reply['result']

    def call(self, op, **kwargs):
        if self._closed or self.proc.poll() is not None:
            raise RuntimeError('codeact_worker_unavailable')
        try:
            end = time.monotonic() + self.timeout
            self._send((json.dumps({'op': op, **kwargs}, ensure_ascii=False) + '\n').encode(), deadline=end)
            return self._receive(deadline=end)
        except BaseException:
            self.close()
            raise
