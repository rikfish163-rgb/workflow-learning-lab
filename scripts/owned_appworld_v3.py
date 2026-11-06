"""Seeded AppWorld host with write-ahead container ownership and safe cleanup.

The callback must durably record intent before returning. An intent is not proof
that a container was created: cleanup checks its image and private labels and
only removes the inspected immutable container ID.
"""

import json
import os
import re
import selectors
import subprocess
import uuid
from pathlib import Path

from wll.appworld import StderrTail
from wll.appworld_v2 import AppWorldEnvironmentV2, AppWorldWorkerV2, _verified_seed_provenance

OWNER_TOKEN_LABEL = 'org.wll.owner-token'
OWNER_PID_LABEL = 'org.wll.owner-pid'
OWNER_TICKS_LABEL = 'org.wll.owner-start-ticks'
QUERY_TIMEOUT = 5
REMOVE_TIMEOUT = 15


def _validate_intent(intent):
    if (type(intent) is not dict or set(intent) != {
            'container_name', 'image', 'owner_pid', 'owner_start_ticks', 'ownership_token'}):
        raise ValueError('invalid container intent fields')
    token, name, image = intent['ownership_token'], intent['container_name'], intent['image']
    if (type(token) is not str or re.fullmatch('[0-9a-f]{32}', token) is None
            or type(name) is not str or name != 'wll-' + token[:16]
            or type(image) is not str or re.fullmatch('sha256:[0-9a-f]{64}', image) is None
            or type(intent['owner_pid']) is not int or intent['owner_pid'] <= 0
            or type(intent['owner_start_ticks']) is not int or intent['owner_start_ticks'] <= 0):
        raise ValueError('invalid container intent identity')


def inspect_owned_container(intent):
    """Return a verified immutable container ID, or None if the exact name is absent.

    Docker transport errors and identity mismatches propagate; they never count
    as an absent container. The caller must separately account for races in a
    still-running or not-yet-complete Docker creation request.
    """
    _validate_intent(intent)
    name = intent['container_name']
    result = subprocess.run(['docker', 'ps', '-a', '--no-trunc', '--filter', f'name=^/{name}$',
                             '--format', '{{.ID}}'], capture_output=True, text=True,
                            timeout=QUERY_TIMEOUT, check=True)
    ids = result.stdout.splitlines()
    if not ids:
        return None
    if len(ids) != 1 or re.fullmatch('[0-9a-f]{64}', ids[0]) is None:
        raise ValueError('unexpected container ID list')
    identity = ids[0]
    result = subprocess.run(['docker', 'inspect', '--type', 'container', identity],
                            capture_output=True, text=True, timeout=QUERY_TIMEOUT, check=True)
    parsed = json.loads(result.stdout)
    if type(parsed) is not list or len(parsed) != 1 or type(parsed[0]) is not dict:
        raise ValueError('invalid container inspection')
    container = parsed[0]
    config = container.get('Config')
    labels = config.get('Labels') if type(config) is dict else None
    expected_labels = {OWNER_TOKEN_LABEL: intent['ownership_token'],
                       OWNER_PID_LABEL: str(intent['owner_pid']),
                       OWNER_TICKS_LABEL: str(intent['owner_start_ticks'])}
    if (container.get('Id') != identity or container.get('Name') != '/' + name
            or container.get('Image') != intent['image'] or type(config) is not dict
            or config.get('Image') != intent['image'] or type(labels) is not dict
            or any(labels.get(key) != value for key, value in expected_labels.items())):
        raise ValueError('container identity or ownership labels differ')
    return identity


def remove_owned_container(intent):
    """Remove only the verified immutable ID; an absent name needs no removal."""
    identity = inspect_owned_container(intent)
    if identity is None:
        return None
    subprocess.run(['docker', 'rm', '-f', identity], capture_output=True, text=True,
                   timeout=REMOVE_TIMEOUT, check=True)
    return identity


class OwnedAppWorldWorkerV3(AppWorldWorkerV2):
    """Frozen RPC with an ownership-intent hook before Docker process creation."""

    def __init__(self, data_dir, image, deps_dir=None, timeout=120, *, on_container_intent):
        if not callable(on_container_intent):
            raise ValueError('container intent callback required')
        if type(image) is not str or re.fullmatch('sha256:[0-9a-f]{64}', image) is None:
            raise ValueError('image must be an immutable sha256 ID')
        data_dir = Path(data_dir).resolve(strict=True)
        if ',' in str(data_dir):
            raise ValueError('comma is not supported in Docker mount paths')
        if deps_dir:
            deps_dir = Path(deps_dir).resolve(strict=True)
            if ',' in str(deps_dir):
                raise ValueError('comma is not supported in Docker mount paths')

        token = uuid.uuid4().hex
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        intent = {'container_name': 'wll-' + token[:16], 'image': image, 'owner_pid': os.getpid(),
                  'owner_start_ticks': ticks, 'ownership_token': token}
        _validate_intent(intent)
        self._intent, self.name, self.timeout = intent, intent['container_name'], timeout
        self._closed = False
        self.proc = self.selector = self.error_log = None
        command = ['docker', 'run', '--rm', '-i', '--name', self.name,
                   '--label', f'{OWNER_TOKEN_LABEL}={token}',
                   '--label', f'{OWNER_PID_LABEL}={intent["owner_pid"]}',
                   '--label', f'{OWNER_TICKS_LABEL}={ticks}',
                   '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges', '--pids-limit', '128',
                   '--memory', '3g', '--cpus', '2', '--user', '10001:10001',
                   '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m,uid=10001,gid=10001',
                   '--tmpfs', '/workspace/experiments:rw,nosuid,nodev,size=1g,uid=10001,gid=10001',
                   '--mount', f'type=bind,src={data_dir},dst=/workspace/data,readonly']
        if deps_dir:
            command.extend(['--mount', f'type=bind,src={deps_dir},dst=/opt/deps,readonly'])
        command.append(image)
        on_container_intent(dict(intent))
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
            self.error_log = StderrTail(self.proc.stderr)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        except BaseException:
            if self.proc is not None:
                self.close()
            raise

    def close(self):
        if self._closed:
            return
        errors = []
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait(timeout=10)
            except BaseException as exc:
                errors.append(exc)
            try:
                remove_owned_container(self._intent)
            except BaseException as exc:
                errors.append(exc)
        if self.selector is not None:
            try:
                self.selector.close()
            except BaseException as exc:
                errors.append(exc)
        if self.proc is not None:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except BaseException as exc:
                    errors.append(exc)
            if self.error_log is not None:
                try:
                    self.error_log.thread.join(timeout=2)
                except BaseException as exc:
                    errors.append(exc)
            try:
                self.proc.stderr.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        self._closed = True


class OwnedAppWorldEnvironmentV3(AppWorldEnvironmentV2):
    """A seed-verified environment using an owned-container worker."""

    def __init__(self, task_id, data_dir, max_steps, image, deps_dir, environment_seed=100,
                 *, on_container_intent):
        if type(environment_seed) is not int or environment_seed != 100:
            raise ValueError('environment seed must be the registered integer 100')
        if type(task_id) is not str or not task_id:
            raise ValueError('task_id must be a nonempty string')
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self.worker = OwnedAppWorldWorkerV3(data_dir, image, deps_dir,
                                            on_container_intent=on_container_intent)
        self.evaluated = False
        try:
            result = self.worker.call('start', task_id=task_id, max_steps=max_steps,
                                      environment_seed=environment_seed)
            self._environment_seed_provenance = _verified_seed_provenance(result, task_id, environment_seed)
        except BaseException:
            self.worker.close()
            raise
