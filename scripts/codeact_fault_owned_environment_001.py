"""Seed/capability-verified host facade; all model Python stays in Docker."""
from pathlib import Path

from scripts.codeact_fault_worker_001 import POLICY_SHA256, actor_observation, capabilities
from scripts.codeact_owned_worker_001 import CodeActOwnedWorker
from wll.appworld_v2 import _verified_seed_provenance


class CodeActFaultOwnedEnvironment:
    def __init__(self, task_id, data_dir, max_steps, image, deps_dir, *,
                 on_container_intent, sources, source_sha256, image_source_sha256):
        if (type(task_id) is not str or not task_id or type(max_steps) is not int
                or not 1 <= max_steps <= 40):
            raise ValueError('codeact_world_configuration')
        directory = Path(__file__).resolve().parent
        expected = {
            'scripts.codeact_execution_observer_001': directory / 'codeact_execution_observer_001.py',
            'scripts.codeact_within_worker_state_001': directory / 'codeact_fault_worker_state_001.py',
            'scripts.codeact_within_worker_001': directory / 'codeact_fault_worker_001.py',
        }
        if (type(sources) is not dict or set(sources) != set(expected)
                or any(Path(sources[key]).is_symlink() or Path(sources[key]).resolve() != path
                       for key, path in expected.items())):
            raise ValueError('codeact_fault_source_binding')
        self.evaluated = self._closed = False
        self.worker = CodeActOwnedWorker(data_dir, image, deps_dir, sources=sources,
            source_sha256=source_sha256, image_source_sha256=image_source_sha256,
            on_container_intent=on_container_intent)
        try:
            if self.worker.call('capabilities') != capabilities():
                raise ValueError('codeact_world_capability_binding')
            description = self.worker.call('describe', task_id=task_id)
            if (type(description) is not dict or set(description) != {'instruction', 'supervisor'}
                    or any(type(v) is not str for v in description.values())):
                raise ValueError('codeact_world_description')
            self.description = dict(description)  # Private controller material, never public logs.
            result = self.worker.call('start', task_id=task_id, max_steps=max_steps, environment_seed=100)
            self._seed = _verified_seed_provenance(result, task_id, 100)
        except BaseException:
            self.worker.close()
            raise

    @property
    def environment_seed_provenance(self):
        return dict(self._seed)

    def _check(self):
        if self.evaluated or self._closed:
            raise RuntimeError('codeact_world_terminal')

    def execute(self, code):
        self._check()
        result = self.worker.call('execute', code=code)
        actor_observation(result)
        return result

    def completed(self):
        self._check()
        value = self.worker.call('completed')
        if type(value) is not bool:
            raise ValueError('codeact_world_completion')
        return value

    def api_count(self):
        self._check()
        value = self.worker.call('api_count')
        if type(value) is not int or value < 0 or value > 500:
            raise ValueError('codeact_world_api_count')
        return value

    def fault_status(self):
        # Trusted parent may query after terminal scoring but before cleanup.
        if self._closed:
            raise RuntimeError('codeact_world_terminal')
        value = self.worker.call('fault_status')
        if (type(value) is not dict or set(value) != {'injected_refusals', 'injected_at_request_attempt', 'injected_error_observed', 'policy_sha256', 'backend_dispatched_for_fault'}
                or type(value['injected_refusals']) is not int or value['injected_refusals'] not in (0, 1)
                or type(value['injected_error_observed']) is not bool
                or (value['injected_refusals'] == 0 and value['injected_error_observed'] is not False)
                or not ((value['injected_refusals'] == 0 and value['injected_at_request_attempt'] is None)
                        or (value['injected_refusals'] == 1 and type(value['injected_at_request_attempt']) is int
                            and 1 <= value['injected_at_request_attempt'] <= 500))
                or value['policy_sha256'] != POLICY_SHA256 or value['backend_dispatched_for_fault'] is not False):
            raise ValueError('codeact_fault_status_invalid')
        return value

    def evaluate(self):
        self._check()
        self.evaluated = True
        value = self.worker.call('evaluate')
        if type(value) is not bool:
            raise ValueError('codeact_world_official_boolean')
        return value

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                if not self.worker._closed and self.worker.proc.poll() is None:
                    if self.worker.call('close') is not True:
                        raise ValueError('codeact_world_close')
            finally:
                self.worker.close()
