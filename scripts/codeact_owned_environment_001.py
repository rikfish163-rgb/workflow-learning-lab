"""Seed/capability-verified host facade; all model Python stays in Docker."""
from scripts.codeact_owned_worker_001 import CodeActOwnedWorker
from scripts.codeact_within_worker_001 import actor_observation, capabilities
from wll.appworld_v2 import _verified_seed_provenance


class CodeActOwnedEnvironment:
    def __init__(self, task_id, data_dir, max_steps, image, deps_dir, *,
                 on_container_intent, sources, source_sha256, image_source_sha256):
        if (type(task_id) is not str or not task_id or type(max_steps) is not int
                or not 1 <= max_steps <= 40):
            raise ValueError('codeact_world_configuration')
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
