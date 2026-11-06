"""Seed-verified AppWorld startup using the frozen Docker/RPC transport.

This requires the separately built seeded-v2 image. Successful provenance proves
the worker's startup handshake, not a real-container acceptance test by itself.
"""
from .appworld import AppWorldEnvironment, AppWorldWorker

DEFAULT_IMAGE = "wll-appworld:seeded-v2"
SEED_PROTOCOL = "wll-appworld-seed-v1"


class AppWorldWorkerV2(AppWorldWorker):
    def __init__(self, data_dir, image=DEFAULT_IMAGE, deps_dir=None, timeout=120):
        self._closed = False
        super().__init__(data_dir, image, deps_dir, timeout)

    def close(self):
        # The frozen call() closes on timeout; constructor cleanup may follow.
        if not self._closed:
            super().close()
            self._closed = True


def _verified_seed_provenance(result, task_id, environment_seed):
    if (type(result) is not dict or
            set(result) != {"protocol", "started", "task_id", "environment_seed"} or
            result["protocol"] != SEED_PROTOCOL or result["started"] is not True or
            type(result["task_id"]) is not str or result["task_id"] != task_id):
        raise ValueError("invalid seeded AppWorld start response")
    seed = result["environment_seed"]
    if (type(seed) is not dict or set(seed) != {"requested", "effective", "verified"} or
            type(seed["requested"]) is not int or type(seed["effective"]) is not int or
            seed["requested"] != environment_seed or seed["effective"] != environment_seed or
            seed["verified"] is not True):
        raise ValueError("invalid seeded AppWorld start response")
    return dict(seed)


class AppWorldEnvironmentV2(AppWorldEnvironment):
    def __init__(self, task_id, data_dir, max_steps, image=DEFAULT_IMAGE, deps_dir=None,
                 environment_seed=100):
        if type(environment_seed) is not int or environment_seed != 100:
            raise ValueError("environment seed must be the registered integer 100")
        if type(task_id) is not str or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.worker = AppWorldWorkerV2(data_dir, image, deps_dir)
        self.evaluated = False
        try:
            result = self.worker.call("start", task_id=task_id, max_steps=max_steps,
                                      environment_seed=environment_seed)
            self._environment_seed_provenance = _verified_seed_provenance(
                result, task_id, environment_seed)
        except BaseException:
            self.worker.close()
            raise

    @property
    def environment_seed_provenance(self):
        """Controller evidence only; do not include it in policy messages."""
        return dict(self._environment_seed_provenance)
