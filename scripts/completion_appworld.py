"""Owned container adapter for the separately pinned completion-check worker."""

from scripts.completion_readonly_registry import capabilities
from scripts.owned_appworld_v3 import OwnedAppWorldEnvironmentV3


class CompletionAppWorldEnvironment(OwnedAppWorldEnvironmentV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            if self.worker.call('capabilities') != capabilities():
                raise ValueError('completion_worker_capability_mismatch')
        except BaseException:
            self.close()
            raise

    def inspect(self, text):
        return self.worker.call('inspect', text=text)

    def api_count(self):
        value = self.worker.call('api_count')
        if type(value) is not int or value < 0:
            raise ValueError('invalid_api_counter')
        return value
