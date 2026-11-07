"""Run the frozen worker entry point with an isolated V3 registry binding."""

import importlib.util
from pathlib import Path

from scripts import verification_readonly_registry_v3 as registry
from scripts.verification_worker_state_v3 import VerificationWorkerStateV3, capabilities


def make_worker():
    source = Path(__file__).with_name('completion_appworld_worker.py')
    spec = importlib.util.spec_from_file_location('scripts._verification_appworld_worker_v3_core', source)
    if spec is None or spec.loader is None:
        raise ImportError('frozen AppWorld worker unavailable')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.ALLOWED_APIS = registry.ALLOWED_APIS
    worker.capabilities = capabilities
    worker.validate_inspection_arguments = registry.validate_inspection_arguments
    worker.validate_inspection_response = registry.validate_inspection_response
    worker.verify_installed_sources = registry.verify_installed_sources
    worker.CompletionWorkerState = VerificationWorkerStateV3
    return worker


def main():
    make_worker().main()


if __name__ == '__main__':
    main()
