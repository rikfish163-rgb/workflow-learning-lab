"""Read-only terminal audit for the frozen authored-workflow writer."""

import argparse
import json
from dataclasses import replace
from types import SimpleNamespace

from scripts import native_state_recovery_fixture_review_001 as fixture
from scripts import native_state_recovery_review_001 as review


def report(plan_sha256, result_sha256):
    writer = fixture.LINEAGE.writer

    def core_for(runner):
        core = writer.core_for(runner)
        # The persisted study journal belongs to the coordinator, while its
        # episode reader supplies the separate private-file boundary.
        core.episode._journal = core._journal
        return core

    adapter = SimpleNamespace(**{**vars(writer), "core_for": core_for})
    lineage = replace(fixture.LINEAGE, writer=adapter)
    return review.report(plan_sha256, result_sha256, lineage=lineage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.plan_sha256, args.result_sha256), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("native_state_recovery_fixture_writer_report_invalid") from None
