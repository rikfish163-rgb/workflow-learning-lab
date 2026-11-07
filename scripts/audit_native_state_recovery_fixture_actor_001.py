"""Read the closed scripted probe without opening a model, worker or scorer."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import native_state_recovery_fixture_episode_001 as fixture
from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts import prepare_native_source_contexts as preflight
from scripts import probe_native_state_recovery_fixture_actor_001 as probe
from scripts import run_native_state_recovery_fixture_001 as run
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode
from scripts.native_generation_accounting_004 import generation_accounting
from scripts.native_source_registration import _public_hash
from scripts.run_native_source_pilot import actor_evidence

PROBE_SOURCE = "scripts/probe_native_state_recovery_fixture_actor_001.py"
PROBE_SHA256 = "e4e31b8393cd3368c19b8f0084c835aba2d75715873c71f5641f934271afa802"
PROOF = "artifacts/native-state-recovery-fixture-actor-audit-CPU-001.json"


def main():
    preflight._host_preflight()
    os.umask(0o077)
    assert _public_hash(WORKSPACE, PROBE_SOURCE) == PROBE_SHA256
    rows = run._design(WORKSPACE)["rows"]
    base, queue, _, _ = run.bindings()
    root = Path(run.PRIVATE_STATE_ROOT) / probe.ID
    reg = SimpleNamespace(**{**vars(base), "PRIVATE_RUN_ROOT": str(root)})
    episode = run.natural.prior._load("native_state_recovery_episode_001", (
        ("native-state-recovery-001-", run.EXPERIMENT, 2),
        ('receipt.get("is_policy_episode") is True', 'receipt.get("is_policy_episode") is False', 1),
        ('intent.get("is_policy_episode") is True', 'intent.get("is_policy_episode") is False', 1),
    ), ID=run.ID, PLAN=run.PLAN, TASK_IDS=tuple(row["template_task_id"] for row in rows))

    def scripted_evidence(receipt, journal, initial):
        assert receipt["is_policy_episode"] is False
        counts = generation_accounting(journal)
        cost = receipt["actor_outcome"]["cost"]
        assert counts["unreturned_generation_intents_upper_bound"] == 0
        assert counts["generation_attempts_after_cancellations"] == cost["generations"]
        for key in ("prompt_tokens", "completion_tokens"):
            assert counts[f"journal_returned_{key}_lower_bound"] == cost[key]
        return actor_evidence(receipt, journal, expected_context=initial)

    episode._actor_evidence = scripted_evidence
    fixture.install(episode, rows)
    bound = episode.bind(reg, queue)
    path, = root.glob("unit-*")
    raw = bound.core._read_private(path / "execution-receipt.json", maximum=1_000_000)
    receipt = _decode(raw)
    checked = fixture.collect(episode, bound.core, path, receipt, fixture._case_context(rows[0], protocol.CASES[0]))
    assert checked["fixture"]["fixture_target_verified"] is True and run._main_pid(receipt["snapshot"]["unit_name"]) == 0
    assert receipt["official_scores_computed"] == receipt["new_GPU_jobs_started_by_controller"] == 0
    proof = {"id": Path(PROOF).stem, "status": "passed", "scope": "audited_scripted_cpu_policy_real_AppWorld_tools",
             "model_calls": 0, "GPU_jobs": 0, "official_scorer_calls": 0, "actor_policy_episode": False,
             "scripted_actions": receipt["actor_outcome"]["cost"]["action_attempts"],
             "fixture_target_verified": True, "worker_release_verified": True,
             "metadata_substitution_verified": True, "execution_receipt_sha256": hashlib.sha256(raw).hexdigest(),
             "probe_source_sha256": PROBE_SHA256, "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (WORKSPACE / PROOF).open("xb") as file:
        file.write(_encode(proof) + b"\n")
        file.flush()
        os.fsync(file.fileno())
    print("closed scripted fixture reconciled without starting a second trial")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("native_state_recovery_fixture_actor_audit_failed") from None
