import pytest

from scripts import native_state_recovery_fixture_review_001 as review
from scripts import run_native_state_recovery_fixture_utility_001 as utility
from scripts import run_native_state_recovery_fixture_writer_001 as writer
from scripts import run_native_state_recovery_utility_001 as common


def test_writer_reauthenticates_fixture_proof_before_using_facts(monkeypatch):
    case = {"same_resource_read_matches_intent": True}
    prepared = {"cases": [case, case], "private_provenance": ["first", "second"]}
    plan_sha, result_sha = "a" * 64, "b" * 64

    def prepare(source_plan, source_result):
        assert (source_plan, source_result) == (plan_sha, result_sha)
        return prepared

    monkeypatch.setattr(writer.audit, "prepare", prepare)
    proof = writer.qualify(plan_sha, result_sha)
    assert "private_provenance" not in proof
    assert writer._prepare({"qualification": proof}) is prepared
    with pytest.raises(ValueError):
        writer._prepare({"qualification": {**proof, "private_input_sha256": "c" * 64}})
    prepared["cases"].pop()
    with pytest.raises(ValueError):
        writer.qualify(plan_sha, result_sha)


def test_fixture_utility_rejects_overlap_and_wrong_review_lineage(monkeypatch):
    origins = [f"{i:07x}" for i in range(4)]
    scenes = [f"{i:07x}" for i in range(100, 109)]
    cohort = {"official_split": "train", "project_unseen": False, "all_train_roles_already_exposed": True,
              "utility_task_ids_reserved": [f"{scene}_{variant}" for scene in scenes for variant in (1, 2, 3)]}
    monkeypatch.setattr(common.roster, "load_registration", lambda *_args, **_kwargs: {"design": {"cohort": cohort}})
    monkeypatch.setattr(common, "_public_hash", lambda *_: "a" * 64)
    source = {"task_ids": [f"{origin}_3" for origin in origins]}
    source_scenarios, pairs, batches, orders = common._cohort(source, utility.LINEAGE)
    assert source_scenarios == origins
    assert len(pairs) == len(orders) == 27 and len(batches) == 14
    assert orders[0] == ["control", "experience"] and orders[-2] == ["experience", "control"]
    with pytest.raises(ValueError):
        common._cohort({"task_ids": [*source["task_ids"][:3], f"{scenes[0]}_3"]}, utility.LINEAGE)
    manifest = {"id": review.LINEAGE.manifest_id, "private": True,
                "source_gate_kind": review.SOURCE_GATE_KIND, "source_plan_sha256": "a" * 64,
                "source_result_sha256": "b" * 64, "writer_plan_sha256": "c" * 64,
                "writer_result_sha256": "d" * 64, "experience_bundle_id": review.BUNDLE_PREFIX + "e" * 64,
                "source_scenario_ids": origins}
    proof = common.qualification(manifest, "f" * 64, lineage=utility.LINEAGE)
    assert proof["source_scenario_ids"] == origins
    with pytest.raises(ValueError):
        common.qualification({**manifest, "source_gate_kind": common.review.SOURCE_GATE_KIND},
                             "f" * 64, lineage=utility.LINEAGE)
