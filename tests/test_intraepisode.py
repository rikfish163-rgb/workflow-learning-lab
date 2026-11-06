from test_runner import TASK, World

from wll.intraepisode import probe_intraepisode_errors
from wll.protocol import Episode, Step


class ErrorWorld(World):
    def execute(self, code):
        return "Execution failed: example" if code == "error" else super().execute(code)


def test_successful_rollout_error_prefix_is_reverified_and_not_pass_one(tmp_path):
    source = Episode(TASK, "test-double", 17,
        [Step("inc", "1"), Step("error", "Execution failed: example"), Step("inc", "2")],
        True, "completed")
    result = probe_intraepisode_errors([source], lambda task: ErrorWorld(),
                                      output=tmp_path / "probes")
    assert len(result) == 1
    assert not result[0].success
    assert result[0].branch == "prefix_probe"
    assert result[0].replayed_steps == 2
    assert result[0].generated_tokens == 0
    assert result[0].metadata["source_rollout_success"] is True
    assert result[0].metadata["parent_episode_hash"]
