import pytest

from wll.partition import partition_train_ids

IDS = [f"s{i}_{j}" for i in range(30) for j in range(3)]


def test_partitions_are_order_independent_and_keep_scenarios_whole():
    a = partition_train_ids(IDS, seed=17, origin_scenarios=12, utility_scenarios=9,
                            reserved_origin={"s0", "s1"})
    b = partition_train_ids(list(reversed(IDS)), seed=17, origin_scenarios=12,
                            utility_scenarios=9, reserved_origin={"s1", "s0"})
    assert a == b
    sets = [{t.rsplit("_", 1)[0] for t in a[k]} for k in ("origin", "utility", "distill")]
    assert [len(s) for s in sets] == [12, 9, 9]
    assert {"s0", "s1"} <= sets[0]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    assert sum(len(a[k]) for k in a) == len(IDS)


def test_invalid_partitions_fail_before_any_experiment():
    with pytest.raises(ValueError, match="duplicate"):
        partition_train_ids(IDS + IDS[:1])
    with pytest.raises(ValueError, match="reserved"):
        partition_train_ids(IDS, reserved_origin={"unknown"})
    with pytest.raises(ValueError, match="nonempty"):
        partition_train_ids(IDS, origin_scenarios=25, utility_scenarios=5)
