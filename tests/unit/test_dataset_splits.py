"""Tests for deterministic group-aware split assignment."""

from __future__ import annotations

from collections import Counter, defaultdict

from tests.helpers import build_contract, make_record

from eos.dataset.splits import assign_splits, split_summary


def build_rows() -> list:
    """Create 200 records with ten two-image duplicate groups."""
    rows = []
    for class_id, class_name in enumerate(("Alpha", "Beta")):
        for index in range(100):
            group_id = f"g{class_id}-{index // 2}" if index < 10 else f"{class_name}-{index}"
            rows.append(
                make_record(
                    f"{class_name}/{index}",
                    class_id,
                    class_name,
                    group_id=group_id,
                )
            )
    return rows


def test_split_is_deterministic_and_balanced() -> None:
    """The same seed must produce the same exact per-class quotas."""
    contract = build_contract(("Alpha", "Beta"), 200)
    rows = build_rows()
    first = assign_splits(rows, contract)
    second = assign_splits(rows, contract)
    assert [record.split for record in first] == [record.split for record in second]
    counts = Counter((record.class_name, record.split) for record in first)
    assert counts[("Alpha", "train")] == 70
    assert counts[("Alpha", "val")] == 15
    assert counts[("Alpha", "test")] == 15
    assert counts[("Beta", "train")] == 70
    assert counts[("Beta", "val")] == 15
    assert counts[("Beta", "test")] == 15


def test_duplicate_groups_never_cross_splits() -> None:
    """Every duplicate group must stay inside one split."""
    contract = build_contract(("Alpha", "Beta"), 200)
    rows = assign_splits(build_rows(), contract)
    groups: dict[str, set[str]] = defaultdict(set)
    for record in rows:
        groups[str(record.group_id)].add(str(record.split))
    assert all(len(splits) == 1 for splits in groups.values())
    summary = split_summary(rows)
    assert sum(summary["train"].values()) == 140
    assert sum(summary["val"].values()) == 30
    assert sum(summary["test"].values()) == 30
