"""Deterministic stratified split assignment with near-duplicate grouping."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Literal

from eos.dataset.contract import DatasetContract
from eos.dataset.manifest import SPLITS, ImageRecord
from eos.dataset.validate import split_targets


def _group_records(
    records: list[ImageRecord],
    group_aware: bool,
) -> list[list[ImageRecord]]:
    """Group records by duplicate component or by identity.

    Args:
        records (list[ImageRecord]): Manifest rows.
        group_aware (bool): Whether to honour perceptual duplicate groups.

    Returns:
        list[list[ImageRecord]]: Groups of records.
    """
    if not group_aware:
        return [[record] for record in records]
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        key = record.group_id or record.image_id
        grouped[key].append(record)
    return [grouped[key] for key in sorted(grouped)]


def _choose_split(
    group_counts: dict[str, int],
    assigned: dict[str, dict[str, int]],
    targets: dict[str, dict[str, int]],
) -> Literal["train", "val", "test"]:
    """Choose the split with the largest normalized target deficit.

    Args:
        group_counts (dict[str, int]): Class counts in the group.
        assigned (dict[str, dict[str, int]]): Current counts per split/class.
        targets (dict[str, dict[str, int]]): Target counts per split/class.

    Returns:
        Literal["train", "val", "test"]: Selected split name.
    """
    best_split: Literal["train", "val", "test"] = SPLITS[0]
    best_score: float | None = None
    for split in SPLITS:
        score = 0.0
        for class_name, count in group_counts.items():
            target = max(targets[split][class_name], 1)
            deficit = targets[split][class_name] - assigned[split][class_name] - count
            score += deficit / target
        if best_score is None or score > best_score:
            best_score = score
            best_split = split
    return best_split


def assign_splits(records: list[ImageRecord], contract: DatasetContract) -> list[ImageRecord]:
    """Assign a deterministic group-aware stratified split.

    Multi-image duplicate groups are allocated first so they can never cross a
    split boundary. Singleton records then fill the remaining per-class quotas.

    Args:
        records (list[ImageRecord]): Manifest rows before split assignment.
        contract (DatasetContract): Dataset contract.

    Returns:
        list[ImageRecord]: Records with populated ``split`` fields.
    """
    groups = _group_records(records, contract.split.group_aware)
    targets: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    class_totals: dict[str, int] = defaultdict(int)
    for record in records:
        class_totals[record.class_name] += 1
    for class_name, count in class_totals.items():
        counts = split_targets(count, contract)
        for split in SPLITS:
            targets[split][class_name] = counts[split]

    assigned: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in class_totals} for split in SPLITS
    }
    rng = random.Random(contract.split.seed)
    multi = [group for group in groups if len(group) > 1]
    singles = [group[0] for group in groups if len(group) == 1]
    rng.shuffle(multi)
    multi.sort(key=len, reverse=True)

    assignments: dict[str, str] = {}
    for group in multi:
        group_counts: dict[str, int] = defaultdict(int)
        for record in group:
            group_counts[record.class_name] += 1
        split = _choose_split(group_counts, assigned, targets)
        for record in group:
            assignments[record.image_id] = split
            assigned[split][record.class_name] += 1

    singles_by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in singles:
        singles_by_class[record.class_name].append(record)
    for class_name, class_singles in singles_by_class.items():
        class_index = contract.source.class_names.index(class_name)
        class_rng = random.Random(contract.split.seed + class_index)
        class_rng.shuffle(class_singles)
        for record in class_singles:
            remaining = {
                split: targets[split][class_name] - assigned[split][class_name] for split in SPLITS
            }
            split = max(
                SPLITS,
                key=lambda candidate: (remaining[candidate], -SPLITS.index(candidate)),
            )
            if remaining[split] <= 0:
                split = min(
                    SPLITS,
                    key=lambda candidate: (
                        assigned[candidate][class_name] - targets[candidate][class_name],
                        SPLITS.index(candidate),
                    ),
                )
            assignments[record.image_id] = split
            assigned[split][class_name] += 1

    return [record.model_copy(update={"split": assignments[record.image_id]}) for record in records]


def split_summary(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    """Summarize actual counts per split and class.

    Args:
        records (list[ImageRecord]): Records with assigned splits.

    Returns:
        dict[str, dict[str, int]]: Nested counts keyed by split and class.
    """
    summary: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for record in records:
        if record.split is None:
            continue
        summary[record.split][record.class_name] = (
            summary[record.split].get(record.class_name, 0) + 1
        )
    return {split: dict(sorted(values.items())) for split, values in summary.items()}
