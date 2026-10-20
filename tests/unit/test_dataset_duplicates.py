"""Tests for perceptual and byte-level duplicate detection."""

from __future__ import annotations

import numpy as np
from PIL import Image
from tests.helpers import make_record

from eos.dataset.duplicates import (
    assign_duplicate_groups,
    hamming_distance,
    perceptual_hash,
)


def test_hamming_distance_counts_differing_bits() -> None:
    """Hamming distance must match the bit difference count."""
    assert hamming_distance(0b1010, 0b0011) == 2


def test_perceptual_hash_is_stable_for_identical_images() -> None:
    """The same pixels must produce the same hash."""
    array = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    image = Image.fromarray(array, mode="RGB")
    assert perceptual_hash(image) == perceptual_hash(image.copy())


def test_perceptual_hash_distinguishes_patterns() -> None:
    """Different spatial patterns should not collapse to the same hash."""
    left = np.zeros((64, 64, 3), dtype=np.uint8)
    right = np.full((64, 64, 3), 255, dtype=np.uint8)
    right[:, 32:, :] = 0
    assert perceptual_hash(Image.fromarray(left, mode="RGB")) != perceptual_hash(
        Image.fromarray(right, mode="RGB")
    )


def test_duplicate_groups_join_exact_and_perceptual_matches() -> None:
    """Records sharing a digest or hash must share a group identifier."""
    digest = "a" * 64
    rows = [
        make_record("Alpha/a", 0, "Alpha", sha256=digest, phash=123),
        make_record("Alpha/b", 0, "Alpha", sha256=digest, phash=123),
        make_record("Beta/c", 1, "Beta", sha256="b" * 64, phash=123),
        make_record("Beta/d", 1, "Beta", sha256="c" * 64, phash=999),
    ]
    updated, report = assign_duplicate_groups(rows, phash_distance=0)
    assert len({record.group_id for record in updated[:3]}) == 1
    assert updated[3].group_id != updated[0].group_id
    assert report.exact_duplicate_groups == 1
    assert report.cross_class_groups == 1
    assert report.near_duplicate_extra_files == 2
