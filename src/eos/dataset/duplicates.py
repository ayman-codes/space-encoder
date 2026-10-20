"""Duplicate and near-duplicate detection for dataset curation.

Byte-identical files are detected with SHA-256. Perceptual similarity uses a
64-bit DCT pHash and a bounded BK-tree search so a 27,000-image dataset does not
require an all-pairs comparison in Python.
"""

from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from eos.dataset.manifest import ImageRecord

READ_CHUNK_BYTES = 1 << 20
PHASH_SIZE = 32
PHASH_LOW_FREQ = 8


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    """Build an orthonormal DCT-II basis matrix.

    Args:
        size (int): Transform dimension.

    Returns:
        np.ndarray: Matrix of shape ``(size, size)``.
    """
    indices = np.arange(size, dtype=np.float64)
    frequencies = indices[:, None]
    samples = indices[None, :]
    angles = np.pi * (2.0 * samples + 1.0) * frequencies / (2.0 * size)
    basis = np.cos(angles)
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    result: np.ndarray = basis * np.sqrt(2.0 / size)
    return result


_DCT = _dct_matrix(PHASH_SIZE)


def perceptual_hash(image: Image.Image) -> int:
    """Compute a 64-bit DCT perceptual hash.

    Args:
        image (PIL.Image.Image): Source image.

    Returns:
        int: Unsigned 64-bit perceptual hash.
    """
    grayscale = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    coefficients = _DCT @ pixels @ _DCT.T
    low_frequency = coefficients[:PHASH_LOW_FREQ, :PHASH_LOW_FREQ]
    threshold = float(np.median(low_frequency[1:, :]))
    bits = (low_frequency > threshold).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def perceptual_hash_file(path: Path) -> int:
    """Compute the perceptual hash of an encoded image file.

    Args:
        path (Path): Image file path.

    Returns:
        int: Unsigned 64-bit perceptual hash.

    Raises:
        OSError: If Pillow cannot open or decode the image.
    """
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        image.load()
        return perceptual_hash(image)


def hamming_distance(left: int, right: int) -> int:
    """Compute the Hamming distance between two 64-bit hashes.

    Args:
        left (int): First hash.
        right (int): Second hash.

    Returns:
        int: Number of differing bits.
    """
    return int((left ^ right).bit_count())


class _BKNode:
    """Node in a Burkhard-Keller tree keyed by Hamming distance."""

    def __init__(self, value: int) -> None:
        """Create a node for one perceptual hash.

        Args:
            value (int): Hash stored at the node.

        Returns:
            None: The node is initialized in place.
        """
        self.value = value
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    """BK-tree supporting bounded Hamming-distance neighbour queries."""

    def __init__(self) -> None:
        """Create an empty BK-tree.

        Returns:
            None: The tree is initialized in place.
        """
        self.root: _BKNode | None = None

    def insert(self, value: int) -> None:
        """Insert a hash into the tree.

        Args:
            value (int): Hash to insert.

        Returns:
            None: The tree is mutated in place.
        """
        if self.root is None:
            self.root = _BKNode(value)
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value)
                return
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        """Find hashes within a Hamming radius.

        Args:
            value (int): Query hash.
            radius (int): Maximum inclusive Hamming distance.

        Returns:
            list[int]: Neighbour hashes, excluding the query itself.
        """
        if self.root is None:
            return []
        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if 0 < distance <= radius:
                matches.append(node.value)
            lower = max(0, distance - radius)
            upper = distance + radius
            for edge, child in node.children.items():
                if lower <= edge <= upper:
                    stack.append(child)
        return matches


class _UnionFind:
    """Disjoint-set structure used to form duplicate components."""

    def __init__(self) -> None:
        """Create an empty disjoint-set structure.

        Returns:
            None: The structure is initialized in place.
        """
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        """Add an item as a singleton set.

        Args:
            item (str): Item identifier.

        Returns:
            None: The structure is mutated in place.
        """
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        """Find the representative of an item.

        Args:
            item (str): Item identifier.

        Returns:
            str: Representative identifier.
        """
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        """Merge two sets by rank.

        Args:
            left (str): First item.
            right (str): Second item.

        Returns:
            None: The structure is mutated in place.
        """
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class DuplicateGroup(BaseModel):
    """A connected component of duplicate or near-duplicate images.

    Args:
        group_id (str): Deterministic group identifier.
        image_ids (tuple[str, ...]): Members sorted by identifier.
        class_names (tuple[str, ...]): Distinct class names in the group.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str
    image_ids: tuple[str, ...]
    class_names: tuple[str, ...]


class DuplicateReport(BaseModel):
    """Summary evidence produced by duplicate analysis.

    Args:
        exact_duplicate_groups (int): Number of SHA-256 collision groups.
        exact_duplicate_extra_files (int): Files beyond one per exact group.
        near_duplicate_groups (int): Number of perceptual duplicate groups.
        near_duplicate_extra_files (int): Files beyond one per perceptual group.
        cross_class_groups (int): Perceptual groups spanning multiple labels.
        groups (tuple[DuplicateGroup, ...]): Groups with more than one member.
    """

    model_config = ConfigDict(frozen=True)

    exact_duplicate_groups: int
    exact_duplicate_extra_files: int
    near_duplicate_groups: int
    near_duplicate_extra_files: int
    cross_class_groups: int
    groups: tuple[DuplicateGroup, ...]


def assign_duplicate_groups(  # noqa: PLR0912
    records: list[ImageRecord],
    phash_distance: int,
) -> tuple[list[ImageRecord], DuplicateReport]:
    """Assign deterministic duplicate group identifiers to records.

    Args:
        records (list[ImageRecord]): Records with SHA-256 and pHash values.
        phash_distance (int): Maximum Hamming distance for near-duplicates.

    Returns:
        tuple[list[ImageRecord], DuplicateReport]: Updated records and evidence.
    """
    union = _UnionFind()
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_phash: dict[int, list[str]] = defaultdict(list)
    for record in records:
        union.add(record.image_id)
        by_sha[record.sha256].append(record.image_id)
        by_phash[record.phash].append(record.image_id)

    for members in by_sha.values():
        for member in members[1:]:
            union.union(members[0], member)

    for members in by_phash.values():
        for member in members[1:]:
            union.union(members[0], member)
    tree = _BKTree()
    for value in by_phash:
        tree.insert(value)
    for value, members in by_phash.items():
        for neighbour in tree.query(value, phash_distance):
            union.union(members[0], by_phash[neighbour][0])

    components: dict[str, list[str]] = defaultdict(list)
    for record in records:
        components[union.find(record.image_id)].append(record.image_id)
    ordered_components = sorted(
        (sorted(members) for members in components.values()),
        key=lambda members: members[0],
    )

    group_ids: dict[str, str] = {}
    for index, members in enumerate(ordered_components, start=1):
        group_id = f"g{index:06d}"
        for member in members:
            group_ids[member] = group_id

    updated = [
        record.model_copy(update={"group_id": group_ids[record.image_id]}) for record in records
    ]
    groups: list[DuplicateGroup] = []
    exact_groups = sum(1 for members in by_sha.values() if len(members) > 1)
    exact_extra = sum(len(members) - 1 for members in by_sha.values() if len(members) > 1)
    near_groups = 0
    near_extra = 0
    cross_class = 0
    for members in ordered_components:
        if len(members) < 2:
            continue
        member_ids = set(members)
        group_records = [record for record in updated if record.image_id in member_ids]
        classes = tuple(sorted({record.class_name for record in group_records}))
        group_id = group_ids[members[0]]
        groups.append(
            DuplicateGroup(
                group_id=group_id,
                image_ids=tuple(members),
                class_names=classes,
            )
        )
        near_groups += 1
        near_extra += len(group_records) - 1
        if len(classes) > 1:
            cross_class += 1

    report = DuplicateReport(
        exact_duplicate_groups=exact_groups,
        exact_duplicate_extra_files=exact_extra,
        near_duplicate_groups=near_groups,
        near_duplicate_extra_files=near_extra,
        cross_class_groups=cross_class,
        groups=tuple(groups),
    )
    return updated, report
