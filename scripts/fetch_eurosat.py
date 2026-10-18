"""Download, verify, and extract the EuroSAT RGB dataset.

The original DFKI mirror is no longer reachable, so the Zenodo mirror is used.
The archive is verified against a pinned SHA-256 before extraction, then the
extracted tree is validated for class count, image count, and patch geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ARCHIVE_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
ARCHIVE_NAME = "EuroSAT_RGB.zip"
ARCHIVE_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
EXPECTED_IMAGES = 27_000
EXPECTED_CLASSES = 10
EXPECTED_SIZE = 64
CHUNK_BYTES = 1 << 20


def download(url: str, destination: Path) -> Path:
    """Stream a remote archive to disk.

    Args:
        url (str): Absolute download URL.
        destination (Path): Target file path.

    Returns:
        Path: The written file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"archive already present: {destination}")
        return destination
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(CHUNK_BYTES):
            handle.write(chunk)
    partial.replace(destination)
    return destination


def sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, destination: Path) -> None:
    """Extract a zip archive, replacing any previous extraction.

    Args:
        archive (Path): Zip archive to extract.
        destination (Path): Directory that receives the archive contents.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)


def resolve_root(destination: Path) -> Path:
    """Locate the directory that directly contains the class folders.

    Args:
        destination (Path): Extraction directory.

    Returns:
        Path: Directory holding the ten class folders.

    Raises:
        RuntimeError: If no directory with ten subdirectories is found.
    """
    candidates = [destination, *[path for path in destination.iterdir() if path.is_dir()]]
    for candidate in candidates:
        folders = [path for path in candidate.iterdir() if path.is_dir()]
        if len(folders) == EXPECTED_CLASSES:
            return candidate
    raise RuntimeError(f"no directory with {EXPECTED_CLASSES} class folders under {destination}")


def validate(root: Path) -> dict[str, int]:
    """Validate class count and image count of the extracted dataset.

    Args:
        root (Path): Directory holding the class folders.

    Returns:
        dict[str, int]: Per-class image counts.

    Raises:
        RuntimeError: If the class count or image count is wrong.
    """
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(f"expected {EXPECTED_CLASSES} classes, found {len(classes)}")
    counts = {path.name: len(list(path.glob("*.jpg"))) for path in classes}
    total = sum(counts.values())
    if total != EXPECTED_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_IMAGES} images, found {total}")
    return counts


def check_geometry(root: Path, image_size: int) -> None:
    """Verify that extracted patches have the expected square geometry.

    Args:
        root (Path): Directory holding the class folders.
        image_size (int): Expected square edge length in pixels.

    Raises:
        RuntimeError: If Pillow is unavailable or a patch has the wrong shape.
    """
    sample = next(iter(sorted(root.glob("*/*.jpg"))))
    with Image.open(sample) as image:
        width, height = image.size
    if (width, height) != (image_size, image_size):
        raise RuntimeError(f"expected {image_size}x{image_size} patches, found {width}x{height}")


def main() -> int:
    """Run the dataset acquisition pipeline.

    Returns:
        int: Process exit code, zero on success.
    """
    parser = argparse.ArgumentParser(description="Download and verify EuroSAT RGB")
    parser.add_argument("--destination", type=Path, default=Path("data/raw/eurosat"))
    parser.add_argument("--expected-sha256", default=ARCHIVE_SHA256)
    parser.add_argument("--image-size", type=int, default=EXPECTED_SIZE)
    arguments = parser.parse_args()

    archive = arguments.destination.parent / ARCHIVE_NAME
    download(ARCHIVE_URL, archive)
    digest = sha256(archive)
    print(f"sha256 {digest}")
    if arguments.expected_sha256 and digest != arguments.expected_sha256:
        print("checksum mismatch", file=sys.stderr)
        return 1
    extract(archive, arguments.destination)
    root = resolve_root(arguments.destination)
    counts = validate(root)
    check_geometry(root, arguments.image_size)
    print(f"dataset root {root}")
    for name, count in counts.items():
        print(f"  {name:<24} {count}")
    print(f"total {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
