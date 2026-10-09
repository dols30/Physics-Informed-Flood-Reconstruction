"""Download published files, verify checksums, and retain provenance."""

import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile

RECORD_ID = "17777387"
EXPERIMENTS = ("C2", "C4", "O2", "O4", "P1", "P2")


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "physics-informed-floods/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def verify(path: Path, checksum: str) -> None:
    algorithm, expected = checksum.split(":", 1)
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {path}; remove it and download again.")


def extract_archive(archive: Path, destination: Path) -> None:
    """Reject path traversal and symlinks before extracting an external ZIP."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def download(experiment: str, root: Path) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Experiment must be one of {EXPERIMENTS}")
    destination = root / "raw" / f"zaragoza-{RECORD_ID}"
    metadata_path = destination / "record.json"
    if not metadata_path.exists():
        fetch(f"https://zenodo.org/api/records/{RECORD_ID}", metadata_path)
    record = json.loads(metadata_path.read_text())
    for name in ("README.md", "requirements.txt", experiment + ".zip"):
        published = next(item for item in record["files"] if item["key"] == name)
        target = destination / name
        if not target.exists():
            print(f"Downloading {name} ({published['size'] / 1e6:.1f} MB)", flush=True)
            fetch(published["links"]["self"], target)
        verify(target, published["checksum"])
        print(f"Verified {name}", flush=True)
    extract_archive(destination / (experiment + ".zip"), destination / "extracted")
    return destination
