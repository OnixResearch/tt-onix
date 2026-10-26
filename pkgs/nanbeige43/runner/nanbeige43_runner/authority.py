# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.checkpoint_authority]
"""Checkpoint admission against the pinned Nanbeige4.3-33B authority.

Every weight-loading command calls `verify_checkpoint` before it opens a device.
A full checkpoint must match the authority file-for-file (SHA-256 and size). A
reduced-layer checkpoint must carry a manifest written by `reduce`. That
manifest names the authority source files its tensors came from, and its output
files must match their recorded SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REDUCED_MANIFEST = "nanbeige43-reduced.json"
REDUCED_SCHEMA = "nanbeige43-reduced-checkpoint/v1"
AUTHORITY_SCHEMA = "nanbeige43-checkpoint-authority/v1"
# Files a Hugging Face download or this package may place beside the checkpoint
# that carry no model semantics.
IGNORED_NAMES = frozenset({"README.md", ".gitattributes", REDUCED_MANIFEST})
_CHUNK = 16 * 1024 * 1024


class CheckpointRejected(SystemExit):
    """Raised (as a process exit) when a checkpoint fails admission."""

    def __init__(self, message: str):
        super().__init__(f"checkpoint rejected: {message}")


def load_authority(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ["NANBEIGE43_AUTHORITY"])
    authority = json.loads(path.read_text())
    if authority.get("schema") != AUTHORITY_SCHEMA:
        raise CheckpointRejected(
            f"authority schema {authority.get('schema')!r} is not {AUTHORITY_SCHEMA}"
        )
    return authority


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_all(paths: list[Path]) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
        return dict(zip((p.name for p in paths), pool.map(sha256_file, paths)))


def _semantic_files(directory: Path) -> set[str]:
    names = set()
    for entry in directory.iterdir():
        if entry.name.startswith(".") or entry.name in IGNORED_NAMES or entry.is_dir():
            continue
        if entry.name.startswith("VERIFIED-") and entry.suffix == ".json":
            continue
        names.add(entry.name)
    return names


def identity_digest(files: dict[str, str]) -> str:
    lines = "".join(f"{name}\0{files[name]}\n" for name in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def _verify_full(directory: Path, authority: dict) -> dict:
    expected = {entry["name"]: entry for entry in authority["files"]}
    present = _semantic_files(directory)
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    if missing:
        raise CheckpointRejected(f"missing {missing[0]}")
    if extra:
        raise CheckpointRejected(f"unexpected file {extra[0]}")
    for name, entry in expected.items():
        size = (directory / name).stat().st_size
        if size != entry["size"]:
            raise CheckpointRejected(f"{name} size {size} != {entry['size']}")
    hashes = _hash_all([directory / name for name in sorted(expected)])
    for name, entry in expected.items():
        if hashes[name] != entry["sha256"]:
            raise CheckpointRejected(
                f"{name} sha256 {hashes[name]} != {entry['sha256']}"
            )
    return {
        "kind": "full",
        "path": str(directory),
        "revision": authority["revision"],
        "layers": authority["architecture"]["num_hidden_layers"],
        "files_sha256_digest": identity_digest(hashes),
    }


def _verify_reduced(directory: Path, authority: dict) -> dict:
    manifest_path = directory / REDUCED_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != REDUCED_SCHEMA:
        raise CheckpointRejected(
            f"{REDUCED_MANIFEST} schema {manifest.get('schema')!r}"
        )
    if manifest.get("source_revision") != authority["revision"]:
        raise CheckpointRejected(
            f"reduced source revision {manifest.get('source_revision')!r} != {authority['revision']}"
        )
    authority_hashes = {entry["name"]: entry["sha256"] for entry in authority["files"]}
    for name, digest in manifest["source_files"].items():
        if authority_hashes.get(name) != digest:
            raise CheckpointRejected(
                f"reduced source file {name} is not the authority's {name}"
            )
    expected = manifest["files"]
    present = _semantic_files(directory)
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    if missing:
        raise CheckpointRejected(f"missing {missing[0]}")
    if extra:
        raise CheckpointRejected(f"unexpected file {extra[0]}")
    hashes = _hash_all([directory / name for name in sorted(expected)])
    for name, digest in expected.items():
        if hashes[name] != digest:
            raise CheckpointRejected(f"{name} sha256 {hashes[name]} != {digest}")
    return {
        "kind": "reduced",
        "path": str(directory),
        "revision": authority["revision"],
        "layers": manifest["layers"],
        "manifest_sha256": sha256_file(manifest_path),
        "files_sha256_digest": identity_digest(hashes),
    }


def verify_checkpoint(
    directory: str | os.PathLike,
    authority: dict,
    *,
    allow_full: bool = True,
    allow_reduced: bool = True,
) -> dict:
    directory = Path(directory)
    if not directory.is_dir():
        raise CheckpointRejected(f"{directory} is not a directory")
    if (directory / REDUCED_MANIFEST).exists():
        if not allow_reduced:
            raise CheckpointRejected(
                "a reduced-layer checkpoint is not admitted for this command"
            )
        return _verify_reduced(directory, authority)
    if not allow_full:
        raise CheckpointRejected("this command admits only a reduced-layer checkpoint")
    return _verify_full(directory, authority)
