# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.checkpoint_authority]
"""Derive a reduced-layer checkpoint from the admitted full checkpoint.

The output keeps every tensor at production shape: the full embedding, the
final norm, the untied LM head, and decoder layers `0..layers-1`. Its
`config.json` differs only in `num_hidden_layers`. A manifest binds the output
to the authority source files it was read from.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from .authority import (
    REDUCED_MANIFEST,
    REDUCED_SCHEMA,
    load_authority,
    sha256_file,
    verify_checkpoint,
)

COPIED_METADATA = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
GLOBAL_TENSORS = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")


def selected(name: str, layers: int) -> bool:
    if name in GLOBAL_TENSORS:
        return True
    prefix = "model.layers."
    if not name.startswith(prefix):
        return False
    return int(name[len(prefix) :].split(".", 1)[0]) < layers


def reduce_checkpoint(source: Path, output: Path, layers: int) -> dict:
    authority = load_authority()
    full_layers = authority["architecture"]["num_hidden_layers"]
    if not 1 <= layers < full_layers:
        raise SystemExit(f"--layers must be in [1, {full_layers - 1}]")
    identity = verify_checkpoint(source, authority, allow_reduced=False)
    authority_hashes = {entry["name"]: entry["sha256"] for entry in authority["files"]}

    index = json.loads((source / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if selected(name, layers):
            by_shard.setdefault(shard, []).append(name)
    expected = len(GLOBAL_TENSORS) + layers * 9
    found = sum(len(names) for names in by_shard.values())
    if found != expected:
        raise SystemExit(f"selected {found} tensors, expected {expected}")

    tensors = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)

    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    save_file(tensors, staging / "model.safetensors", metadata={"format": "pt"})
    del tensors
    config = json.loads((source / "config.json").read_text())
    config["num_hidden_layers"] = layers
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in COPIED_METADATA:
        shutil.copyfile(source / name, staging / name)

    files = {entry.name: sha256_file(entry) for entry in sorted(staging.iterdir())}
    source_files = {
        name: authority_hashes[name]
        for name in sorted(
            set(by_shard)
            | set(COPIED_METADATA)
            | {"config.json", "model.safetensors.index.json"}
        )
    }
    manifest = {
        "schema": REDUCED_SCHEMA,
        "source_revision": authority["revision"],
        "source_files_sha256_digest": identity["files_sha256_digest"],
        "layers": layers,
        "tensors": sorted(name for names in by_shard.values() for name in names),
        "source_files": source_files,
        "files": files,
    }
    (staging / REDUCED_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)
    return verify_checkpoint(output, authority, allow_full=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(reduce_checkpoint(args.source, args.output, args.layers), indent=2)
    )


if __name__ == "__main__":
    main()
