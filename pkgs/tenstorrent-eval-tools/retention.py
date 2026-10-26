"""Frozen historical corpus and strict full-vocabulary retention scoring."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
SCHEMA = "qwen38-six-passage-v1"
RECIPE = "targets-13-end:f32-to-f64-logsoftmax:equal-six-passage-means"
ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)
REQUIRED = {
    "model",
    "checkpoint",
    "runtime",
    "package",
    "cache",
    "precision_flags",
    "execution",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_manifest(model_directory):
    """Hash the index and every referenced shard, never directory/revision labels."""
    root = Path(model_directory).resolve(strict=True)
    index_path = root / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("checkpoint index requires a nonempty weight_map")
    if any(not isinstance(name, str) or not name for name in mapping.values()):
        raise ValueError("checkpoint shard names must be nonempty strings")
    shards = {}
    for name in sorted(set(mapping.values())):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "checkpoint index shard paths must stay inside the model directory"
            )
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                "checkpoint shard must be a file within the model directory"
            )
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("checkpoint shard changed while hashing: " + name)
        shards[name] = {"sha256": sha, "bytes": after.st_size}
    if index_path.read_bytes() != index_bytes:
        raise ValueError("checkpoint index changed while hashing")
    return {
        "format": "indexed-safetensors-sha256-v1",
        "index": {
            "name": index_path.name,
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
            "bytes": len(index_bytes),
        },
        "shards": shards,
    }


def weight_identity(manifest):
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return "weights-sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_weight_identity(prov, manifest):
    if manifest is None:
        if prov["execution"].startswith("hardware:"):
            raise ValueError(
                "hardware retention requires index and all-shard content identities"
            )
        return
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "indexed-safetensors-sha256-v1"
        or not isinstance(manifest.get("shards"), dict)
        or not manifest["shards"]
    ):
        raise ValueError("invalid checkpoint weight manifest")
    entries = [manifest.get("index", {})] + list(manifest["shards"].values())
    for entry in entries:
        sha = entry.get("sha256", "")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
        ):
            raise ValueError("invalid checkpoint content digest or size")
    if manifest["index"].get("name") != "model.safetensors.index.json" or prov[
        "model"
    ] != weight_identity(manifest):
        raise ValueError(
            "model provenance does not match checkpoint shard content identity"
        )


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def passages():
    return load(ROOT / "data/passages.json")


def freeze(model, output, checkpoint):
    import transformers
    from transformers import AutoTokenizer

    directory = Path(model).resolve(strict=True)
    assets = {
        name: digest(directory / name)
        for name in ASSETS
        if (directory / name).is_file()
    }
    if not {"tokenizer.json", "tokenizer_config.json"} <= assets.keys():
        raise ValueError(
            "actual tokenizer.json and tokenizer_config.json assets required"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    config = load(directory / "config.json")
    vocab = config.get("text_config", config)["vocab_size"]
    if type(vocab) is not int or vocab < len(tokenizer):
        raise ValueError("model vocabulary must contain every tokenizer ID")
    result = {
        "schema": SCHEMA,
        "recipe": RECIPE,
        "checkpoint": checkpoint,
        "model_config_sha256": digest(directory / "config.json"),
        "tokenizer_assets_sha256": assets,
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocabulary_size": len(tokenizer),
        "vocabulary_size": vocab,
        "add_special_tokens": True,
        "chat_template_applied": False,
        "passages": [],
    }
    for text in passages():
        ids = tokenizer(text, add_special_tokens=True).input_ids
        if len(ids) < 14:
            raise ValueError("passage has no scored targets")
        result["passages"].append({
            "text": text,
            "input_ids": ids,
            "target_indices": list(range(13, len(ids))),
        })
    save(output, result)
    return result


def corpus(path):
    obj = load(path)
    if obj.get("schema") != SCHEMA or obj.get("recipe") != RECIPE:
        raise ValueError("unsupported corpus schema or scoring recipe")
    rows = obj.get("passages", [])
    if len(rows) != 6 or [r["text"] for r in rows] != passages():
        raise ValueError("exact historical six passages required")
    if not obj.get("tokenizer_assets_sha256"):
        raise ValueError("tokenizer asset identities missing")
    if (
        type(obj.get("vocabulary_size")) is not int
        or obj["vocabulary_size"] <= 0
        or not obj.get("model_config_sha256")
    ):
        raise ValueError(
            "authoritative model configuration and vocabulary size required"
        )
    for row in rows:
        ids = row["input_ids"]
        if len(ids) < 14 or any(
            type(i) is not int or not 0 <= i < obj["vocabulary_size"] for i in ids
        ):
            raise ValueError("invalid passage token IDs")
        if row["target_indices"] != list(range(13, len(ids))):
            raise ValueError("target indices must be exactly 13..end")
    return obj


def provenance(obj):
    if not isinstance(obj, dict) or set(obj) != REQUIRED:
        raise ValueError("provenance requires exactly " + ", ".join(sorted(REQUIRED)))
    if any(
        not isinstance(obj[k], str) or not obj[k]
        for k in REQUIRED - {"precision_flags"}
    ):
        raise ValueError("provenance identities must be nonempty strings")
    if not isinstance(obj["precision_flags"], dict) or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in obj["precision_flags"].items()
    ):
        raise ValueError("precision_flags must map explicit flag names to strings")
    return obj


def matched(base, treatment, changes):
    provenance(base)
    provenance(treatment)
    if not isinstance(changes, dict) or set(changes) - {
        "package",
        "cache",
        "precision_flags",
    }:
        raise ValueError(
            "only declared package, cache or precision_flags treatment changes permitted"
        )
    expected = dict(base)
    expected.update(changes)
    if expected != treatment:
        raise ValueError(
            "baseline/treatment provenance mismatch outside declared changes"
        )
    if any(base[k] == v for k, v in changes.items()):
        raise ValueError("declared treatment change does not change provenance")


def validate_receipt(receipt):
    provenance(receipt["provenance"])
    validate_weight_identity(receipt["provenance"], receipt.get("weight_manifest"))
    values = receipt.get("passage_nll", [])
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("recipe") != RECIPE
        or len(values) != 6
    ):
        raise ValueError("unsupported or incomplete scoring receipt")
    if any(
        not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in values
    ):
        raise ValueError("invalid passage NLL")
    if receipt.get("mean_nll") != sum(values) / 6:
        raise ValueError("receipt mean does not equal six passage means")


def score(corpus_path, logits_path, output, baseline=None, changes=None):
    import torch

    frozen = corpus(corpus_path)
    export = torch.load(logits_path, map_location="cpu", weights_only=True)
    if export.get("corpus_sha256") != digest(corpus_path):
        raise ValueError("logit export corpus hash mismatch")
    prov = provenance(export["provenance"])
    validate_weight_identity(prov, export.get("weight_manifest"))
    rows = export["rows"]
    if len(rows) != 6 or export.get("target_indices") != [
        p["target_indices"] for p in frozen["passages"]
    ]:
        raise ValueError("six logit arrays aligned to target indices 13..end required")
    values = []
    for passage, logits in zip(frozen["passages"], rows):
        targets = passage["input_ids"][13:]
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or logits.shape[0] != len(targets)
        ):
            raise ValueError("logit rows must match every scored target exactly")
        if logits.shape[1] != frozen["vocabulary_size"]:
            raise ValueError(
                "exact model full vocabulary logits required; no extra padding or truncation"
            )
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("logits contain non-finite float32 values")
        lp = torch.log_softmax(logits.double(), -1)
        nll = float(-lp[torch.arange(len(targets)), torch.tensor(targets)].mean())
        values.append(nll)
    result = {
        "schema": SCHEMA,
        "recipe": RECIPE,
        "corpus_sha256": digest(corpus_path),
        "logits_sha256": digest(logits_path),
        "provenance": prov,
        "tokenizer_assets_sha256": frozen["tokenizer_assets_sha256"],
        "vocabulary_size": frozen["vocabulary_size"],
        "model_config_sha256": frozen["model_config_sha256"],
        "scored_token_counts": [len(p["target_indices"]) for p in frozen["passages"]],
        "passage_nll": values,
        "mean_nll": sum(values) / 6,
        "scoring_engine": "torch-" + torch.__version__ + ":cpu",
        "baseline_sha256": None,
        "declared_changes": {},
    }
    if "weight_manifest" in export:
        result["weight_manifest"] = export["weight_manifest"]
    if baseline:
        ref = load(baseline)
        validate_receipt(ref)
        if ref["baseline_sha256"] is not None:
            raise ValueError("reference must be an original baseline receipt")
        if (
            ref["corpus_sha256"] != result["corpus_sha256"]
            or ref["tokenizer_assets_sha256"] != result["tokenizer_assets_sha256"]
        ):
            raise ValueError("reference corpus/tokenizer mismatch")
        matched(ref["provenance"], prov, changes or {})
        result.update(baseline_sha256=digest(baseline), declared_changes=changes or {})
    elif changes:
        raise ValueError("treatment changes require a baseline receipt")
    save(output, result)
    return result


def compare(baseline, treatment, diagnostic=False):
    base, trial = load(baseline), load(treatment)
    validate_receipt(base)
    validate_receipt(trial)
    if base["baseline_sha256"] is not None or trial["baseline_sha256"] != digest(
        baseline
    ):
        raise ValueError("treatment is not content-bound to this original baseline")
    for key in (
        "corpus_sha256",
        "tokenizer_assets_sha256",
        "scored_token_counts",
        "vocabulary_size",
        "model_config_sha256",
    ):
        if base[key] != trial[key]:
            raise ValueError("baseline/treatment " + key + " mismatch")
    matched(base["provenance"], trial["provenance"], trial["declared_changes"])
    delta = trial["mean_nll"] - base["mean_nll"]
    result = {
        "baseline_sha256": digest(baseline),
        "treatment_sha256": digest(treatment),
        "mean_nll_delta": delta,
        "passage_nll_delta": [
            b - a for a, b in zip(base["passage_nll"], trial["passage_nll"])
        ],
        "scope": "six historical passages only; not global quality admission",
    }
    if diagnostic:
        result["original_diagnostic"] = {
            "ceiling_delta_nats": 0.010,
            "passes": delta <= 0.010,
            "scope": "original six-passage diagnostic only; no global floor",
        }
    return result
