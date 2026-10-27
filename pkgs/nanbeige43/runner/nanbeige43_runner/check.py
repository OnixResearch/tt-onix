# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.nanbeige43_p150x2.checkpoint_authority.valid]
# r[verify tenstorrent.nanbeige43_p150x2.checkpoint_authority.invalid]
"""Device-free checks for the Nanbeige4.3-33B P150x2 bring-up.

The check runs the packaged tt_transformers helpers (no device is opened)
against the pinned metadata. It verifies that the TP2 shapes, vocabulary
padding, MLP grid, RoPE parameters, host-sampling decision, per-device DRAM
budget, and tokenizer/chat template behave as the design records.

The budget assertions bind both the `accuracy` preset and whatever preset the
runner ships as `DEFAULT_PRESET`, and the per-preset tensor precisions are read
back from the runner's own `decoder_precision` so the budget table cannot drift
from the mix the package builds. The batched-prefill opt-in is checked against
the packaged limits rather than local copies. The check also exercises the
checkpoint admission code on synthetic checkpoints. With `--checkpoint` it
additionally verifies a real checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .authority import (
    CheckpointRejected,
    identity_digest,
    load_authority,
    sha256_file,
    verify_checkpoint,
)

TP = 2
# Blackhole P150: 8 DRAM banks of dram_bank_size 4278190080 B
# (tt_metal/soc_descriptors/blackhole_140_arch.yaml:112-113).
DEVICE_DRAM_BYTES = 8 * 4278190080
# Weights + KV must leave room for activations, program and trace buffers.
DEVICE_BUDGET_BYTES = 28 * 1024**3
BYTES = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625}
PRESETS = {
    # (WQKV/WO, FF1/FF3, FF2, KV) from ModelOptimizations for an unknown model name.
    "accuracy": ("bf16", "bfp8", "bfp8", "bf16"),
    "performance": ("bfp8", "bfp4", "bfp8", "bfp8"),
    # The all-BFP8 mix is the runner default (DEFAULT_PRESET); it is listed here so
    # the shipped configuration's device budget is checked, not just accuracy's.
    "bfp8": ("bfp8", "bfp8", "bfp8", "bfp8"),
    # The remaining pure-dtype mixes the CLI accepts. A mix whose budget depends on
    # which layers are quantized (bfp4-ff13-l13-39) cannot be expressed as one row.
    # Order is (WQKV/WO, FF1/FF3, FF2, KV); the runner's MIXES use
    # (attention, KV, FF1/FF3, FF2), so entries are not copied across verbatim.
    "bfp4-ff13": ("bfp8", "bfp4", "bfp8", "bfp8"),
    "bfp4-ff2": ("bfp8", "bfp8", "bfp4", "bfp8"),
}
# Mixes the runner offers that a single dtype row cannot describe.
LAYER_RANGE_MIXES = ("bfp4-ff13-l13-39",)


def expect(condition: bool, message: str, results: list) -> None:
    results.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise SystemExit(f"check failed: {message}")


def per_device_bytes(config: dict, preset: str, kv_tokens: int) -> dict:
    h, layers, heads, kv_heads, head_dim = (
        config["hidden_size"],
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    inter, vocab = config["intermediate_size"], config["vocab_size"]
    attn_p, ff13_p, ff2_p, kv_p = PRESETS[preset]
    attention = h * heads * head_dim * 2 + h * kv_heads * head_dim * 2
    layer_bytes = (
        attention * BYTES[attn_p]
        + 2 * h * inter * BYTES[ff13_p]
        + h * inter * BYTES[ff2_p]
    )
    budget = {
        "layers": layers * layer_bytes / TP,
        "lm_head": h * vocab * BYTES["bfp8"] / TP,
        "embedding": h * vocab * BYTES["bf16"] / TP,
        "kv_cache": layers * 2 * (kv_heads // TP) * head_dim * BYTES[kv_p] * kv_tokens,
    }
    budget["total"] = sum(budget.values())
    return budget


def check_metadata(metadata: Path, authority: dict, results: list) -> dict:
    hashes = {entry["name"]: entry["sha256"] for entry in authority["files"]}
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        expect(
            sha256_file(metadata / name) == hashes[name],
            f"{name} matches the authority",
            results,
        )

    from models.tt_transformers.tt.common import (
        get_base_model_name,
        get_rope_scaling,
        get_rope_theta,
        nearest_multiple,
    )
    from models.tt_transformers.tt.model_config import (
        ModelArgs,
        compute_padded_vocab_size,
    )
    from transformers import AutoConfig

    from .run import MLP_PAD_CORES

    raw = json.loads((metadata / "config.json").read_text())
    arch = authority["architecture"]
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
    ):
        expect(raw[key] == arch[key], f"config {key} == {arch[key]}", results)
    expect(
        raw["architectures"] == ["LlamaForCausalLM"] and raw["model_type"] == "llama",
        "plain LlamaForCausalLM",
        results,
    )
    expect(
        not raw["tie_word_embeddings"]
        and not raw["attention_bias"]
        and not raw["mlp_bias"],
        "untied LM head, no biases",
        results,
    )

    h, layers, inter, vocab = (
        raw["hidden_size"],
        raw["num_hidden_layers"],
        raw["intermediate_size"],
        raw["vocab_size"],
    )
    heads, kv_heads, head_dim = (
        raw["num_attention_heads"],
        raw["num_key_value_heads"],
        raw["head_dim"],
    )
    per_layer = (
        h * heads * head_dim * 2 + h * kv_heads * head_dim * 2 + 3 * h * inter + 2 * h
    )
    params = layers * per_layer + 2 * h * vocab + h
    expect(
        params == arch["parameter_count"],
        f"parameter count {params} == {arch['parameter_count']}",
        results,
    )
    index = json.loads((metadata / "model.safetensors.index.json").read_text())
    expect(
        index["metadata"]["total_size"] == 2 * params,
        "index total_size is BF16 of every parameter",
        results,
    )

    expect(
        heads % TP == 0 and kv_heads % TP == 0,
        "query and KV heads split across TP2",
        results,
    )
    expect(
        (inter // TP) % 32 == 0 and (h // TP) % 32 == 0,
        "per-device MLP and hidden widths are tile multiples",
        results,
    )
    padded = compute_padded_vocab_size(vocab, TP)
    expect(padded == vocab, f"vocabulary {vocab} needs no TP2 padding", results)
    expect(
        vocab // TP > 64 * 1024,
        "per-device vocabulary exceeds the on-device sampling limit: host sampling",
        results,
    )
    grid = ModelArgs.find_grid_k_n(None, h // 32, inter // TP // 32)
    expect(
        grid == (1, 8),
        f"unpadded decode MLP DRAM-sharded grid is {grid} (8 cores)",
        results,
    )
    padded_inter = nearest_multiple(inter, MLP_PAD_CORES * 32 * TP)
    expect(
        padded_inter == 22528,
        f"PAD_MLP_CORES={MLP_PAD_CORES} pads the MLP hidden size to {padded_inter}",
        results,
    )
    padded_grid = ModelArgs.find_grid_k_n(None, h // 32, padded_inter // TP // 32)
    expect(
        padded_grid[0] * padded_grid[1] == 32,
        f"padded decode MLP DRAM-sharded grid is {padded_grid} (32 cores)",
        results,
    )
    hf = AutoConfig.from_pretrained(metadata).to_dict()
    expect(
        get_rope_theta(hf) == raw["rope_theta"] == 120000000,
        "rope_theta 1.2e8 reaches tt_transformers",
        results,
    )
    expect(get_rope_scaling(hf) is None, "no RoPE scaling is applied", results)
    expect(
        get_base_model_name("nanbeige4.3-33b-c1093c501cc6") == "nanbeige4.3-33b",
        "base model name matches no per-model table",
        results,
    )

    # Budgets use the padded MLP hidden size the runner loads.
    budgets = {
        preset: per_device_bytes(
            {**raw, "intermediate_size": padded_inter}, preset, 8192
        )
        for preset in PRESETS
    }
    # Every parameter in BF16 plus an 8k BF16 KV pool, per device.
    bf16_total = params * 2 / TP + layers * 2 * (kv_heads // TP) * head_dim * 2 * 8192
    expect(
        DEVICE_DRAM_BYTES - bf16_total < 1e9,
        f"BF16 everywhere ({bf16_total / 1e9:.2f} GB) leaves under 1 GB of the {DEVICE_DRAM_BYTES / 1e9:.2f} GB per device",
        results,
    )
    from .run import MIXES
    from .run import PRESETS as RUNNER_PRESETS

    # Every preset the CLI accepts must fit the device budget, not just accuracy and
    # the default: a selectable preset that cannot load is a broken surface.
    for preset in sorted(PRESETS):
        expect(
            budgets[preset]["total"] <= DEVICE_BUDGET_BYTES,
            f"preset {preset} {budgets[preset]['total'] / 1e9:.2f} GB fits the device budget",
            results,
        )
    # Coverage: a new mix must come with a budget row, or its budget is unchecked.
    unmodelled = [
        mix for mix in MIXES if mix not in PRESETS and mix not in LAYER_RANGE_MIXES
    ]
    expect(
        not unmodelled,
        f"every runner mix has a budget row; unmodelled {unmodelled}",
        results,
    )
    expect(
        set(RUNNER_PRESETS)
        == set(PRESETS) | {"accuracy", "performance"} | set(LAYER_RANGE_MIXES),
        f"checker presets cover the runner's presets {sorted(RUNNER_PRESETS)}",
        results,
    )
    return {
        preset: {key: round(value / 1e9, 2) for key, value in budget.items()}
        for preset, budget in budgets.items()
    }


def check_tokenizer(metadata: Path, results: list) -> dict:
    from transformers import AutoTokenizer

    from .prompts import chat_ids

    tokenizer = AutoTokenizer.from_pretrained(metadata)
    expect(
        (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        == (166100, 166101, 0),
        "BOS/EOS/pad ids",
        results,
    )
    slow = AutoTokenizer.from_pretrained(metadata, use_fast=False)
    probes = [
        "Hello, world!",
        "南北阁是一款大语言模型。",
        "def f(x):\n    return x**2\n",
    ]
    expect(
        all(slow.encode(p) == tokenizer.encode(p) for p in probes),
        "use_fast=False and default tokenizers agree",
        results,
    )
    messages = [{"role": "user", "content": "What is 2+2?"}]
    ids = chat_ids(tokenizer, messages, False)
    expect(
        ids[0] == 166100 and ids[1] != 166100,
        "chat template emits one leading <|im_start|>",
        results,
    )
    text = tokenizer.decode(ids)
    expect(
        text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "enable_thinking=False closes an empty think block",
        results,
    )
    return {"vocabulary_length": len(tokenizer), "chat_prompt_tokens": len(ids)}


def self_test_admission(results: list) -> None:
    """Exercise accept and reject paths of the admission code on synthetic checkpoints."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        source = root / "full"
        source.mkdir()
        files = {"model-00001-of-00001.safetensors": b"weights", "config.json": b"{}"}
        for name, data in files.items():
            (source / name).write_bytes(data)
        (source / "README.md").write_text("ignored")
        authority = {
            "revision": "r",
            "architecture": {"num_hidden_layers": 2},
            "files": [
                {"name": n, "size": len(d), "sha256": sha256_file(source / n)}
                for n, d in files.items()
            ],
        }
        identity = verify_checkpoint(source, authority)
        expect(
            identity["kind"] == "full",
            "admission accepts a matching checkpoint",
            results,
        )

        def rejected(mutate, message, **kwargs):
            case = root / message.replace(" ", "-")
            case.mkdir()
            for name, data in files.items():
                (case / name).write_bytes(data)
            mutate(case)
            try:
                verify_checkpoint(case, authority, **kwargs)
            except CheckpointRejected:
                expect(True, f"admission rejects {message}", results)
                return
            expect(False, f"admission rejects {message}", results)

        rejected(lambda d: (d / "config.json").write_bytes(b"[]"), "an altered file")
        rejected(lambda d: (d / "config.json").unlink(), "a missing file")
        rejected(
            lambda d: (d / "model-00002-of-00002.safetensors").write_bytes(b"x"),
            "an extra shard",
        )
        rejected(
            lambda d: None, "a full checkpoint in reduced-only mode", allow_full=False
        )

        reduced = root / "reduced"
        reduced.mkdir()
        (reduced / "model.safetensors").write_bytes(b"small")
        outputs = {"model.safetensors": sha256_file(reduced / "model.safetensors")}
        manifest = {
            "schema": "nanbeige43-reduced-checkpoint/v1",
            "source_revision": "r",
            "layers": 1,
            "source_files": {"config.json": authority["files"][1]["sha256"]},
            "files": outputs,
        }
        (reduced / "nanbeige43-reduced.json").write_text(json.dumps(manifest))
        expect(
            verify_checkpoint(reduced, authority, allow_full=False)["kind"]
            == "reduced",
            "admission accepts a matching reduced checkpoint",
            results,
        )
        try:
            verify_checkpoint(reduced, authority, allow_reduced=False)
            expect(
                False,
                "admission rejects a reduced checkpoint where only full is admitted",
                results,
            )
        except CheckpointRejected:
            expect(
                True,
                "admission rejects a reduced checkpoint where only full is admitted",
                results,
            )
        manifest["source_files"] = {"config.json": "0" * 64}
        (reduced / "nanbeige43-reduced.json").write_text(json.dumps(manifest))
        try:
            verify_checkpoint(reduced, authority)
            expect(
                False,
                "admission rejects a reduced checkpoint from foreign source files",
                results,
            )
        except CheckpointRejected:
            expect(
                True,
                "admission rejects a reduced checkpoint from foreign source files",
                results,
            )
        expect(
            identity_digest({"a": "1"}) != identity_digest({"a": "2"}),
            "identity digest binds file hashes",
            results,
        )


def self_test_comparison(results: list) -> None:
    """A correct first user must not conceal another user's corrupted logits."""
    import torch

    from .run import batch_row_report, evaluate_gates

    reference = torch.arange(6, dtype=torch.float32)
    corpus = {"prefill": {"pcc": 1.0}, "top1": 1.0, "top5": 1.0}
    for rows, accepted in (
        (reference.unsqueeze(0), True),
        (torch.stack([reference, reference]), True),
        (torch.stack([reference, reference.flip(0)]), False),
    ):
        report = batch_row_report(rows, reference)
        gates = evaluate_gates(
            {"kind": "full"}, [{"name": "users", "rows": [report]}], corpus
        )
        expect(
            all(gate["passed"] for gate in gates) == accepted,
            f"comparison {'accepts matching users' if accepted else 'rejects a hidden corrupted user'} (batch {len(rows)})",
            results,
        )


def self_test_batched_prefill(results: list) -> None:
    """P5's opt-in must be gated by the packaged upstream limits, not local copies."""
    from models.tt_transformers.tt.common import get_padded_prefill_len
    from models.tt_transformers.tt.generator import (
        MAX_BATCHED_PREFILL_SEQ_LEN,
        SUPPORTED_PREFILL_BATCH_SIZES,
        batched_prefill_padded_batch,
    )

    batch = 32
    slots = list(range(batch))
    padded = batched_prefill_padded_batch(batch, slots, batch)
    expect(
        padded == batch,
        f"batched prefill keeps {batch} rows on one pass (padded batch {padded})",
        results,
    )
    prefill_len = get_padded_prefill_len(128)
    expect(
        padded * prefill_len < MAX_BATCHED_PREFILL_SEQ_LEN,
        f"a {padded} x {prefill_len} batched prefill stays under the packaged {MAX_BATCHED_PREFILL_SEQ_LEN}-token limit",
        results,
    )
    expect(
        batch in SUPPORTED_PREFILL_BATCH_SIZES,
        f"batch {batch} is a packaged supported prefill batch {SUPPORTED_PREFILL_BATCH_SIZES}",
        results,
    )


def self_test_batch_surface(results: list) -> None:
    """The shipped command must refuse the batch sizes it cannot serve correctly.

    Batch 8 is upstream-supported but writes stale rows 5-7 on this path
    (P42-P45), so both modes refuse it. The check drives the real parser, so a
    widening of `ACCEPTED_BATCHES` fails here instead of on a device run.
    """
    import contextlib
    import io

    from .run import main as run_main

    for mode in ("bench", "generate"):
        argv = [
            mode,
            "--mesh",
            "p150x2",
            "--batch",
            "8",
            "--checkpoint",
            "/nonexistent",
            "--cache-root",
            "/nonexistent",
            "--receipt",
            "/nonexistent.json",
        ]
        try:
            # argparse writes the usage and error to stderr before raising; a
            # passing check must not emit "error:" lines into its own output.
            with contextlib.redirect_stderr(io.StringIO()):
                run_main(argv)
        except SystemExit as exit_error:
            refused = "invalid choice" in str(exit_error.code) or exit_error.code == 2
        else:
            refused = False
        expect(refused, f"{mode} refuses --batch 8 at the parser", results)


def self_test_precision_table(results: list) -> None:
    """The budget table must describe the precision the runner actually builds.

    `per_device_bytes` turns PRESETS into a device tally, so a change to the
    runner's MIXES that is not mirrored here would make the budget check validate
    a mix the package does not run.
    """
    from types import SimpleNamespace

    from models.tt_transformers.tt.model_config import TensorGroup

    from .run import decoder_precision

    names = {"bf16": "BFLOAT16", "bfp8": "BFLOAT8_B", "bfp4": "BFLOAT4_B"}
    stub = SimpleNamespace(n_layers=2, model_name="nanbeige4.3-33b")
    for preset, (attention, ff13, ff2, kv) in PRESETS.items():
        precision = decoder_precision(preset)(stub)
        observed = {
            group: getattr(precision.get_tensor_dtype(0, group), "name", None)
            for group in (
                TensorGroup.WQKV,
                TensorGroup.WO,
                TensorGroup.FF1_FF3,
                TensorGroup.FF2,
                TensorGroup.KV_CACHE,
            )
        }
        expect(
            observed[TensorGroup.WQKV] == observed[TensorGroup.WO] == names[attention]
            and observed[TensorGroup.FF1_FF3] == names[ff13]
            and observed[TensorGroup.FF2] == names[ff2]
            and observed[TensorGroup.KV_CACHE] == names[kv],
            f"{preset} tensor precisions match the budget table {observed}",
            results,
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="directory with the pinned config/tokenizer/index files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="also verify this checkpoint directory (hashes every shard)",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    authority = load_authority()
    results: list = []
    report = {
        "schema": "nanbeige43-device-free-check/v1",
        "revision": authority["revision"],
    }
    report["memory_per_device_gb"] = check_metadata(args.metadata, authority, results)
    report["tokenizer"] = check_tokenizer(args.metadata, results)
    self_test_admission(results)
    self_test_comparison(results)
    self_test_batched_prefill(results)
    self_test_precision_table(results)
    self_test_batch_surface(results)
    if args.checkpoint:
        report["checkpoint"] = verify_checkpoint(args.checkpoint, authority)
        results.append({
            "check": f"checkpoint {args.checkpoint} admitted",
            "passed": True,
        })
    report["results"] = results
    report["passed"] = all(r["passed"] for r in results)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
