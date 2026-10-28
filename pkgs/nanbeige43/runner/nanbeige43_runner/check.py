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
from .contention import admitted, flag_runs

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
    """Batch 8 is admitted only because the package disables fused QK for it.

    The packaged fused QK path leaves rows 5-7 with stale K at batch 8
    (docs/nanbeige-batch8.md), so the package patches `use_qk_fused` off for
    `max_batch_size == 8`. The parser must accept the size and the patch must be
    present in the installed tree; either one missing means batch 8 is corrupt or
    unreachable, and both are visible here rather than on a device run.
    """
    import contextlib
    import io
    from pathlib import Path

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
            # Reaching the checkpoint check proves the parser admitted the size.
            with contextlib.redirect_stderr(io.StringIO()):
                run_main(argv)
        except SystemExit as exit_error:
            admitted_size = "invalid choice" not in str(exit_error.code)
        else:
            admitted_size = True
        expect(admitted_size, f"{mode} admits --batch 8 at the parser", results)

    model_config = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "tt_transformers"
        / "tt"
        / "model_config.py"
    )
    source = model_config.read_text() if model_config.is_file() else ""
    expect(
        "if self.max_batch_size == 8:" in source
        and source.count("self.use_qk_fused = False") >= 2,
        "installed model_config disables fused QK for batch 8",
        results,
    )


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


def self_test_contention(results: list) -> None:
    """The contention screen must catch a collapsed run and leave clean ones alone.

    The numbers are recorded measurements, not invented ones. P99 produced a
    batch-16 session where one run's decode collapsed to 51.87 aggregate beside a
    29.204 s first token while the others ran at 210.07 and 183.72; the middle run
    was clean at 2.058 s. P101's clean session ran 216.10/207.44/220.16/208.46 with a
    4.916 s first token against a 1.940 s shortest - which must not be called an
    outlier at a threefold factor.
    """
    # ``contended`` is the historical receipt name; "admitted" reads as "admit
    # this run to the summary".
    p99 = [
        {"first_token_seconds": 29.204, "decode_tokens_per_second": 51.87},
        {"first_token_seconds": 2.058, "decode_tokens_per_second": 210.07},
        {"first_token_seconds": 11.328, "decode_tokens_per_second": 183.72},
    ]
    flags = flag_runs(p99)
    expect(
        flags["decode_contended"] == 1,
        "contention screen flags exactly the collapsed P99 run",
        results,
    )
    expect(
        flags["first_token_outlier"] == 2,
        "contention screen flags both P99 first-token stalls against the 2.058 s best",
        results,
    )
    expect(
        len(admitted(p99)) == 2,
        "contention screen admits the two P99 runs whose decode held",
        results,
    )
    p101 = [
        {"first_token_seconds": 1.985, "decode_tokens_per_second": 216.10},
        {"first_token_seconds": 1.940, "decode_tokens_per_second": 207.44},
        {"first_token_seconds": 4.916, "decode_tokens_per_second": 220.16},
        {"first_token_seconds": 2.007, "decode_tokens_per_second": 208.46},
    ]
    clean_flags = flag_runs(p101)
    expect(
        clean_flags["decode_contended"] == 0
        and clean_flags["first_token_outlier"] == 0,
        "contention screen leaves a clean session untouched",
        results,
    )
    expect(
        len(admitted(p101)) == 4,
        "contention screen admits every clean run",
        results,
    )
    single = [{"first_token_seconds": 1.9, "decode_tokens_per_second": 210.0}]
    expect(
        flag_runs(single) == {"decode_contended": 0, "first_token_outlier": 0},
        "contention screen does not flag a single run against itself",
        results,
    )


def self_test_prefix_cache(results: list) -> None:
    """The block index, LRU, refcounts and accounting must hold without a device.

    These are the facts the runner's receipt relies on: exact-token identity over
    32-token blocks, a miss on any changed token, shared read-only blocks pinned
    by leases, LRU eviction only of idle entries, a copy marker for partial tails,
    and cached/new/output counts that cannot bill recomputed tokens as cached.
    """
    from . import prefix_cache_core as cache

    identity_keys = {key: f"value-{key}" for key in cache.IDENTITY_KEYS}
    digest = cache.cache_identity(identity_keys)
    expect(len(digest) == 64, "prefix identity digests the exact key set", results)
    expect(
        cache.cache_identity({**identity_keys, "preset": "accuracy"}) != digest,
        "prefix identity changes with the preset",
        results,
    )

    def rejects(fn, message, results):
        try:
            fn()
        except cache.PrefixCacheError:
            expect(True, message, results)
            return
        expect(False, message, results)

    rejects(
        lambda: cache.cache_identity({
            key: value for key, value in identity_keys.items() if key != "mesh"
        }),
        "prefix identity rejects a missing key",
        results,
    )
    rejects(
        lambda: cache.cache_identity({**identity_keys, "extra": "x"}),
        "prefix identity rejects an unbound key",
        results,
    )

    tokens = list(range(1, 97))  # three 32-token blocks
    index = cache.PrefixCache(budget_blocks=8)
    entry = index.insert(digest, tokens, 96)
    expect(
        entry.tokens == 96 and entry.blocks == 3 and entry.generation == 1,
        "a 96-token prefix records three 32-token blocks",
        results,
    )
    expect(
        index.lookup(digest, tokens).cached_tokens == 96,
        "the full 96-token prefix hits",
        results,
    )
    expect(
        index.lookup(digest, tokens[:95]).cached_tokens == 64,
        "a 95-token prompt hits only its two whole blocks",
        results,
    )
    expect(
        index.lookup(digest, tokens[:31]).cached_tokens == 0,
        "31 tokens is below one block and misses",
        results,
    )
    altered = list(tokens)
    altered[70] ^= 1
    expect(
        index.lookup(digest, altered).cached_tokens == 64,
        "a changed token in the third block falls back to two blocks",
        results,
    )
    expect(
        not index.lookup(
            cache.cache_identity({**identity_keys, "preset": "accuracy"}), tokens
        ).hit,
        "the same tokens under a different preset miss",
        results,
    )
    expect(
        not index.lookup(digest, list(range(1000, 1096))).hit,
        "different tokens are a miss, not a prefix hit",
        results,
    )

    lease = index.retain(digest, tokens)
    expect(
        lease.hit
        and lease.copy_tail_block is False
        and lease.write_start_block == 3
        and lease.generation == 1,
        "an aligned lease writes from a fresh block and needs no tail copy",
        results,
    )
    expect(
        not index.writable(lease, 95) and index.writable(lease, 96),
        "shared prefix blocks are read-only; the next block is writable",
        results,
    )
    rejects(
        lambda: index.assert_writable(lease, 64),
        "a write into a shared prefix block is rejected",
        results,
    )
    expect(
        index.evict(target=0) == 0 and index.resident_blocks == 3,
        "a leased entry cannot be evicted",
        results,
    )
    index.release(lease)
    expect(
        index.evict(target=0) == 3 and index.resident_blocks == 0,
        "releasing the lease lets eviction free every block",
        results,
    )

    shared = cache.PrefixCache(budget_blocks=4)
    two_blocks = list(range(1, 65))
    shared.insert(digest, two_blocks)
    shared.insert(digest, two_blocks + [999] * 32)
    expect(
        shared.resident_blocks == 3
        and sorted(shared.block_references().values()) == [1, 2, 2],
        "overlapping prefixes share blocks with refcounts",
        results,
    )

    lru = cache.PrefixCache(budget_blocks=4)
    first = list(range(1, 65))
    second = list(range(101, 165))
    third = list(range(201, 265))
    lru.insert(digest, first)
    lru.insert(digest, second)
    lru.lookup(digest, first)  # touch the first entry so `second` is LRU
    expect(
        lru.evict(target=2) == 2
        and lru.lookup(digest, first).hit
        and not lru.lookup(digest, second).hit,
        "LRU evicts the untouched entry and keeps the touched one",
        results,
    )
    lru.insert(digest, third)
    expect(
        lru.resident_blocks == 4 and lru.lookup(digest, third).hit,
        "an insert evicts down to the block budget",
        results,
    )
    expect(lru.evicted_blocks == 2, "evicted block counts accumulate", results)

    tight = cache.PrefixCache(budget_blocks=2)
    rejects(
        lambda: tight.insert(digest, tokens, 96),
        "an over-budget prefix is rejected",
        results,
    )
    expect(
        tight.resident_blocks == 0,
        "a rejected insert leaves the index unchanged",
        results,
    )

    expect(
        cache.tail_copy_required(33) and not cache.tail_copy_required(32),
        "a partial tail block requires a copy marker",
        results,
    )
    expect(
        cache.block_statistics(96)
        == {"tokens": 96, "blocks": 3, "partial_tail_tokens": 0},
        "block statistics count whole blocks",
        results,
    )
    accounting = cache.request_accounting(96, 64, 32)
    expect(
        (
            accounting.cached_input_tokens,
            accounting.new_input_tokens,
            accounting.output_tokens,
        )
        == (64, 32, 32),
        "accounting splits cached and new input tokens",
        results,
    )
    for bad in ((96, 65, 1), (96, 97, 1), (-1, 0, 0), (96, -32, 1)):
        rejects(
            lambda bad=bad: cache.request_accounting(*bad),
            f"accounting rejects {bad}",
            results,
        )


def self_test_prefix_preflight(results: list) -> None:
    """The CLI guards must fail closed before a weight load.

    The alignment mirror is checked against the packaged helpers: a 128-token
    prefix over a 256-token prompt settles at 128 (alignment 64), while a 32-token
    prefix floors to 0 and would reuse nothing, so the runner must reject it.
    """
    from types import SimpleNamespace

    from .run import effective_resume_tokens, prefix_preflight

    def args(**overrides):
        base = {
            "batch": 1,
            "batched_prefill": False,
            "prompt_mode": "prefill",
            "page_block": 32,
            "page_blocks": 8192,
            "max_seq_len": 8192,
            "prefix_tokens": 128,
            "prefix_prompt_length": 256,
            "prefix_decode_tokens": 32,
            "prefix_cache_blocks": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    preflight = prefix_preflight(args())
    expect(
        preflight["prefix_blocks"] == 4
        and preflight["prefix_cache_blocks"] == 4
        and preflight["resume_alignment"]["prefix_tokens"] == 128
        and preflight["resume_alignment"]["alignment"] == 64,
        "prefix preflight admits a 128-token prefix on a 256-token prompt",
        results,
    )
    expect(
        effective_resume_tokens(256, 32)["prefix_tokens"] == 0,
        "a 32-token prefix floors to zero under the 64-token resume alignment",
        results,
    )
    # The packaged non-traced resumed prefill truncated the page table to the new
    # chunk and zero-padded the new pages, so the new K/V overwrote block 0
    # (device receipt prefix-cache-b1 diverged at token 0). The patch hands the
    # full prompt's cached and new pages to paged_fill_cache; the installed tree
    # must carry it or prefix-cache mode is silently wrong on a device.
    generator = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "tt_transformers"
        / "tt"
        / "generator.py"
    )
    source = generator.read_text() if generator.is_file() else ""
    expect(
        "use_full_prompt_len=(" in source and "int(start_pos[idx]) > 0" in source,
        "installed generator gives a resumed prefill the cached and new pages",
        results,
    )
    for overrides, message in (
        ({"batch": 2}, "prefix preflight rejects batch 2"),
        ({"batched_prefill": True}, "prefix preflight rejects batched prefill"),
        ({"prompt_mode": "decode"}, "prefix preflight rejects the decode prompt path"),
        ({"page_block": 64}, "prefix preflight rejects a 64-token page"),
        ({"prefix_tokens": 100}, "prefix preflight rejects an unaligned prefix"),
        (
            {"prefix_prompt_length": 128},
            "prefix preflight rejects a prefix as long as the prompt",
        ),
        (
            {"prefix_cache_blocks": 2},
            "prefix preflight rejects a budget below the prefix",
        ),
        (
            {
                "prefix_tokens": 256,
                "prefix_prompt_length": 8192,
                "prefix_decode_tokens": 1,
            },
            "prefix preflight rejects a pair that overflows the per-row pool",
        ),
        ({"prefix_decode_tokens": 0}, "prefix preflight rejects zero decode tokens"),
    ):
        try:
            prefix_preflight(args(**overrides))
        except SystemExit:
            expect(True, message, results)
        else:
            expect(False, message, results)


def self_test_prefix_diagnostics(results: list) -> None:
    """The cold/warm numeric comparison helpers must report what they see."""
    import torch

    from .run import kv_digest_parity, logits_parity

    reference = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    identical = logits_parity(reference, reference.clone())
    expect(
        identical["bit_identical"]
        and identical["pcc"] > 0.999999
        and identical["top1_equal"],
        "logits parity reports identical prefill logits",
        results,
    )
    perturbed = reference.clone()
    perturbed[0, 3] = 2.0
    report = logits_parity(reference, perturbed)
    expect(
        not report["bit_identical"]
        and report["max_abs_diff"] == 1.0
        and report["pcc"] < 1.0
        and report["top1_equal"] is False
        and len(report["rows"]) == 2,
        "logits parity reports a changed argmax and per-row diffs",
        results,
    )
    import copy

    cold = [
        {
            "layer": 0,
            "rows": [
                {"k_prefix": "a", "k_suffix": "b", "v_prefix": "c", "v_suffix": "d"}
            ],
        },
        {
            "layer": 1,
            "rows": [
                {"k_prefix": "e", "k_suffix": "f", "v_prefix": "g", "v_suffix": "h"}
            ],
        },
    ]
    warm = copy.deepcopy(cold)
    parity = kv_digest_parity(cold, warm)
    expect(
        parity["cached_prefix_pages_identical"]
        and parity["suffix_pages_identical"]
        and parity["first_suffix_mismatch_layer"] is None,
        "KV digest parity accepts identical caches",
        results,
    )
    warm[1]["rows"][0]["k_suffix"] = "x"
    suffix = kv_digest_parity(cold, warm)
    expect(
        suffix["cached_prefix_pages_identical"]
        and not suffix["suffix_pages_identical"]
        and suffix["suffix_mismatch_layers"] == [1]
        and suffix["first_suffix_mismatch_layer"] == 1
        and suffix["prefix_mismatches"] == [],
        "KV digest parity isolates a changed suffix page layer",
        results,
    )
    warm[0]["rows"][0]["v_prefix"] = "y"
    prefix = kv_digest_parity(cold, warm)
    expect(
        not prefix["cached_prefix_pages_identical"]
        and prefix["prefix_mismatches"]
        == [{"layer": 0, "row": 0, "region": "v_prefix"}],
        "KV digest parity reports a changed cached-prefix page",
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
    failure: BaseException | None = None
    try:
        report["memory_per_device_gb"] = check_metadata(
            args.metadata, authority, results
        )
        report["tokenizer"] = check_tokenizer(args.metadata, results)
        self_test_admission(results)
        self_test_comparison(results)
        self_test_batched_prefill(results)
        self_test_precision_table(results)
        self_test_batch_surface(results)
        self_test_contention(results)
        self_test_prefix_cache(results)
        self_test_prefix_preflight(results)
        self_test_prefix_diagnostics(results)
        if args.checkpoint:
            report["checkpoint"] = verify_checkpoint(args.checkpoint, authority)
            results.append({
                "check": f"checkpoint {args.checkpoint} admitted",
                "passed": True,
            })
    except (SystemExit, Exception) as error:  # noqa: BLE001 - report, then re-raise
        # A failing check must still leave a report: `expect` raises as soon as a
        # check fails, so without this the run left no artifact at all. Catching
        # broadly is the point here; the error is re-raised below.
        failure = error
        report["aborted"] = {
            "type": type(error).__name__,
            "message": str(error)[:4000],
        }
    report["results"] = results
    report["passed"] = failure is None and all(r["passed"] for r in results)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    if failure is not None:
        raise SystemExit(str(failure) or type(failure).__name__)


if __name__ == "__main__":
    main()
