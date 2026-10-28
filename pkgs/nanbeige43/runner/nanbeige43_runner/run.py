# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.reference_agreement]
"""Run Nanbeige4.3-33B through tt_transformers and compare it with the CPU reference.

`compare` prefills each reference prompt, then decodes the reference
continuation teacher-forced, and compares every logits row with the reference.
With a full checkpoint it also runs the teacher-forced corpus gate.
`generate` runs a free greedy chat completion.
`bench` measures first-token latency, decode tokens/s and prefill tokens/s on a
fixed chat prompt under the O0 protocol.
`prefix-cache` runs one bounded cold/warm pair in a single session: the cold pass
resets, prefills and decodes; the warm pass keeps that KV and page table and
prefills the same prompt from a cached 32-token-aligned prefix through upstream's
sequential prefix-cached path, then decodes. Greedy ids must match and the
receipt records cached/new/output tokens with both passes' timings.
Every mode writes a receipt and exits non-zero when a gate fails.

Greedy decoding argmaxes the device logits and reads back one index per row
(`--no-device-argmax` keeps the host path); `compare` keeps the full-logit path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

import torch

from . import prefix_cache_core as prefix_cache
from .authority import load_authority, sha256_file, verify_checkpoint
from .contention import admitted, flag_runs
from .mesh import MESHES, open_mesh
from .prompts import chat_ids, corpus_tokens

RECEIPT_SCHEMA = "nanbeige43-device-receipt/v1"
# Gates declared before any device result (design.md, Validation ladder).
REDUCED_ROW_PCC = 0.99  # every compared row; device argmax must equal reference argmax
FULL_PREFILL_PCC = (
    0.94  # upstream full-model accuracy-mode PCC floor (tests/test_model.py:136)
)
FULL_TOP1 = 0.87
FULL_TOP5 = 0.97
TEACHER_FORCED_POSITIONS = 512
MLP_PAD_CORES = 16
# Fixed decode trace region when --trace is set: upstream reserves 90 MB for the 64-layer
# Qwen3-32B on bh_p150/bh_p300 (models/model_trace_region_sizes.yaml); 128 MiB covers 52 layers.
TRACE_REGION_BYTES = 128 * 1024 * 1024
# Batch sizes admitted by predeclared gates (tasks.md): 2 and 4 bit-equal to batch 1 (P12, O3),
# 16 and 32 under the teacher-forced fallback (O3, P2). Batch 8 is admitted since P103: the
# packaged model is patched to use the non-fused QK path there, because the fused path leaves
# rows 5-7 with stale K for every head (docs/nanbeige-batch8.md). The correction that made it
# worth doing is in P103 - the "113.8 for two batch-4 streams" figure was a SUM of two runs, not
# two concurrent streams, so the real alternative for eight rows is 56.91 aggregate and the
# non-fused path at 101.77 is 1.79x better. Batch > 32 needs a tile-row dataflow change (P10/P11).
ACCEPTED_BATCHES = (1, 2, 4, 8, 16, 32)
# Prefix-cache mode runs every row through upstream's sequential per-row prefill
# (`disable_batched_prefill`), which accepts a per-user `start_pos`; the batched
# prefill path zeroes the cached offset. Only the row counts issue #11 names are
# admitted here, so a batch whose prefix path was never exercised fails closed.
PREFIX_CACHE_BATCHES = (1, 8, 32)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@contextlib.contextmanager
def single_threaded_host_math():
    """Set torch to one thread for the accuracy metrics.

    These tensors hold vocabulary rows, so the work is a few hundred thousand
    element operations per row; torch's default pool costs far more than the
    arithmetic at batch 1 (587.96 ms against 7.22 ms per step) and dominates the
    teacher-forced phase at batch 32 (P23/P24).
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def row_report(device: torch.Tensor, reference: torch.Tensor) -> dict:
    with single_threaded_host_math():
        return _row_report(device, reference)


def _row_report(device: torch.Tensor, reference: torch.Tensor) -> dict:
    ref_top5 = torch.topk(reference, 5).indices.tolist()
    dev_top5 = torch.topk(device, 5).indices.tolist()
    ref_sorted = torch.sort(reference, descending=True).values
    diff = device.float() - reference.float()
    device_top1_reference_logit = reference[dev_top5[0]]
    return {
        "pcc": pcc(device, reference),
        "device_top1": dev_top5[0],
        "reference_top1": ref_top5[0],
        "top1_equal": dev_top5[0] == ref_top5[0],
        "device_top1_in_reference_top5": dev_top5[0] in ref_top5,
        "device_top1_rank_in_reference": int(
            (reference > device_top1_reference_logit).sum()
        ),
        "reference_top1_top2_margin": float(ref_sorted[0] - ref_sorted[1]),
        # Reference logit of its own top-1 minus its logit for the device's top-1 (0 when equal).
        "reference_gap_to_device_top1": float(
            ref_sorted[0] - device_top1_reference_logit
        ),
        "max_abs_diff": float(diff.abs().max()),
        "rms_diff": float(diff.pow(2).mean().sqrt()),
        "reference_std": float(reference.float().std()),
    }


def batch_row_report(device: torch.Tensor, reference: torch.Tensor) -> dict:
    """Keep each user's evidence and gate on the least-correlated logits row."""
    rows = [row_report(row, reference) for row in device]
    report = dict(min(rows, key=lambda row: row["pcc"]))
    report["top1_equal"] = all(row["top1_equal"] for row in rows)
    if len(rows) > 1:
        report["batch_rows"] = rows
    return report


# Upstream presets, plus the Phase 6 (data-parallel) screening mixes. `bfp8` (all-BFP8) is the
# runner default since P9 and `accuracy` stays selectable; the rest are screening selections.
# Each mix is
# (WQKV/WO, KV cache, FF1/FF3, FF2, BFP4 FF1/FF3 layer range or None for every layer);
# fidelities stay at the upstream defaults (MLP HiFi2-FP16) except where the preset sets them.
MIXES = {
    "bfp8": ("bfp8", "bfp8", "bfp8", "bfp8", None),
    "bfp4-ff13": ("bfp8", "bfp8", "bfp4", "bfp8", None),
    "bfp4-ff13-l13-39": ("bfp8", "bfp8", "bfp4", "bfp8", range(13, 40)),
    "bfp4-ff2": ("bfp8", "bfp8", "bfp8", "bfp4", None),
}
PRESETS = ("accuracy", "performance", *MIXES)
# O5's predeclared rule promotes the all-BFP8 mix once its speed gates pass;
# P9 re-earned them on the device-argmax path (batch 1 +14.8 %, batch 32
# +11.5 %). Accuracy stays selectable for maximum fidelity: the mix costs 1.37
# points of teacher-forced top-1 (0.921875 -> 0.908203).
DEFAULT_PRESET = "bfp8"


def decoder_precision(preset: str):
    """Map a preset or mix name to the `optimizations` callable ModelArgs accepts."""
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        ModelOptimizations,
        PrecisionSetting,
        TensorGroup,
    )

    if preset not in MIXES:
        factory = getattr(DecodersPrecision, preset)
        return lambda model_args: factory(model_args.n_layers, model_args.model_name)
    attention, kv, ff13, ff2, bfp4_layers = MIXES[preset]

    def optimizations(ff13_dtype: str) -> ModelOptimizations:
        return ModelOptimizations({
            "TensorPrecision": {
                TensorGroup.WQKV: PrecisionSetting(attention),
                TensorGroup.WO: PrecisionSetting(attention),
                TensorGroup.KV_CACHE: PrecisionSetting(kv),
                TensorGroup.FF1_FF3: PrecisionSetting(ff13_dtype),
                TensorGroup.FF2: PrecisionSetting(ff2),
            }
        })

    def build(model_args):
        if bfp4_layers is None:
            return DecodersPrecision(
                model_args.n_layers, model_args.model_name, optimizations(ff13)
            )
        precision = DecodersPrecision(
            model_args.n_layers, model_args.model_name, optimizations("bfp8")
        )
        for layer in bfp4_layers:
            precision.set_decoder_conf(layer, optimizations(ff13))
        return precision

    return build


def host_rss() -> dict:
    """Current and peak resident set of this process (bytes), from /proc/self/status."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) * 1024
    return {"rss": fields["VmRSS"], "peak": fields["VmHWM"]}


class Session:
    """One model on the admitted mesh, with paged KV and greedy sampling.

    `bench`/`generate` argmax on the device and read back one index per row
    (P8 default); `compare` keeps the full-logit path.
    """

    def __init__(
        self,
        mesh_device,
        preset: str,
        max_seq_len: int,
        page_block: int,
        page_blocks: int,
        prompt_mode: str = "prefill",
        enable_trace: bool = False,
        batch: int = 1,
        batched_prefill: bool = False,
        device_argmax: bool = False,
    ):
        import ttnn
        from models.tt_transformers.tt.common import (
            PagedAttentionConfig,
            create_tt_model,
        )
        from models.tt_transformers.tt.generator import Generator

        self.ttnn = ttnn
        self.prompt_mode = prompt_mode
        # Decode-only trace capture/replay (Generator._decode_forward_trace_text); prefill stays eager.
        self.enable_trace = enable_trace
        self.batch = batch
        # Upstream's batched prefill runs padded_batch equal-length prompts in one
        # forward pass. The runner keeps the sequential per-row path unless asked
        # because upstream disables batched prefill selectively (tt-metal#47238:
        # prefill logits are batch-variant under seeded sampling). P5 measures it.
        self.batched_prefill = batched_prefill and batch > 1
        # Device argmax instead of the gathered full-vocab readback (P8).
        self.device_argmax = device_argmax and prompt_mode != "decode"
        if page_blocks % batch:
            raise SystemExit("--page-blocks must divide evenly across the batch rows")
        if page_block * page_blocks < max_seq_len:
            raise SystemExit("paged KV pool is smaller than --max-seq-len")
        paged = PagedAttentionConfig(block_size=page_block, max_num_blocks=page_blocks)
        precision = decoder_precision(preset)
        started = time.time()
        self.args, self.model, kv_cache, state_dict = create_tt_model(
            mesh_device,
            instruct=False,
            max_batch_size=batch,
            optimizations=precision,
            max_seq_len=max_seq_len,
            paged_attention_config=paged,
            dtype=ttnn.bfloat8_b,
        )
        if batch > 1:
            # Sequential per-row prefill is the default; --batched-prefill opts into
            # upstream's single-pass path (see self.batched_prefill).
            self.args.disable_batched_prefill = not self.batched_prefill
        self.kv_cache = [kv_cache]
        del state_dict
        self.load_seconds = time.time() - started
        # Deterministic shuffled block mapping, so paging is exercised reproducibly.
        permutation = torch.randperm(
            page_blocks, generator=torch.Generator().manual_seed(0)
        )
        table = torch.argsort(permutation).reshape(batch, page_blocks // batch)
        self.page_table = table
        self.generator = Generator(
            [self.model], [self.args], mesh_device, tokenizer=self.args.tokenizer
        )
        self.vocab = self.args.vocab_size
        self.max_seq_len = max_seq_len
        # Each row owns page_blocks // batch blocks of the pool (see page_table).
        self.row_tokens = min(max_seq_len, page_block * (page_blocks // batch))
        self.mesh_device = mesh_device
        self.l1_log: list[dict] = []
        self.l1_snapshot("loaded")

    def l1_snapshot(self, label: str) -> None:
        """Record the worker-L1 allocator state (per bank) of the first instance."""
        view = self.ttnn.get_memory_view(self.mesh_device, self.ttnn.BufferType.L1)
        self.l1_log.append({
            "label": label,
            "bytes_per_bank": int(view.total_bytes_per_bank),
            "allocated_per_bank": int(view.total_bytes_allocated_per_bank),
            "largest_contiguous_free_per_bank": int(
                view.largest_contiguous_bytes_free_per_bank
            ),
        })

    def close(self) -> None:
        """Release decode traces while the mesh is still open (Generator.__del__ runs too late)."""
        store = getattr(self.generator, "trace_ids_decode", {})
        for key, trace_ids in store.items():
            for trace_id in (trace_ids or {}).values():
                if trace_id is not None:
                    self.ttnn.release_trace(self.mesh_device, trace_id)
            store[key] = None

    def reset(self):
        for layer in self.model.layers:
            k_cache, v_cache = layer.attention.layer_past
            self.ttnn.mul(k_cache, 0, output_tensor=k_cache)
            self.ttnn.mul(v_cache, 0, output_tensor=v_cache)
        self.generator.prev_page_table = None
        self.first_decode = True

    def prefill(self, ids: list[int]) -> torch.Tensor:
        return self.prefill_rows([ids] * self.batch)[0]

    def decode(self, token: int, position: int) -> torch.Tensor:
        return self.decode_rows([token] * self.batch, [position] * self.batch)[0]

    def prefill_rows(
        self, rows: list[list[int]], start_pos: list[int] | None = None
    ) -> torch.Tensor:
        """Prefill one prompt per batch row; returns last-position logits [batch, vocab].

        `start_pos` is the per-row count of already-cached tokens. The row's tokens
        are still the full prompt; upstream slices off `start_pos[row]` before the
        prefill and writes K/V from that absolute position (its sequential
        prefix-cached path). `None` is the ordinary prefill from token 0.
        """
        if len(rows) != self.batch:
            raise ValueError(f"expected {self.batch} rows, got {len(rows)}")
        if start_pos is not None and len(start_pos) != self.batch:
            raise ValueError(
                f"expected {self.batch} start positions, got {len(start_pos)}"
            )
        lengths = [len(ids) for ids in rows]
        if max(lengths) >= self.row_tokens:
            raise SystemExit(
                f"prompt of {max(lengths)} tokens does not fit the per-row KV budget of {self.row_tokens}"
            )
        if start_pos is not None and max(start_pos) >= min(lengths):
            raise SystemExit(
                f"cached prefix {max(start_pos)} must be shorter than the {min(lengths)}-token prompt"
            )
        if self.prompt_mode == "decode":
            if self.batch != 1:
                raise SystemExit("--prompt-mode decode supports batch 1 only")
            if start_pos is not None and any(start_pos):
                raise SystemExit(
                    "--prompt-mode decode has no cached-prefix ingestion path"
                )
            # Token-at-a-time ingestion through the decode path (see --prompt-mode).
            for position, token in enumerate(rows[0]):
                logits = self.decode_rows([token], [position])
            return logits
        tokens = torch.zeros(self.batch, max(lengths), dtype=torch.long)
        for row, ids in enumerate(rows):
            tokens[row, : len(ids)] = torch.tensor(ids)
        logits = self.generator.prefill_forward_text(
            tokens,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=lengths,
            enable_trace=False,
            warmup_prefill=False,
            sampling_params=None,
            start_pos=start_pos,
        )
        return logits.reshape(self.batch, -1)[:, : self.vocab].float()

    def decode_rows(self, tokens: list[int], positions: list[int]) -> torch.Tensor:
        """One decode step for every row; returns logits [batch, vocab]."""
        if max(positions) >= self.row_tokens:
            raise SystemExit(
                f"decode position {max(positions)} exceeds the per-row KV budget of {self.row_tokens}"
            )
        logits, _ = self.generator.decode_forward(
            torch.tensor(tokens, dtype=torch.long).reshape(self.batch, 1),
            torch.tensor(positions),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            enable_trace=self.enable_trace,
            reset_batch=self.first_decode,
            sampling_params=None,
        )
        self.first_decode = False
        return logits.reshape(self.batch, -1)[:, : self.vocab].float()

    def decode_greedy(self, tokens: list[int], positions: list[int]) -> list[int]:
        """Sample active rows only; compare retains the full-logit decode path."""
        if max(positions) >= self.row_tokens:
            raise SystemExit(
                f"decode position {max(positions)} exceeds the per-row KV budget of {self.row_tokens}"
            )
        device_output = self.generator.decode_forward(
            torch.tensor(tokens, dtype=torch.long).reshape(self.batch, 1),
            torch.tensor(positions),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            enable_trace=self.enable_trace,
            reset_batch=self.first_decode,
            sampling_params=None,
            read_from_device=False,
        )
        self.first_decode = False
        if self.device_argmax:
            # P8: argmax the device logits the trace already produced and read back
            # one index per row instead of the gathered [batch, vocab] tensor.
            logits = device_output[0][0]
            indices = self.ttnn.argmax(logits, dim=-1)
            host = self.ttnn.to_torch(self.ttnn.get_device_tensors(indices)[0])
            return [int(index) for index in host.reshape(-1)[: self.batch]]
        # Blocking read owns the host tensor through argmax. Candidate 0.77
        # model.py:833-860 all-gathers the full vocabulary onto each device.
        host_output = self.generator.read_decode_output(device_output)
        shard = self.ttnn.get_device_tensors(host_output[0][0])[0]
        rows = self.ttnn.to_torch(shard)[0, 0, : self.batch, : self.vocab]
        return rows.float().argmax(-1).tolist()


def compare_prompts(session: Session, reference: dict) -> list[dict]:
    reports = []
    for prompt in reference["prompts"]:
        ids, continuation, ref_logits = (
            prompt["prompt_ids"],
            prompt["continuation_ids"],
            prompt["logits"],
        )
        session.reset()
        started = time.time()
        rows = [
            batch_row_report(session.prefill_rows([ids] * session.batch), ref_logits[0])
        ]
        session.l1_snapshot(f"after-prefill:{prompt['name']}")
        prefill_seconds = time.time() - started
        started = time.time()
        for step, token in enumerate(continuation):
            rows.append(
                batch_row_report(
                    session.decode_rows(
                        [token] * session.batch, [len(ids) + step] * session.batch
                    ),
                    ref_logits[step + 1],
                )
            )
        decode_seconds = time.time() - started
        session.l1_snapshot(f"after-decode:{prompt['name']}")
        reports.append({
            "name": prompt["name"],
            "prompt_tokens": len(ids),
            "decode_steps": len(continuation),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "rows": rows,
        })
        worst = min(row["pcc"] for row in rows)
        equal = sum(row["top1_equal"] for row in rows)
        print(
            f"[compare] {prompt['name']}: prefill pcc {rows[0]['pcc']:.5f}, worst pcc {worst:.5f}, top1 equal {equal}/{len(rows)}",
            flush=True,
        )
    return reports


def teacher_forced(session: Session, corpus: dict) -> dict:
    """Teacher-forced decode on every batch row (identical inputs); gates use the worst row."""
    tokens = corpus["reference_tokens"][0].tolist()
    split = corpus["split"]
    top5 = corpus["top5_tokens"][split - 1 :]
    window = corpus["window_logits"]
    batch = session.batch
    session.reset()
    started = time.time()
    logits = session.prefill_rows([tokens[:split]] * batch)
    prefill = min(
        (row_report(row, window[0].float()) for row in logits),
        key=lambda report: report["pcc"],
    )
    prefill_seconds = time.time() - started
    predicted = [[int(token)] for token in logits.argmax(-1)]
    decode_pcc = []
    started = time.time()
    for step in range(1, TEACHER_FORCED_POSITIONS):
        position = split + step - 1
        logits = session.decode_rows([tokens[position]] * batch, [position] * batch)
        with single_threaded_host_math():
            for row, row_logits in enumerate(logits):
                predicted[row].append(int(row_logits.argmax()))
                decode_pcc.append(pcc(row_logits, window[step].float()))
    decode_seconds = time.time() - started
    positions = range(TEACHER_FORCED_POSITIONS)
    row_top1 = [
        sum(row[i] == int(top5[i, 0]) for i in positions) / TEACHER_FORCED_POSITIONS
        for row in predicted
    ]
    row_top5 = [
        sum(row[i] in top5[i].tolist() for i in positions) / TEACHER_FORCED_POSITIONS
        for row in predicted
    ]
    top1, top5_hits = min(row_top1), min(row_top5)
    print(
        f"[teacher-forced] prefill pcc {prefill['pcc']:.5f}, top1 {top1:.4f}, top5 {top5_hits:.4f}, min decode pcc {min(decode_pcc):.5f} (worst of {batch} rows)",
        flush=True,
    )
    result = {
        "split": split,
        "positions": TEACHER_FORCED_POSITIONS,
        "prefill": prefill,
        "top1": top1,
        "top5": top5_hits,
        "decode_pcc_min": min(decode_pcc),
        "decode_pcc_mean": sum(decode_pcc) / len(decode_pcc),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "predicted": predicted[0],
    }
    if batch > 1:
        result["row_top1"] = row_top1
        result["row_top5"] = row_top5
        result["rows_identical"] = all(row == predicted[0] for row in predicted)
        result["row_predicted"] = predicted
    return result


def evaluate_gates(
    identity: dict, prompts: list[dict], corpus: dict | None
) -> list[dict]:
    gates = []
    if identity["kind"] == "reduced":
        for prompt in prompts:
            for index, row in enumerate(prompt["rows"]):
                gates.append({
                    "gate": f"{prompt['name']}[{index}] pcc>={REDUCED_ROW_PCC} and top1 equal",
                    "passed": row["pcc"] >= REDUCED_ROW_PCC and row["top1_equal"],
                    "observed": {"pcc": row["pcc"], "top1_equal": row["top1_equal"]},
                })
        return gates
    for prompt in prompts:
        row = prompt["rows"][0]
        gates.append({
            "gate": f"{prompt['name']} prefill pcc>={FULL_PREFILL_PCC}",
            "passed": row["pcc"] >= FULL_PREFILL_PCC,
            "observed": row["pcc"],
        })
    if corpus is None:
        gates.append({
            "gate": "teacher-forced corpus present",
            "passed": False,
            "observed": None,
        })
        return gates
    gates.append({
        "gate": f"corpus prefill pcc>={FULL_PREFILL_PCC}",
        "passed": corpus["prefill"]["pcc"] >= FULL_PREFILL_PCC,
        "observed": corpus["prefill"]["pcc"],
    })
    gates.append({
        "gate": f"teacher-forced top1>={FULL_TOP1}",
        "passed": corpus["top1"] >= FULL_TOP1,
        "observed": corpus["top1"],
    })
    gates.append({
        "gate": f"teacher-forced top5>={FULL_TOP5}",
        "passed": corpus["top5"] >= FULL_TOP5,
        "observed": corpus["top5"],
    })
    return gates


def generate(
    session: Session, ids: list[int], max_tokens: int, stop_tokens: set[int]
) -> dict:
    """Greedy completion for every batch row (identical prompts per row).

    Rows decode together through the admitted batched path. Generation stops when
    every row has emitted a stop token or `max_tokens` is reached, so with
    identical prompts the rows stop together and their ids must match.
    """
    session.reset()
    batch = session.batch
    started = time.time()
    tokens = [int(token) for token in session.prefill_rows([ids] * batch).argmax(-1)]
    first_token_seconds = time.time() - started
    outputs = [[token] for token in tokens]
    # The first decode step captures the decode trace, so time from the second
    # step: otherwise the one-time capture dominates a short generation (P38).
    started = None
    while len(outputs[0]) < max_tokens and any(
        output[-1] not in stop_tokens for output in outputs
    ):
        position = len(ids) + len(outputs[0]) - 1
        tokens = session.decode_greedy(
            [output[-1] for output in outputs], [position] * batch
        )
        for row, token in enumerate(tokens):
            outputs[row].append(token)
        if started is None:
            started = time.time()
    decode_seconds = time.time() - (started if started is not None else time.time())
    output = outputs[0]
    result = {
        "prompt_tokens": len(ids),
        "output_ids": output,
        "stopped_at_eos": output[-1] in stop_tokens,
        "first_token_seconds": first_token_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (len(output) - 1) / decode_seconds
        if len(output) > 1
        else None,
        "batch": batch,
        "rows_identical": all(row == output for row in outputs),
    }
    if batch > 1:
        result["row_output_ids"] = outputs
        result["decode_tokens_per_second_aggregate"] = (
            ((len(output) - 1) * batch) / decode_seconds if len(output) > 1 else None
        )
    return result


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (p90 of 5 samples is the maximum)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summary(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def cache_probe(session: Session, ids: list[int], block_size: int, phase: str) -> dict:
    """Read each row's paged K and V back at a decode position.

    Diagnostic for the batch-8 defect: it reports, per row, the physical block the
    page table selects, the slot within that block, and the first two values at that
    slot for every kv head of layer 0's K and V caches. It reads state back and
    writes nothing, so it cannot change a measured path.
    """
    batch = session.batch
    session.reset()
    tokens = session.prefill_rows([ids] * batch).argmax(-1).tolist()
    if phase == "decode":
        session.decode_greedy(tokens, [len(ids)] * batch)
    pair = session.model.layers[0].attention.layer_past

    def read(tensor):
        host = session.ttnn.to_torch(session.ttnn.get_device_tensors(tensor)[0])
        if host.dim() == 5:
            host = host[0]
        return host.float()

    k, v = read(pair[0]), read(pair[1])
    rows = []
    for row in range(batch):
        position = len(ids)
        physical = int(session.page_table[row][position // block_size])
        site = position % block_size
        rows.append({
            "row": row,
            "position": position,
            "physical_block": physical,
            "site": site,
            "k_heads": [
                [round(x, 4) for x in k[physical, head, site, :2].tolist()]
                for head in range(k.shape[1])
            ],
            "v_heads": [
                [round(x, 4) for x in v[physical, head, site, :2].tolist()]
                for head in range(v.shape[1])
            ],
        })
    return {"shape": list(k.shape), "phase": phase, "rows": rows}


def bench(session: Session, ids: list[int], corpus: list[int], args) -> dict:
    """Synchronized host-wall timing; each decode returns host greedy tokens."""
    runs = []
    paired = args.prefill_modes == "both"
    for run in range(args.warmup + args.runs):
        prefill_mode = None
        if paired:
            # P83: alternate the two prefill paths inside one session, so the
            # comparison is paired instead of spanning sessions whose medians
            # differ by more than the effect.
            prefill_mode = "batched" if run % 2 == 0 else "sequential"
            session.batched_prefill = prefill_mode == "batched"
            session.args.disable_batched_prefill = prefill_mode == "sequential"
        session.reset()
        batch = session.batch
        started = time.perf_counter()
        tokens = session.prefill_rows([ids] * batch).argmax(-1).tolist()
        first_token = time.perf_counter() - started
        outputs = [[token] for token in tokens]
        started = time.perf_counter()
        for step in range(args.decode_tokens - 1):
            tokens = session.decode_greedy(tokens, [len(ids) + step] * batch)
            for row, token in enumerate(tokens):
                outputs[row].append(token)
        decode_seconds = time.perf_counter() - started
        per_user = (args.decode_tokens - 1) / decode_seconds
        runs.append({
            "prefill_mode": prefill_mode,
            "warmup": run < args.warmup,
            "first_token_seconds": first_token,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": per_user,
            "aggregate_decode_tokens_per_second": per_user * batch,
            "output_ids": outputs[0],
            "rows_identical": all(output == outputs[0] for output in outputs),
            "row_output_ids": outputs if batch > 1 else None,
        })
        print(
            f"[bench] run {run}: first token {first_token:.3f}s, decode {per_user:.2f} tok/s/user, {per_user * batch:.2f} tok/s aggregate (batch {batch})",
            flush=True,
        )
    measured = [run for run in runs if not run["warmup"]]
    # P100/P101: screen measured runs for host contention before summarizing. The
    # rule lives in contention.py so the device-free suite can test it against real
    # recorded numbers; a contended run is excluded from the summary, not silently
    # averaged into it.
    flags = flag_runs(measured)
    if flags["decode_contended"]:
        print(
            f"[bench] excluded {flags['decode_contended']} decode-contended run(s); "
            f"admitted {len(admitted(measured))} of {len(measured)}",
            flush=True,
        )
    measured = admitted(measured)
    by_mode = {}
    if paired:
        for mode in ("batched", "sequential"):
            group = [run for run in measured if run["prefill_mode"] == mode]
            if group:
                by_mode[mode] = {
                    "runs": len(group),
                    "first_token_seconds": summary([
                        run["first_token_seconds"] for run in group
                    ]),
                    "decode_tokens_per_second": summary([
                        run["decode_tokens_per_second"] for run in group
                    ]),
                    "outputs_identical_within_mode": all(
                        run["output_ids"] == group[0]["output_ids"] for run in group
                    ),
                }
    prefill = []
    # Batch 1 measures each prefill length. Batch > 1 measures only the first length
    # that fits the per-row KV budget, because each pass is batch * length tokens;
    # the throughput is reported over all rows of that pass.
    if session.batch == 1:
        lengths = list(args.prefill_lengths)
    else:
        lengths = [
            length for length in args.prefill_lengths[:1] if length < session.row_tokens
        ]
    for length in lengths:
        if args.prefill_runs <= 0:
            # P99: an explicit --prefill-runs 0 means "skip the sweep". It used to
            # crash in statistics.median on an empty sample list.
            break
        samples = []
        for run in range(1 + args.prefill_runs):
            session.reset()
            started = time.perf_counter()
            if session.batch == 1:
                session.prefill(corpus[:length])
            else:
                session.prefill_rows([corpus[:length]] * session.batch)
            elapsed = time.perf_counter() - started
            if run:
                samples.append(length * session.batch / elapsed)
        entry = {"input_tokens": length, "tokens_per_second": summary(samples)}
        if session.batch > 1:
            entry["rows"] = session.batch
        prefill.append(entry)
        print(
            f"[bench] prefill {length} x {session.batch} row(s): median {statistics.median(samples):.1f} tok/s",
            flush=True,
        )
    return {
        "prompt_tokens": len(ids),
        "decode_tokens": args.decode_tokens,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "first_token_seconds": summary([
            run["first_token_seconds"] for run in measured
        ]),
        "decode_tokens_per_second": summary([
            run["decode_tokens_per_second"] for run in measured
        ]),
        "aggregate_decode_tokens_per_second": summary([
            run["aggregate_decode_tokens_per_second"] for run in measured
        ]),
        "batch": session.batch,
        "outputs_identical_across_runs": all(
            run["output_ids"] == runs[0]["output_ids"] for run in runs
        ),
        "first_token_seconds_by_mode": by_mode or None,
        "prefill_modes": args.prefill_modes,
        "rows_identical_in_every_run": all(run["rows_identical"] for run in runs),
        "contention_flags": flags,
        "admitted_runs": len(measured),
        "output_ids": runs[0]["output_ids"],
        "runs": runs,
        "prefill": prefill,
    }


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    partial.rename(path)
    print(f"[receipt] {path}", flush=True)


def per_row_tokens(args) -> int:
    """Positions available per batch row in the paged KV pool."""
    return min(args.max_seq_len, args.page_block * (args.page_blocks // args.batch))


def prefix_identity(identity: dict, args) -> dict:
    """Exact identity keys for this process's prefix entries.

    The packaged wrappers export the package/runtime identities; a developer
    invocation without them cannot claim a stable identity, so it fails instead
    of merging two different trees under a placeholder.
    """
    package = os.environ.get("NANBEIGE43_PACKAGE_IDENTITY")
    runtime = os.environ.get("NANBEIGE43_RUNTIME_IDENTITY")
    if not package or not runtime:
        raise SystemExit(
            "prefix-cache mode needs NANBEIGE43_PACKAGE_IDENTITY and "
            "NANBEIGE43_RUNTIME_IDENTITY (run the packaged nanbeige43 binary)"
        )
    return {
        "checkpoint_digest": identity["files_sha256_digest"],
        "checkpoint_revision": identity["revision"],
        "package_identity": package,
        "runtime_identity": runtime,
        "preset": args.preset,
        "mesh": args.mesh,
        # Everything that changes what a cached block means, short of the tokens.
        "execution": (
            f"prompt_mode={args.prompt_mode};page_block={args.page_block};"
            f"page_blocks={args.page_blocks};trace={args.trace}"
        ),
    }


def resume_alignment(padded_suffix: int) -> int:
    """Upstream's resume-offset alignment for one padded suffix length.

    `get_attn_sdpa_prefill_program_config` pins q_chunk 64 below 2048 tokens and
    256 from 2048 (models/tt_transformers/tt/model_config.py:1563-1603, reached
    with `chunk_start_idx=0` for the traced config), and
    `Generator._resume_offset_alignment` takes its lcm with the paged block size
    (models/tt_transformers/tt/generator.py:1491-1513).
    """
    q_chunk = 64 if padded_suffix < 2048 else 256
    return math.lcm(prefix_cache.BLOCK_TOKENS, q_chunk)


def effective_resume_tokens(prompt_tokens: int, prefix_tokens: int) -> dict:
    """Mirror `Generator._align_resume_offsets` for one user, on the host.

    Upstream floors a cached offset to `lcm(page block, traced q_chunk)`; doing
    that arithmetic here lets the runner reject a prefix that would silently be
    recomputed instead of discovering it after a device run.
    """
    from models.tt_transformers.tt.common import (
        get_all_padded_prefill_lengths,
        get_padded_prefill_len,
    )

    floored = prefix_cache.aligned_tokens(prefix_tokens)
    alignment = None
    padded_suffix = None
    for _ in range(len(get_all_padded_prefill_lengths(prompt_tokens)) + 1):
        padded_suffix = get_padded_prefill_len(prompt_tokens - floored)
        alignment = resume_alignment(padded_suffix)
        settled = (prefix_tokens // alignment) * alignment
        if settled == floored:
            return {
                "prefix_tokens": floored,
                "requested_tokens": prefix_tokens,
                "alignment": alignment,
                "padded_suffix": padded_suffix,
                "floored": floored != prefix_tokens,
            }
        floored = settled
    raise SystemExit(
        f"resume offset alignment did not settle for start_pos={prefix_tokens}, "
        f"prompt={prompt_tokens}; last alignment {alignment}"
    )


def prefix_preflight(args) -> dict:
    """Reject an impossible prefix-cache request before the mesh is opened."""
    if args.batch not in PREFIX_CACHE_BATCHES:
        raise SystemExit(
            f"prefix-cache mode admits batch {list(PREFIX_CACHE_BATCHES)} only "
            f"(upstream's sequential prefix path); got {args.batch}"
        )
    if args.batched_prefill:
        raise SystemExit(
            "prefix-cache mode requires the sequential per-row prefill path; "
            "--batched-prefill zeroes the cached offset"
        )
    if args.prompt_mode != "prefill":
        raise SystemExit(
            "prefix-cache mode needs the prefill path; --prompt-mode decode has no "
            "cached-prefix ingestion"
        )
    if args.page_block != prefix_cache.BLOCK_TOKENS:
        raise SystemExit(
            f"prefix-cache mode uses {prefix_cache.BLOCK_TOKENS}-token blocks; "
            f"--page-block {args.page_block} would misalign every entry"
        )
    if args.prefix_tokens % prefix_cache.BLOCK_TOKENS:
        raise SystemExit(
            f"--prefix-tokens {args.prefix_tokens} must be a multiple of "
            f"{prefix_cache.BLOCK_TOKENS}"
        )
    if args.prefix_tokens <= 0:
        raise SystemExit("--prefix-tokens must be positive")
    if args.prefix_decode_tokens < 1:
        raise SystemExit("--prefix-decode-tokens must be at least 1")
    if args.prefix_tokens >= args.prefix_prompt_length:
        raise SystemExit(
            f"--prefix-tokens {args.prefix_tokens} must be shorter than the "
            f"--prefix-prompt-length {args.prefix_prompt_length}-token prompt"
        )
    effective = effective_resume_tokens(args.prefix_prompt_length, args.prefix_tokens)
    if effective["prefix_tokens"] != args.prefix_tokens:
        raise SystemExit(
            f"--prefix-tokens {args.prefix_tokens} would be floored to "
            f"{effective['prefix_tokens']} by upstream resume alignment "
            f"{effective['alignment']} at padded suffix {effective['padded_suffix']}; "
            f"choose a prefix that is a multiple of the alignment"
        )
    blocks = args.prefix_tokens // prefix_cache.BLOCK_TOKENS
    if args.prefix_cache_blocks is None:
        args.prefix_cache_blocks = blocks
    if args.prefix_cache_blocks < blocks:
        raise SystemExit(
            f"--prefix-cache-blocks {args.prefix_cache_blocks} cannot hold the "
            f"{blocks}-block prefix"
        )
    capacity = per_row_tokens(args)
    required = args.prefix_prompt_length + args.prefix_decode_tokens - 1
    if required >= capacity:
        raise SystemExit(
            f"prefix-cache run needs {required} positions per row (prompt "
            f"{args.prefix_prompt_length} plus {args.prefix_decode_tokens - 1} decode "
            f"steps) but the per-row KV budget is {capacity}"
        )
    return {
        "batch": args.batch,
        "block_tokens": prefix_cache.BLOCK_TOKENS,
        "prefix_tokens": args.prefix_tokens,
        "prefix_blocks": blocks,
        "prefix_cache_blocks": args.prefix_cache_blocks,
        "prompt_tokens": args.prefix_prompt_length,
        "decode_tokens": args.prefix_decode_tokens,
        "row_tokens": capacity,
        "resume_alignment": effective,
        "prefill_path": "sequential-per-row start_pos",
        "kv_check": bool(getattr(args, "prefix_kv_check", False)),
    }


def prefix_phase(
    session: Session, ids: list[int], decode_tokens: int, start_pos: list[int] | None
) -> tuple[dict, torch.Tensor]:
    """One prefill + greedy decode pass; timings are synchronized host wall clock.

    Returns the receipt record plus the host prefill logits [batch, vocab], which
    the caller keeps for cold/warm numerical comparison and then drops.
    """
    batch = session.batch
    timings = {}
    started = time.perf_counter()
    logits = session.prefill_rows([ids] * batch, start_pos=start_pos)
    timings["prefill_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    tokens = [int(token) for token in logits.argmax(-1)]
    timings["first_token_seconds"] = timings["prefill_seconds"] + (
        time.perf_counter() - started
    )
    outputs = [[token] for token in tokens]
    started = time.perf_counter()
    for step in range(decode_tokens - 1):
        position = len(ids) + step
        tokens = session.decode_greedy(
            [output[-1] for output in outputs], [position] * batch
        )
        for row, token in enumerate(tokens):
            outputs[row].append(int(token))
    timings["decode_seconds"] = time.perf_counter() - started
    timings["decode_tokens_per_second"] = (
        (decode_tokens - 1) / timings["decode_seconds"]
        if decode_tokens > 1 and timings["decode_seconds"] > 0
        else None
    )
    rows_identical = all(row == outputs[0] for row in outputs)
    return (
        {
            **timings,
            "start_pos": start_pos,
            "output_ids": outputs[0],
            "output_tokens": len(outputs[0]),
            "rows_identical": rows_identical,
            "row_output_ids": outputs if batch > 1 else None,
        },
        logits,
    )


def logits_parity(cold: torch.Tensor, warm: torch.Tensor) -> dict:
    """Compare two [batch, vocab] host logit tensors row by row."""
    with single_threaded_host_math():
        rows = []
        for row in range(cold.shape[0]):
            diff = (cold[row] - warm[row]).abs()
            rows.append({
                "row": row,
                "pcc": pcc(cold[row], warm[row]),
                "max_abs_diff": float(diff.max()),
                "top1_equal": int(cold[row].argmax()) == int(warm[row].argmax()),
            })
        summary = dict(min(rows, key=lambda entry: entry["pcc"]))
        summary["top1_equal"] = all(entry["top1_equal"] for entry in rows)
        summary["bit_identical"] = bool(torch.equal(cold, warm))
        summary["rows"] = rows if len(rows) > 1 else None
        return summary


def kv_page_digests(
    session: Session, prefix_blocks: int, prompt_blocks: int
) -> list[dict]:
    """SHA-256 digests of every layer's K/V pages, split cached prefix vs suffix.

    The physical pages come from the session's own page table, so a cached block
    and its cold copy are the same addresses. Digests are exact equality evidence
    with a constant-size receipt; `--prefix-kv-check` is opt-in because the whole
    per-layer cache is copied to the host.
    """
    import hashlib

    digests = []
    for index, layer in enumerate(session.model.layers):
        hosts = {}
        for name, tensor in zip(("k", "v"), layer.attention.layer_past):
            host = session.ttnn.to_torch(session.ttnn.get_device_tensors(tensor)[0])
            if host.dim() == 5:
                host = host[0]
            hosts[name] = host.float()
        rows = []
        for row in range(session.batch):
            pages = [int(page) for page in session.page_table[row][:prompt_blocks]]
            entry = {}
            for name, host in hosts.items():
                for region, selection in (
                    ("prefix", pages[:prefix_blocks]),
                    ("suffix", pages[prefix_blocks:]),
                ):
                    payload = host[selection].contiguous().numpy().tobytes()
                    entry[f"{name}_{region}"] = hashlib.sha256(payload).hexdigest()
            rows.append(entry)
        digests.append({"layer": index, "rows": rows})
    return digests


def kv_digest_parity(cold: list[dict], warm: list[dict]) -> dict:
    """Compare per-layer page digests; the cached prefix must never change."""
    mismatches = []
    for cold_layer, warm_layer in zip(cold, warm):
        for row, (cold_row, warm_row) in enumerate(
            zip(cold_layer["rows"], warm_layer["rows"])
        ):
            for region, digest in cold_row.items():
                if digest != warm_row[region]:
                    mismatches.append({
                        "layer": cold_layer["layer"],
                        "row": row,
                        "region": region,
                    })
    prefix_mismatches = [
        item for item in mismatches if item["region"].endswith("prefix")
    ]
    suffix_layers = sorted({
        item["layer"] for item in mismatches if item["region"].endswith("suffix")
    })
    return {
        "layers": len(cold),
        "cached_prefix_pages_identical": not prefix_mismatches,
        "prefix_mismatches": prefix_mismatches,
        "suffix_pages_identical": not suffix_layers,
        "suffix_mismatch_layers": suffix_layers,
        "first_suffix_mismatch_layer": suffix_layers[0] if suffix_layers else None,
        "mismatch_count": len(mismatches),
    }


def prefix_cache_pair(
    session: Session, ids: list[int], args, identity: dict
) -> tuple[dict, list[dict]]:
    """Cold/warm pair on one prompt in one session, with block accounting.

    Cold: reset, full prefill, greedy decode N, then record the prefix entry.
    Warm: keep that KV and page table, prefill the full prompt with the cached
    prefix offset, greedy decode N, and compare ids. No reset happens between the
    passes, so the warm pass reuses the cold K/V for the cached prefix.
    """
    batch = session.batch
    prompt_tokens = len(ids)
    prefix_tokens = args.prefix_tokens
    decode_tokens = args.prefix_decode_tokens
    identity_keys = prefix_identity(identity, args)
    digest = prefix_cache.cache_identity(identity_keys)
    cache = prefix_cache.PrefixCache(args.prefix_cache_blocks)

    session.reset()
    cold, cold_logits = prefix_phase(session, ids, decode_tokens, None)
    cold_accounting = prefix_cache.request_accounting(
        prompt_tokens, 0, cold["output_tokens"]
    )
    session_blocks = prompt_tokens // prefix_cache.BLOCK_TOKENS
    kv_check = bool(getattr(args, "prefix_kv_check", False))
    prefix_blocks = prefix_tokens // prefix_cache.BLOCK_TOKENS
    cold_kv = (
        kv_page_digests(session, prefix_blocks, session_blocks) if kv_check else None
    )
    entry = cache.insert(digest, ids, prefix_tokens)

    lease = cache.retain(digest, ids)
    if not lease.hit:
        raise SystemExit(
            f"cold pass recorded prefix {prefix_tokens} but the warm lookup missed"
        )
    if lease.cached_tokens != prefix_tokens:
        raise SystemExit(
            f"warm lookup resolved {lease.cached_tokens} cached tokens, not the "
            f"recorded {prefix_tokens}"
        )
    for row in range(batch):
        cache.assert_writable(lease, prompt_tokens)
    warm, warm_logits = prefix_phase(
        session, ids, decode_tokens, [lease.cached_tokens] * batch
    )
    warm_kv = (
        kv_page_digests(session, prefix_blocks, session_blocks) if kv_check else None
    )
    cache.release(lease)
    warm_accounting = prefix_cache.request_accounting(
        prompt_tokens, lease.cached_tokens, warm["output_tokens"]
    )

    prefill_logits = logits_parity(cold_logits, warm_logits)
    del cold_logits, warm_logits
    kv = kv_digest_parity(cold_kv, warm_kv) if cold_kv is not None else None
    del cold_kv, warm_kv
    cold_ids, warm_ids = cold["output_ids"], warm["output_ids"]
    first_divergence = next(
        (index for index, (a, b) in enumerate(zip(cold_ids, warm_ids)) if a != b),
        None,
    )
    token_parity = cold_ids == warm_ids
    print(
        f"[prefix-cache] prefix {prefix_tokens} tokens (hit {lease.hit}, "
        f"generation {entry.generation}): cold prefill {cold['prefill_seconds']:.3f}s "
        f"warm prefill {warm['prefill_seconds']:.3f}s, cold first token "
        f"{cold['first_token_seconds']:.3f}s warm {warm['first_token_seconds']:.3f}s, "
        f"ids {'equal' if token_parity else f'diverge at {first_divergence}'}",
        flush=True,
    )
    result = {
        "identity": identity_keys,
        "identity_digest": digest,
        "block_tokens": prefix_cache.BLOCK_TOKENS,
        "budget_blocks": args.prefix_cache_blocks,
        "resident_blocks": cache.resident_blocks,
        "requested_prefix_tokens": prefix_tokens,
        "effective_prefix_tokens": lease.cached_tokens,
        "prefix_blocks": prefix_tokens // prefix_cache.BLOCK_TOKENS,
        "prompt_tokens": prompt_tokens,
        "batch": batch,
        "entry": entry.as_dict(),
        "hit": lease.hit,
        "entry_generation": entry.generation,
        "evicted_blocks": cache.evicted_blocks,
        "copy_tail_block": lease.copy_tail_block,
        "write_start_block": lease.write_start_block,
        "accounting": {
            "per_request": {
                "cold": cold_accounting.as_dict(),
                "warm": warm_accounting.as_dict(),
            },
            "batch_rows": batch,
            "warm_cached_input_tokens_total": warm_accounting.cached_input_tokens
            * batch,
            "warm_new_input_tokens_total": warm_accounting.new_input_tokens * batch,
            "warm_output_tokens_total": warm_accounting.output_tokens * batch,
        },
        "cold": cold,
        "warm": warm,
        "token_parity": token_parity,
        "first_divergence": first_divergence,
        "prefill_logits": prefill_logits,
        "kv_parity": kv,
        "scope": {
            "cache": "process-local, one entry, one session",
            "sharing": "no cross-process or cross-tenant sharing is claimed",
            "decode_rate_note": (
                "decode rate is reported per pass; the cold pass captures the "
                "decode trace, so it is not a cache-speed claim"
            ),
            "parity": (
                "greedy ids are recorded with their first divergence; the gate is "
                "prefill-logit agreement at the runner's 0.99 row floor because the "
                "warm suffix is recomputed under a different prefill shape than the "
                "cold full prompt"
            ),
        },
    }
    gates = [
        {
            # The warm suffix is recomputed by a different prefill shape (padded
            # suffix vs padded full prompt), so bfp8 logits are not bit-identical
            # even when the cached K/V is. The id comparison is recorded, and the
            # declared equivalence is the runner's existing row-level PCC floor
            # plus top-1 agreement.
            "gate": f"cold/warm prefill logits pcc>={REDUCED_ROW_PCC} and top1 equal",
            "passed": prefill_logits["pcc"] >= REDUCED_ROW_PCC
            and prefill_logits["top1_equal"],
            "observed": {
                **prefill_logits,
                "greedy_ids_equal": token_parity,
                "first_divergence": first_divergence,
                "cold_tokens": len(cold_ids),
                "warm_tokens": len(warm_ids),
            },
        },
        {
            "gate": f"prefix-cache hit for the {prefix_tokens}-token prefix",
            "passed": lease.hit and lease.cached_tokens == prefix_tokens,
            "observed": lease.as_dict(),
        },
        {
            "gate": "every batch row identical in both passes",
            "passed": cold["rows_identical"] and warm["rows_identical"],
            "observed": {
                "cold": cold["rows_identical"],
                "warm": warm["rows_identical"],
            },
        },
    ]
    if kv is not None:
        gates.append({
            "gate": "cold/warm K/V identical for the cached prefix pages in every layer",
            "passed": kv["cached_prefix_pages_identical"],
            "observed": kv,
        })
    return result, gates


def preflight_compare(args, reference_path: Path) -> None:
    """Fail before opening the mesh when the reference cannot fit the pool.

    The requirement is arithmetic over the reference's own lengths, so an
    impossible request does not need an 80 s weight load to be rejected.
    """
    capacity = per_row_tokens(args)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    corpus = reference.get("teacher_forced") or {}
    # teacher_forced() prefills `split` tokens and decodes TEACHER_FORCED_POSITIONS
    # - 1 further positions, so the corpus needs split + TEACHER_FORCED_POSITIONS - 1
    # positions per row.
    required = (
        int(corpus["split"]) + TEACHER_FORCED_POSITIONS - 1 if "split" in corpus else 0
    )
    steps = int(reference.get("continuation_steps") or 0)
    for prompt in reference.get("prompts") or []:
        required = max(required, len(prompt.get("prompt_ids") or []) + steps)
    if required > capacity:
        blocks = args.batch * -(-required // args.page_block)
        raise SystemExit(
            f"reference needs {required} positions per row (teacher-forced split"
            f" {corpus.get('split')} plus {TEACHER_FORCED_POSITIONS - 1} steps) but the"
            f" per-row KV budget is {capacity}; raise --page-blocks to at least {blocks}"
        )


def run_device(
    args, spec, identity: dict, reference: dict | None, receipt: dict
) -> None:
    if args.mode == "compare" and args.reference is not None:
        # Cheap and device-free: an impossible pool size is rejected before an
        # 80 s weight load and a partially executed run.
        preflight_compare(args, args.reference)
    if args.mode == "prefix-cache":
        # Also device-free: batch, block size, prefix alignment and pool capacity are
        # arithmetic, so a rejected request must not cost a weight load either.
        receipt["prefix_cache_preflight"] = prefix_preflight(args)
    with open_mesh(spec, TRACE_REGION_BYTES if args.trace else 0) as (
        mesh_device,
        mesh_record,
    ):
        receipt["mesh"] = mesh_record
        session = Session(
            mesh_device,
            args.preset,
            args.max_seq_len,
            args.page_block,
            args.page_blocks,
            args.prompt_mode,
            enable_trace=args.trace,
            batch=args.batch,
            batched_prefill=args.batched_prefill,
            device_argmax=args.device_argmax,
        )
        receipt["l1"] = (
            session.l1_log
        )  # same list: snapshots taken later still reach a failure receipt
        try:
            run_session(args, session, identity, reference, receipt)
        finally:
            session.close()


def run_session(
    args, session: Session, identity: dict, reference: dict | None, receipt: dict
) -> None:
    receipt["load_seconds"] = session.load_seconds
    # After create_tt_model returned and the runner dropped the host state dict.
    receipt["host_rss_after_load"] = host_rss()
    receipt["device_name"] = session.args.device_name
    receipt["model_name"] = session.args.model_name
    receipt["n_layers"] = session.args.n_layers
    receipt["on_device_sampling"] = bool(
        getattr(session.model, "_supports_on_device_sampling", False)
    )
    # Requested and effective prefill path: upstream falls back to sequential rows
    # when the prompts differ in length, a prefix is cached, or the trace is on.
    receipt["batched_prefill"] = {
        "requested": bool(getattr(args, "batched_prefill", False)),
        "effective": session.batched_prefill,
        "upstream_disable_batched_prefill": bool(
            getattr(session.args, "disable_batched_prefill", False)
        ),
    }
    print(
        f"[load] {session.args.model_name} on {session.args.device_name}: {session.load_seconds:.1f}s",
        flush=True,
    )
    if args.mode == "compare":
        receipt["prompts"] = compare_prompts(session, reference)
        corpus = None
        if identity["kind"] == "full" and reference.get("teacher_forced"):
            corpus = teacher_forced(session, reference["teacher_forced"])
        receipt["teacher_forced"] = corpus
        receipt["gates"] = evaluate_gates(identity, receipt["prompts"], corpus)
    elif args.mode == "bench":
        tokenizer = session.args.tokenizer
        ids = chat_ids(
            tokenizer, [{"role": "user", "content": args.message}], args.enable_thinking
        )
        result = bench(
            session, ids, corpus_tokens(tokenizer, max(args.prefill_lengths)), args
        )
        result["message"] = args.message
        result["enable_thinking"] = args.enable_thinking
        result["text"] = tokenizer.decode(result["output_ids"])
        receipt["bench"] = result
        if args.cache_probe:
            receipt["cache_probe"] = cache_probe(
                session, ids, args.page_block, args.cache_probe
            )
        # Predeclared bench gates: deterministic across runs and rows, and (when given)
        # identical to the token ids of an earlier accepted bench receipt.
        paired = result["prefill_modes"] == "both"
        receipt["gates"] = [
            {
                "gate": "bench outputs identical across all runs"
                if not paired
                else "bench outputs identical within each prefill mode",
                "passed": result["rows_identical_in_every_run"]
                and (
                    all(
                        group["outputs_identical_within_mode"]
                        for group in (
                            result["first_token_seconds_by_mode"] or {}
                        ).values()
                    )
                    if paired
                    else result["outputs_identical_across_runs"]
                ),
                "observed": {
                    "across_runs": result["outputs_identical_across_runs"],
                    "by_mode": {
                        mode: group["outputs_identical_within_mode"]
                        for mode, group in (
                            result["first_token_seconds_by_mode"] or {}
                        ).items()
                    },
                }
                if paired
                else result["outputs_identical_across_runs"],
            },
            {
                "gate": "every batch row identical in every run",
                "passed": result["rows_identical_in_every_run"],
                "observed": result["rows_identical_in_every_run"],
            },
        ]
        if args.expect_output_from:
            expected = json.loads(args.expect_output_from.read_text())["bench"][
                "output_ids"
            ]
            actual = result["output_ids"]
            first = next(
                (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None
            )
            observed = {
                "expected_tokens": len(expected),
                "observed_tokens": len(actual),
                "first_divergence": first,
                "prefix_equal": expected[: len(actual)] == actual[: len(expected)],
            }
            if first is None and len(expected) != len(actual):
                # P90: a 32-token reference against a 128-token run reported
                # first_divergence None while failing, which reads as a contradiction.
                observed["note"] = (
                    "length mismatch; no token diverges over the shorter sequence"
                )
            receipt["gates"].append({
                "gate": f"output ids equal {args.expect_output_from.name}",
                "passed": expected == actual,
                "observed": observed,
            })
    elif args.mode == "prefix-cache":
        tokenizer = session.args.tokenizer
        # The corpus is a long, fixed public-domain prompt, so the requested 32-token
        # prefix fits and the pair is reproducible; a chat message is typically under
        # the 64-token resume alignment.
        ids = corpus_tokens(tokenizer, args.prefix_prompt_length)
        result, gates = prefix_cache_pair(session, ids, args, identity)
        receipt["prefix_cache"] = result
        receipt["gates"] = gates
    else:
        tokenizer = session.args.tokenizer
        messages = [{"role": "user", "content": args.message}]
        ids = chat_ids(tokenizer, messages, args.enable_thinking)
        stop_tokens = set(tokenizer.stop_tokens)
        runs = [
            generate(session, ids, args.max_tokens, stop_tokens)
            for _ in range(args.generate_runs)
        ]
        result = runs[0]
        # Per-run rates, not the run dicts: `result` is `runs[0]`, so nesting the
        # list would make the receipt reference itself and fail to serialise.
        result["run_count"] = len(runs)
        # P98: run 1 can carry first-use JIT compilation for a kernel set the cache
        # has not seen, which inflated a batch-16 generate receipt to 13.371 s against
        # a 1.2-1.6 s bench median in the same session. Record every run's first
        # token and admit the median, so a cold run is visible instead of silent.
        result["run_first_token_seconds"] = [run["first_token_seconds"] for run in runs]
        first_tokens = [run["first_token_seconds"] for run in runs]
        if first_tokens:
            result["first_token_seconds"] = summary(first_tokens)["median"]
            result["first_token_samples"] = first_tokens
        result["run_decode_tokens_per_second"] = [
            run["decode_tokens_per_second"] for run in runs
        ]
        result["runs_identical"] = all(
            run["output_ids"] == result["output_ids"] for run in runs
        )
        rates = [
            run["decode_tokens_per_second"]
            for run in runs
            if run["decode_tokens_per_second"] is not None
        ]
        if rates:
            # P76 measured an 18.5 % swing between consecutive identical generate
            # runs, so the median of `--generate-runs` is the reproducible figure.
            result["decode_tokens_per_second_median"] = statistics.median(rates)
        result["messages"] = messages
        result["enable_thinking"] = args.enable_thinking
        result["text"] = tokenizer.decode([
            t for t in result["output_ids"] if t not in tokenizer.stop_tokens
        ])
        receipt["generation"] = result
        print(f"[generate] {result['text']!r}", flush=True)
        receipt["gates"] = [
            {
                "gate": "generation produced tokens",
                "passed": len(result["output_ids"]) > 1,
                "observed": len(result["output_ids"]),
            },
            {
                "gate": "every batch row identical",
                "passed": result["rows_identical"],
                "observed": result["rows_identical"],
            },
            {
                "gate": "every run produced identical ids",
                "passed": result["runs_identical"],
                "observed": result["runs_identical"],
            },
        ]
        if args.expect_output_from:
            expected = json.loads(args.expect_output_from.read_text())["generation"][
                "output_ids"
            ]
            first = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(expected, result["output_ids"]))
                    if a != b
                ),
                None,
            )
            receipt["gates"].append({
                "gate": f"output ids equal {args.expect_output_from.name}",
                "passed": expected == result["output_ids"],
                "observed": {
                    "expected_tokens": len(expected),
                    "observed_tokens": len(result["output_ids"]),
                    "first_divergence": first,
                    "prefix_equal": expected[: len(result["output_ids"])]
                    == result["output_ids"][: len(expected)],
                },
            })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("compare", "generate", "bench", "prefix-cache")
    )
    parser.add_argument("--mesh", choices=sorted(MESHES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, help="reference artifact (compare mode)"
    )
    parser.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--page-block", type=int, default=32)
    parser.add_argument(
        "--page-blocks",
        type=int,
        help="paged KV pool size in blocks; default gives every row --max-seq-len positions, "
        "i.e. batch * ceil(max_seq_len / page_block), which is the admitted configuration "
        "(P87). Ids are pool-invariant (32/32 and 8/8 prefixes matched across 1024, 4096 and "
        "8192 blocks), so this affects capacity and latency only",
    )
    parser.add_argument(
        "--message",
        default="Introduce yourself in two sentences.",
        help="user message (generate mode)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="generate mode: keep the template's thinking block",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="generate mode token limit"
    )
    parser.add_argument(
        "--prefill-modes",
        choices=("as-requested", "both"),
        default="as-requested",
        help="bench, batch > 1: 'both' alternates the batched and sequential prefill paths run by run inside one session, so the comparison is paired (P83); ids are then compared within each mode",
    )
    parser.add_argument(
        "--generate-runs",
        type=int,
        default=1,
        help="generate mode: measured runs; the receipt gains a per-run list and a median rate, because P76 measured an 18.5 % swing between consecutive identical runs",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="capture and replay the decode step as a device trace (default; validated bit-exact in O1). --no-trace runs eager decode",
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=ACCEPTED_BATCHES,
        default=1,
        help="user rows with identical inputs (bench, and compare: teacher-forced gates on the worst row); generate uses batch 1",
    )
    parser.add_argument(
        "--batched-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="batch > 1: prefill every row in one pass through upstream's batched path instead of one row at a time (P5; off until admitted)",
    )
    parser.add_argument(
        "--device-argmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bench/generate: argmax the device logits and read back one index per row instead of the gathered vocabulary (P8; default). --no-device-argmax keeps the host argmax",
    )
    parser.add_argument(
        "--cache-probe",
        choices=("prefill", "decode"),
        help="bench: read layer 0's K and V back after prefill only, or after one decode step, and record per-row per-head values (diagnostic for the batch-8 defect; reads only)",
    )
    parser.add_argument(
        "--expect-output-from",
        type=Path,
        help="bench/generate: receipt whose output ids must be reproduced exactly",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=128,
        help="bench mode: greedy tokens per run (EOS ignored)",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=128,
        help="prefix-cache mode: cached prefix length; must be a multiple of 32 and of upstream's resume alignment (64 for a padded suffix under 2048, else 256)",
    )
    parser.add_argument(
        "--prefix-prompt-length",
        type=int,
        default=256,
        help="prefix-cache mode: corpus tokens in the shared prompt (must exceed --prefix-tokens)",
    )
    parser.add_argument(
        "--prefix-decode-tokens",
        type=int,
        default=32,
        help="prefix-cache mode: greedy tokens per cold/warm pass",
    )
    parser.add_argument(
        "--prefix-cache-blocks",
        type=int,
        help="prefix-cache mode: block budget for the index (default: the prefix's own block count); a smaller budget than the prefix fails closed",
    )
    parser.add_argument(
        "--prefix-kv-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="prefix-cache mode: after each pass, copy layer 0's K/V for the prompt's pages to the host and require the cached pages to be identical; the gate and per-row diffs land in the receipt",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="bench mode: unmeasured runs"
    )
    parser.add_argument("--runs", type=int, default=5, help="bench mode: measured runs")
    parser.add_argument(
        "--prefill-lengths",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[128, 1024, 4096],
    )
    parser.add_argument(
        "--prefill-runs",
        type=int,
        default=3,
        help="bench mode: measured prefills per length (after one warmup)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("prefill", "decode"),
        default="prefill",
        help="decode: ingest prompts token-at-a-time through the decode path. A single-device mesh cannot run the "
        "full-width (7168) prefill RMSNorm (multichip prefill uses the distributed norm), so the 1x1 smoke uses this.",
    )
    args = parser.parse_args(argv)
    if args.page_blocks is None:
        # P87: the old default of 256 blocks gave 8 blocks per row at batch 32 - 256
        # positions against a 8192 max_seq_len - and made `compare` fail its own
        # preflight. Derive full context instead; the pool is a capacity/latency knob.
        args.page_blocks = args.batch * math.ceil(args.max_seq_len / args.page_block)

    spec = MESHES[args.mesh]
    authority = load_authority()
    # Checkpoint admission precedes every device touch.
    identity = verify_checkpoint(
        args.checkpoint, authority, allow_full=spec.name != "smoke-1x1"
    )
    reference = None
    if args.mode == "compare":
        if args.reference is None:
            raise SystemExit("compare mode requires --reference")
        reference = torch.load(args.reference, weights_only=False)
        if reference.get("schema") != "nanbeige43-reference/v1":
            raise SystemExit("reference artifact schema mismatch")
        if (
            reference["checkpoint"]["files_sha256_digest"]
            != identity["files_sha256_digest"]
        ):
            raise SystemExit("reference was generated from a different checkpoint")

    os.environ["HF_MODEL"] = str(args.checkpoint)
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Upstream MLP padding knob (model_config.py:2821-2833, used by default for the
    # Qwen2.5 32B/72B models): pad the MLP hidden size to a multiple of
    # PAD_MLP_CORES*32*TP. Unpadded, 11008/device gives an 8-core DRAM-sharded decode
    # grid whose FF3 program fills L1 to 1,220,928 B; at decoder layer 2 the
    # fragmented L1 placed the FF1 output at 1,218,304 B and the program threw
    # (receipts/L3b-p150x2-full*.json). With 16, the hidden size pads
    # 22016 -> 22528 with zero rows/columns (exact), and the grid widens to 32 cores.
    os.environ["PAD_MLP_CORES"] = str(MLP_PAD_CORES)
    # Weight cache names carry dtype and layout but not shape, so padding is part of the path.
    os.environ["TT_CACHE_PATH"] = str(
        args.cache_root / args.checkpoint.name / f"mlp-pad-{MLP_PAD_CORES}"
    )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "mode": args.mode,
        "simulated": bool(os.environ.get("TT_METAL_SIMULATOR")),
        "runtime_identity": os.environ.get("NANBEIGE43_RUNTIME_IDENTITY"),
        "package_identity": os.environ.get("NANBEIGE43_PACKAGE_IDENTITY"),
        "checkpoint": identity,
        "reference": {
            "path": str(args.reference),
            "sha256": sha256_file(args.reference),
        }
        if reference
        else None,
        "preset": args.preset,
        "pad_mlp_cores": MLP_PAD_CORES,
        "max_seq_len": args.max_seq_len,
        "paged_kv": {"block_size": args.page_block, "max_num_blocks": args.page_blocks},
        "prompt_mode": args.prompt_mode,
        "decode_trace": args.trace,
        "batch": args.batch,
        "batched_prefill_requested": args.batched_prefill,
        "device_argmax": args.device_argmax,
        "tt_cache_path": os.environ["TT_CACHE_PATH"],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        run_device(args, spec, identity, reference, receipt)
    except (Exception, SystemExit) as error:
        # SystemExit is a BaseException: the runner's own budget and mode guards
        # raise it, and without catching it a failed run left no receipt at all.
        # r[impl tenstorrent.nanbeige43_p150x2.reference_agreement.blocked]
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error)[-4000:],
            "traceback": traceback.format_exc()[-8000:],
        }
        receipt["gates"] = receipt.get("gates") or [
            {
                "gate": "device run completed",
                "passed": False,
                "observed": type(error).__name__,
            }
        ]
        receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        receipt["passed"] = False
        write_receipt(args.receipt, receipt)
        raise
    receipt["host_rss_end"] = host_rss()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    receipt["passed"] = all(gate["passed"] for gate in receipt["gates"])
    write_receipt(args.receipt, receipt)
    failed = [gate["gate"] for gate in receipt["gates"] if not gate["passed"]]
    if failed:
        raise SystemExit(
            f"gates failed: {failed[:5]}{' ...' if len(failed) > 5 else ''}"
        )
    print("[result] all gates passed", flush=True)


if __name__ == "__main__":
    main()
