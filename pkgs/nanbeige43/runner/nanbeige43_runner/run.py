# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.reference_agreement]
"""Run Nanbeige4.3-33B through tt_transformers and compare it with the CPU reference.

`compare` prefills each reference prompt, then decodes the reference
continuation teacher-forced, and compares every logits row with the reference.
With a full checkpoint it also runs the teacher-forced corpus gate.
`generate` runs a free greedy chat completion.
`bench` measures first-token latency, decode tokens/s and prefill tokens/s on a
fixed chat prompt under the O0 protocol.
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

from .authority import load_authority, sha256_file, verify_checkpoint
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
# 16 and 32 under the teacher-forced fallback (O3, P2). Batch 8 corrupts rows 5-7 at every
# page-pool size tried and is refused; its cause is narrowed to the per-row page-block path
# (P14-P16). Batch > 32 needs a tile-row dataflow change (P10/P11).
ACCEPTED_BATCHES = (1, 2, 4, 16, 32)


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

    def prefill_rows(self, rows: list[list[int]]) -> torch.Tensor:
        """Prefill one prompt per batch row; returns last-position logits [batch, vocab]."""
        if len(rows) != self.batch:
            raise ValueError(f"expected {self.batch} rows, got {len(rows)}")
        lengths = [len(ids) for ids in rows]
        if max(lengths) >= self.row_tokens:
            raise SystemExit(
                f"prompt of {max(lengths)} tokens does not fit the per-row KV budget of {self.row_tokens}"
            )
        if self.prompt_mode == "decode":
            if self.batch != 1:
                raise SystemExit("--prompt-mode decode supports batch 1 only")
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
    session.reset()
    started = time.time()
    token = int(session.prefill(ids).argmax())
    first_token_seconds = time.time() - started
    output = [token]
    started = time.time()
    while token not in stop_tokens and len(output) < max_tokens:
        token = session.decode_greedy([token], [len(ids) + len(output) - 1])[0]
        output.append(token)
    decode_seconds = time.time() - started
    return {
        "prompt_tokens": len(ids),
        "output_ids": output,
        "stopped_at_eos": output[-1] in stop_tokens,
        "first_token_seconds": first_token_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (len(output) - 1) / decode_seconds
        if len(output) > 1
        else None,
    }


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


def bench(session: Session, ids: list[int], corpus: list[int], args) -> dict:
    """Synchronized host-wall timing; each decode returns host greedy tokens."""
    runs = []
    for run in range(args.warmup + args.runs):
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
        "rows_identical_in_every_run": all(run["rows_identical"] for run in runs),
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


def run_device(
    args, spec, identity: dict, reference: dict | None, receipt: dict
) -> None:
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
        # Predeclared bench gates: deterministic across runs and rows, and (when given)
        # identical to the token ids of an earlier accepted bench receipt.
        receipt["gates"] = [
            {
                "gate": "bench outputs identical across all runs",
                "passed": result["outputs_identical_across_runs"],
                "observed": result["outputs_identical_across_runs"],
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
                "observed": {"first_divergence": first},
            })
    else:
        tokenizer = session.args.tokenizer
        messages = [{"role": "user", "content": args.message}]
        ids = chat_ids(tokenizer, messages, args.enable_thinking)
        result = generate(session, ids, args.max_tokens, set(tokenizer.stop_tokens))
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
            }
        ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "generate", "bench"))
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
    parser.add_argument("--page-blocks", type=int, default=256)
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
        "--expect-output-from",
        type=Path,
        help="bench mode: receipt whose bench output ids must be reproduced exactly",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=128,
        help="bench mode: greedy tokens per run (EOS ignored)",
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

    if args.batch != 1 and args.mode == "generate":
        raise SystemExit("--batch applies to bench and compare modes only")
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
    except Exception as error:
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
