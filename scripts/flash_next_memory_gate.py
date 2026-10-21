#!/usr/bin/env python3
"""Screen pinned Flash Next GGUF layer placement without downloading model weights.

Inputs are byte-range prefixes of the two first shards and their shared n-gram
shard. This is a *lower bound*, not device admission: GGUF quant kernels, actual
TT DRAM capacity, prompt scratch and trace residency need separate measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
STRATA_REVISION = "3ce2523c2823687de5372be3af58534f56cbf286"
SHARDS = {
    "Q2_0": (
        "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        37_623_740_192,
        "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720",
    ),
    "IQ2_XS": (
        "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
        39_225_954_592,
        "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7",
    ),
    "ngram": (
        "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        28_800_138_432,
        "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
    ),
}
HEADER_SHA256 = {
    "Q2_0": "e94c1089ec83ed0c23a589e1436d567649f06551eb2e447f6f6760e89f6883ca",
    "IQ2_XS": "5c42d97e6ffd1c1633ff30f68602fb65932bfee427bdf39c9ed649626997b009",
    "ngram": "49f992d94de7d687ab0042f54d72fbabb4fec16e6860181f1b5bd91e46c89a96",
}
HEADER_BYTES = {"Q2_0": 11_024_672, "IQ2_XS": 11_024_672, "ngram": 192}
# TT-Metal 0.77 Tensor DataType -> DataFormat -> tile_size; not GGUF encodings.
TTNN_TILE_BYTES = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}
P150_NOMINAL_BYTES = 32_000_000_000
# Only a *provisional* floor: the existing 27B decode-profile.ncl reserves 1 GiB.
TRACE_FLOOR_BYTES = 1 << 30
HOST_OTHER_FLOOR_BYTES = 6_000_000_000  # Provisional non-expert CPU fallback allowance.
# GGML storage blocks used by the pinned checkpoint, not TT BLOCKFP2/4 formats.
# Source: Strata tools/gguf_reader.py at STRATA_REVISION (MIT).
BLOCKS = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    6: (32, 22),
    8: (32, 34),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    30: (1, 2),
    42: (64, 18),
}
SCALARS = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
LAYER = re.compile(r"^blk\.(\d+)\.")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    kind: int
    offset: int


@dataclass(frozen=True)
class Shard:
    name: str
    directory_sha256: str
    data_start: int
    tensors: tuple[tuple[Tensor, int], ...]  # Tensor and its aligned storage span.
    metadata: dict[str, Any]


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0

    def take(self, count: int) -> bytes:
        if count < 0 or count > len(self.data) - self.at:
            raise ValueError("GGUF directory exceeds the bounded header prefix")
        out = self.data[self.at : self.at + count]
        self.at += count
        return out

    def scalar(self, kind: int) -> Any:
        if kind not in SCALARS:
            raise ValueError(f"unsupported GGUF metadata scalar type {kind}")
        fmt, size = SCALARS[kind]
        return struct.unpack("<" + fmt, self.take(size))[0]

    def text(self, *, decode: bool = True) -> str:
        size = self.scalar(10)
        raw = self.take(size)
        return raw.decode("utf-8") if decode else ""

    def value(self, kind: int, key: str) -> Any:
        if kind == 8:
            return self.text(decode=key == "general.architecture")
        if kind == 9:
            member, count = self.scalar(4), self.scalar(10)
            if member == 8:
                # Tokenizer string arrays dominate the 11-MB header. Never retain them.
                for _ in range(count):
                    self.text(decode=False)
                return None
            if member not in SCALARS:
                raise ValueError(f"unsupported GGUF metadata array type {member}")
            fmt, size = SCALARS[member]
            raw = self.take(count * size)
            if key == "qwen4exp.attention.compress_ratios":
                return struct.unpack(f"<{count}{fmt}", raw)
            return None
        return self.scalar(kind)


def read_shard(name: str, prefix_path: Path) -> Shard:
    path, file_size, _ = SHARDS[name]
    limit = HEADER_BYTES[name]
    with prefix_path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) != limit:
        raise ValueError(
            f"{name}: expected exactly {limit} bytes of GGUF header, without payload"
        )
    reader = Reader(data)
    if reader.take(4) != b"GGUF" or reader.scalar(4) != 3:
        raise ValueError(f"{name}: expected little-endian GGUF v3")
    n_tensors, n_meta = reader.scalar(10), reader.scalar(10)
    if (n_tensors, n_meta) != ((1, 3) if name == "ngram" else (1223, 73)):
        raise ValueError(f"{name}: unexpected tensor or metadata count")
    meta = {}
    for _ in range(n_meta):
        key = reader.text()
        if key in meta:
            raise ValueError(f"{name}: duplicate GGUF metadata {key}")
        meta[key] = reader.value(reader.scalar(4), key)
    tensors = []
    names = set()
    for _ in range(n_tensors):
        tensor_name = reader.text()
        dims = reader.scalar(4)
        if not 1 <= dims <= 4 or tensor_name in names:
            raise ValueError(
                f"{name}: invalid dimensions or repeated tensor {tensor_name}"
            )
        names.add(tensor_name)
        shape = tuple(reader.scalar(10) for _ in range(dims))
        tensors.append(Tensor(tensor_name, shape, reader.scalar(4), reader.scalar(10)))
    alignment = meta.get("general.alignment") or 32
    if alignment != 32:
        raise ValueError(f"{name}: unexpected GGUF alignment {alignment}")
    data_start = (reader.at + alignment - 1) // alignment * alignment
    if data_start != limit:
        raise ValueError(f"{name}: expected data to begin after byte {limit}")
    if meta.get("split.count") != 2 or meta.get("split.no") != (
        1 if name == "ngram" else 0
    ):
        raise ValueError(f"{name}: unexpected split position")
    if name != "ngram" and meta.get("general.architecture") != "qwen4exp":
        raise ValueError(f"{name}: wrong model architecture")
    if name == "ngram" and (
        len(tensors) != 1 or tensors[0].name != "per_layer_token_embd.weight"
    ):
        raise ValueError("ngram: expected only the n-gram lookup tensor")
    ordered = sorted(tensors, key=lambda tensor: tensor.offset)
    payload_size = file_size - data_start
    if not ordered or ordered[0].offset != 0:
        raise ValueError(f"{name}: data must start at offset zero")
    spans = []
    for index, tensor in enumerate(ordered):
        end = ordered[index + 1].offset if index + 1 < len(ordered) else payload_size
        if tensor.kind not in BLOCKS or not tensor.shape or not all(tensor.shape):
            raise ValueError(
                f"{name}: unsupported GGML format or shape for {tensor.name}"
            )
        elements = 1
        for dim in tensor.shape:
            elements *= dim
        block_elements, block_bytes = BLOCKS[tensor.kind]
        if elements % block_elements:
            raise ValueError(f"{name}: incomplete GGML block for {tensor.name}")
        stored = elements // block_elements * block_bytes
        if (
            tensor.offset % alignment
            or not stored <= end - tensor.offset < stored + alignment
        ):
            raise ValueError(
                f"{name}: tensor {tensor.name} does not match its offset bracket"
            )
        spans.append((tensor, end - tensor.offset))
    if sum(span for _, span in spans) != payload_size:
        raise ValueError(f"{name}: tensor spans do not cover the pinned file size")
    fingerprint = hashlib.sha256(data[: reader.at]).hexdigest()
    if fingerprint != HEADER_SHA256[name]:
        raise ValueError(f"{name}: GGUF directory does not match the pinned checkpoint")
    return Shard(path, fingerprint, data_start, tuple(spans), meta)


def memory_plan(model: Shard, *, context: int, trace_floor: int) -> dict[str, Any]:
    m = model.metadata
    layers = m.get("qwen4exp.block_count")
    interval = m.get("qwen4exp.full_attention_interval")
    ratios = m.get("qwen4exp.attention.compress_ratios")
    if (
        layers != 48
        or interval != 4
        or ratios != tuple(4 if i % 4 == 3 else 0 for i in range(48))
    ):
        raise ValueError(
            "GGUF layer count or QSA compression layout differs from the audited model"
        )
    if (
        m.get("qwen4exp.attention.head_count_kv"),
        m.get("qwen4exp.attention.key_length"),
        m.get("qwen4exp.attention.value_length"),
        m.get("qwen4exp.attention.indexer.key_length"),
    ) != (2, 256, 256, 128):
        raise ValueError("GGUF QSA head geometry differs from the audited model")
    if not 1 <= context <= m.get("qwen4exp.context_length", 0):
        raise ValueError("context is outside the GGUF model's native window")
    layer_bytes = [0] * layers
    expert_bytes = [0] * layers
    globals_ = {}
    for tensor, span in model.tensors:
        match = LAYER.match(tensor.name)
        if match:
            index = int(match.group(1))
            if index >= layers:
                raise ValueError("GGUF contains an out-of-range model layer")
            layer_bytes[index] += span
            if "_exps." in tensor.name:
                expert_bytes[index] += span
        else:
            globals_[tensor.name] = span
    if len([1 for tensor, _ in model.tensors if "_exps." in tensor.name]) != 144:
        raise ValueError("GGUF has an unexpected routed-expert tensor count")
    if set(globals_) != {
        "token_embd.weight",
        "output.weight",
        "output_hc_down.weight",
        "output_hc_norm.weight",
        "output_hc_up.weight",
    }:
        raise ValueError("GGUF has an unexpected non-layer weight")
    # Conservative text-only, single-stream BF16 K/V; QSA pooled indexer keys FP32.
    kv_layer = context * 2 * 2 * 256 * 2
    index_layer = ((context + 3) // 4) * 128 * 4 + 3 * 128 * 4
    channels = (
        m["qwen4exp.ssm.inner_size"]
        + 2 * m["qwen4exp.ssm.group_count"] * m["qwen4exp.ssm.state_size"]
    )
    state_layer = (
        m["qwen4exp.ssm.state_size"] ** 2 * m["qwen4exp.ssm.time_step_rank"]
        + (m["qwen4exp.ssm.conv_kernel"] - 1) * channels
    ) * 4
    if (kv_layer, state_layer, m["qwen4exp.expert_count"]) != (
        context * 2048,
        3_268_608,
        512,
    ):
        raise ValueError("GGUF state geometry differs from the audited model")
    splits = []
    for split in range(1, layers):
        devices = []
        for chip, indices in ((0, range(split)), (1, range(split, layers))):
            weight = sum(layer_bytes[i] for i in indices)
            weight += (
                globals_["token_embd.weight"]
                if chip == 0
                else sum(
                    span
                    for name, span in globals_.items()
                    if name != "token_embd.weight"
                )
            )
            qsa = sum(i % interval == interval - 1 for i in indices)
            gdn = len(indices) - qsa
            kv, index = qsa * kv_layer, qsa * index_layer
            state = gdn * state_layer
            floor = weight + kv + index + state + trace_floor
            devices.append({
                "id": chip,
                "layers": [indices.start, indices.stop],
                "weights_bytes": weight,
                "routed_experts_bytes": sum(expert_bytes[i] for i in indices),
                "kv_bf16_bytes": kv,
                "indexer_fp32_bytes": index,
                "gdn_state_fp32_bytes": state,
                "trace_floor_bytes": trace_floor,
                "measured_trace_peak_bytes": None,
                "prompt_scratch_peak_bytes": None,
                "weight_materialization_overhead_bytes": None,
                "allocatable_dram_bytes": None,
                "known_floor_bytes": floor,
                "nominal_headroom_bytes": P150_NOMINAL_BYTES - floor,
            })
        splits.append((
            max(device["known_floor_bytes"] for device in devices),
            split,
            devices,
        ))
    _, split, devices = min(splits)
    return {
        "split_after_layer": split - 1,
        "devices": devices,
        "total_routed_experts_bytes": sum(expert_bytes),
        "nominal_floor_fits": all(
            device["nominal_headroom_bytes"] > 0 for device in devices
        ),
    }


def ttnn_format_projection(model: Shard, plan: dict[str, Any]) -> dict[str, Any]:
    """Optimistic fully resident 32x32 TTNN TILE layout at the GGUF-selected split."""
    split = plan["split_after_layer"] + 1
    overhead = [
        device["known_floor_bytes"] - device["weights_bytes"]
        for device in plan["devices"]
    ]
    projections = {}
    for dtype, tile_bytes in TTNN_TILE_BYTES.items():
        weights = [0, 0]
        for tensor, _ in model.tensors:
            shape = (1, *tensor.shape) if len(tensor.shape) == 1 else tensor.shape
            tiles = (
                math.prod(shape[:-2])
                * ((shape[-2] + 31) // 32)
                * ((shape[-1] + 31) // 32)
            )
            match = LAYER.match(tensor.name)
            if match:
                chip = 0 if int(match.group(1)) < split else 1
            else:
                chip = 0 if tensor.name == "token_embd.weight" else 1
            weights[chip] += tiles * tile_bytes
        total = sum(weights)
        projections[dtype] = {
            "tile_bytes": tile_bytes,
            "total_weight_bytes": total,
            "exceeds_two_card_nominal_weight_capacity": total > 2 * P150_NOMINAL_BYTES,
            "devices": [
                {
                    "id": chip,
                    "weights_bytes": weight,
                    "known_floor_bytes": weight + overhead[chip],
                    "nominal_headroom_bytes": P150_NOMINAL_BYTES
                    - weight
                    - overhead[chip],
                }
                for chip, weight in enumerate(weights)
            ],
        }
    return projections


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("cannot measure host MemAvailable")


def run(headers: Path, context: int, disk_path: Path) -> dict[str, Any]:
    names = {"Q2_0": "q2", "IQ2_XS": "iq2", "ngram": "ngram"}
    shards = {key: read_shard(key, headers / f"{names[key]}.header") for key in SHARDS}
    for name in ("Q2_0", "IQ2_XS"):
        if any(
            shards[name].metadata[key] != shards["Q2_0"].metadata[key]
            for key in (
                "qwen4exp.block_count",
                "qwen4exp.context_length",
                "qwen4exp.attention.compress_ratios",
            )
        ):
            raise ValueError(
                f"{name}: model geometry differs across checkpoint variants"
            )
    host_available = available_memory()
    disk_available = shutil.disk_usage(disk_path).free
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "source": {
            "hf_repository": REPOSITORY,
            "hf_revision": REVISION,
            "strata_revision": STRATA_REVISION,
            "header_only": True,
        },
        "scenario": {
            "architecture": "P150x2 layer pipeline (not existing 27B tensor parallel)",
            "context_tokens": context,
            "concurrent_requests": 1,
            "kv_dtype": "bf16",
            "vision": False,
            "mtp": False,
            "nominal_bytes_per_p150": P150_NOMINAL_BYTES,
            "trace_floor_from_27b_bytes_per_chip": TRACE_FLOOR_BYTES,
        },
        "host": {
            "mem_available_bytes": host_available,
            "extra_host_floor_bytes": HOST_OTHER_FLOOR_BYTES,
            "pinnable_bytes": None,
        },
        "ssd": {
            "path": str(disk_path),
            "available_bytes": disk_available,
            "shared_ngram_shard_bytes": SHARDS["ngram"][1],
            "ngram_page_cache_bytes": None,
        },
        "shards": {},
        "variants": {},
    }
    for name, shard in shards.items():
        path, size, sha = SHARDS[name]
        receipt["shards"][name] = {
            "path": path,
            "pinned_bytes": size,
            "lfs_sha256_unverified": sha,
            "directory_sha256": shard.directory_sha256,
            "data_start_bytes": shard.data_start,
            "tensor_count": len(shard.tensors),
            "ggml_type_tensor_counts": dict(
                Counter(tensor.kind for tensor, _ in shard.tensors)
            ),
        }
    for name in ("Q2_0", "IQ2_XS"):
        plan = memory_plan(shards[name], context=context, trace_floor=TRACE_FLOOR_BYTES)
        plan["ttnn_all_tensor_tile_projection"] = ttnn_format_projection(
            shards[name], plan
        )
        host_floor = plan["total_routed_experts_bytes"] + HOST_OTHER_FLOOR_BYTES
        disk_floor = SHARDS[name][1] + SHARDS["ngram"][1]
        blockers = []
        if not plan["nominal_floor_fits"]:
            blockers.append("per-chip known floor exceeds nominal P150 DRAM")
        if host_available < host_floor:
            blockers.append(
                "host MemAvailable cannot cover an expert CPU fallback plus reserve"
            )
        if disk_available < disk_floor:
            blockers.append("free SSD cannot hold both pinned shards")
        if plan["ttnn_all_tensor_tile_projection"]["BFLOAT4_B"][
            "exceeds_two_card_nominal_weight_capacity"
        ]:
            blockers.append(
                "even optimistic fully resident TTNN BFLOAT4_B weights exceed both nominal P150 DRAM capacities"
            )
        blockers.extend((
            "Flash Next TT-native GGUF quantized operators and placement not implemented",
            "Flash Next per-chip usable DRAM, prompt scratch, weight conversion and trace peak unmeasured",
        ))
        plan.update({
            "host_fallback_floor_bytes": host_floor,
            "host_fallback_floor_fits": host_available >= host_floor,
            "ssd_download_bytes": disk_floor,
            "ssd_download_fits": disk_available >= disk_floor,
            "memory_admitted": not blockers,
            "one_layer_device_check_ready": not blockers,
            "blockers": blockers,
        })
        receipt["variants"][name] = plan
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headers",
        type=Path,
        required=True,
        help="directory of q2.header, iq2.header, ngram.header from pinned HTTP byte ranges",
    )
    parser.add_argument("--context", type=int, default=65_536)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.headers, args.context, args.disk_path), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
