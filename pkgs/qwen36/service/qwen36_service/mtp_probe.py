"""Run the diagnostic Qwen3.8 target-verification probe on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.mtp_verification_probe]

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from transformers import AutoTokenizer

from .decode_adapter import TracedDecodeAdapter
from .decode_core import DecodePolicyError, DecodeProfile
from .mtp_probe_adapter import MtpVerificationProbeAdapter
from .mtp_probe_core import (
    MtpProbeIdentity,
    MtpProbePolicyError,
    MtpProbeProfile,
    MtpProbeSample,
    evaluate_mtp_probe,
)

RECEIPT_SCHEMA_VERSION = 3
EXPECTED_PCIE_DEVICE_IDS = [0, 1]
MESH_ROWS = 1
MESH_COLUMNS = 2
MAXIMUM_BATCH_SIZE = 1
EXPECTED_LAYER_COUNT = 64
L1_SMALL_SIZE_BYTES = 24_576
COMMAND_QUEUE_COUNT = 1
TRACE_REGION_SIZE_BYTES = 1_073_741_824
DECODE_SLOT_ID = "mtp-probe-slot"
INITIAL_BUFFER_GENERATION = 1
MODEL_ALIAS = "Qwen3.8-27B"
MEASUREMENT_MODE = "diagnostic-mtp-target-verification"


class MtpProbeCommandError(RuntimeError):
    """The physical probe command cannot produce a trustworthy receipt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--mtp-probe-profile", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--package-identity", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = _existing_absolute_directory(args.model_path, "--model-path")
    decode_profile_path = _existing_absolute_file(
        args.decode_profile, "--decode-profile"
    )
    probe_profile_path = _existing_absolute_file(
        args.mtp_probe_profile, "--mtp-probe-profile"
    )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        raise SystemExit("--output must be an absolute path")
    if output_path.exists():
        raise SystemExit("--output must not already exist")
    if not output_path.parent.is_dir():
        raise SystemExit("--output parent directory does not exist")

    decode_profile = _load_decode_profile(decode_profile_path)
    probe_profile = _load_probe_profile(probe_profile_path)
    if model_path.name != probe_profile.checkpoint_revision:
        raise SystemExit("model path does not match the MTP probe checkpoint")
    if decode_profile.authority_blake3 != probe_profile.authority_blake3:
        raise SystemExit("decode and MTP probe authority identities differ")
    if decode_profile.checkpoint_revision != probe_profile.checkpoint_revision:
        raise SystemExit("decode and MTP probe checkpoint identities differ")
    if decode_profile.runtime_source_revision != probe_profile.model_source_revision:
        raise SystemExit("decode and MTP probe source identities differ")
    if decode_profile.tokenizer_revision != probe_profile.tokenizer_revision:
        raise SystemExit("decode and MTP probe tokenizer identities differ")
    if decode_profile.trace_region_size_bytes != TRACE_REGION_SIZE_BYTES:
        raise SystemExit("decode profile trace-region size does not match the probe")

    os.environ["HF_MODEL"] = str(model_path)
    os.environ["TT_CACHE_PATH"] = str(model_path)
    pcie_device_ids = list(ttnn.get_pcie_device_ids())
    if pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
        raise SystemExit(
            f"expected physical PCIe devices {EXPECTED_PCIE_DEVICE_IDS}, got {pcie_device_ids}"
        )

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh_device = None
    model = None
    decode_adapter = None
    samples: tuple[MtpProbeSample, ...] | None = None
    cleanup_complete = False
    identity = MtpProbeIdentity(
        runtime_identity=args.runtime_identity,
        package_identity=args.package_identity,
        model_identity=f"{MODEL_ALIAS}:{probe_profile.checkpoint_revision}",
    )
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(MESH_ROWS, MESH_COLUMNS),
            l1_small_size=L1_SMALL_SIZE_BYTES,
            num_command_queues=COMMAND_QUEUE_COUNT,
            trace_region_size=TRACE_REGION_SIZE_BYTES,
        )
        if mesh_device.get_num_devices() != len(EXPECTED_PCIE_DEVICE_IDS):
            raise MtpProbeCommandError("P150x2 mesh did not open both devices")
        mesh_device.enable_program_cache()
        model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=MAXIMUM_BATCH_SIZE,
            max_seq_len=decode_profile.maximum_sequence_length,
        )
        if len(model.layers) != EXPECTED_LAYER_COUNT:
            raise MtpProbeCommandError(
                f"expected {EXPECTED_LAYER_COUNT} layers, got {len(model.layers)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_tokens = tokenizer(
            probe_profile.prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(torch.int32)
        decode_adapter = TracedDecodeAdapter(
            model=model,
            profile=decode_profile,
            runtime_identity=identity.runtime_identity,
            package_identity=identity.package_identity,
            model_identity=identity.model_identity,
            slot_id=DECODE_SLOT_ID,
            buffer_generation=INITIAL_BUFFER_GENERATION,
        )
        probe_adapter = MtpVerificationProbeAdapter(
            model=model,
            decode_adapter=decode_adapter,
            profile=probe_profile,
            identity=identity,
        )
        samples = probe_adapter.run(prompt_tokens)
    finally:
        if decode_adapter is not None:
            decode_adapter.close()
            decode_adapter = None
        if model is not None:
            model.free_kv_caches()
            model = None
            gc.collect()
        try:
            if mesh_device is not None:
                for submesh in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
                mesh_device = None
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        cleanup_complete = True

    if samples is None:
        raise MtpProbeCommandError("target probe produced no samples")
    decision = evaluate_mtp_probe(probe_profile, identity, samples)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "measurement_mode": MEASUREMENT_MODE,
        "diagnostic_only": True,
        "admission_eligible": False,
        "simulated": False,
        "cleanup_complete": cleanup_complete,
        "identity": {
            "runtime": identity.runtime_identity,
            "package": identity.package_identity,
            "model": identity.model_identity,
            "authority_blake3": probe_profile.authority_blake3,
            "checkpoint_revision": probe_profile.checkpoint_revision,
            "model_source_revision": probe_profile.model_source_revision,
            "tokenizer_revision": probe_profile.tokenizer_revision,
        },
        "physical_pcie_device_ids": EXPECTED_PCIE_DEVICE_IDS,
        "mesh_shape": [MESH_ROWS, MESH_COLUMNS],
        "host_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "trace_lifetime": "decode-released-native-captured-per-sample",
        "probe_profile_id": probe_profile.profile_id,
        "verification_widths": list(probe_profile.verification_widths),
        "execution_path": probe_profile.execution_path,
        "native_width_limit": probe_profile.native_width_limit,
        "runtime_supports_fused_chunk_gdn": hasattr(
            ttnn.transformer, "chunk_gated_delta_rule"
        ),
        "samples": [_sample_mapping(sample) for sample in samples],
        "decision": decision.to_mapping(),
        "non_claims": [
            "No MTP drafter execution or acceptance result is established.",
            "No rejection rollback behavior is established.",
            "No speculative decoding speedup or production admission is established.",
            "The hybrid packed-decode path is target-only evidence.",
        ],
    }
    _write_receipt(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _sample_mapping(sample: MtpProbeSample) -> dict[str, Any]:
    return {
        "width": sample.width,
        "sample_index": sample.sample_index,
        "physical_pcie_device_ids": list(sample.physical_pcie_device_ids),
        "sequential_token_ids": list(sample.sequential_token_ids),
        "verification_token_ids": list(sample.verification_token_ids),
        "sequential_seconds": sample.sequential_seconds,
        "verification_trace_capture_seconds": sample.verification_trace_capture_seconds,
        "verification_device_seconds": sample.verification_device_seconds,
        "verification_readback_seconds": sample.verification_readback_seconds,
        "logits_pcc": sample.logits_pcc,
        "logits_nmse": sample.logits_nmse,
        "recurrent_state_pcc": sample.recurrent_state_pcc,
        "recurrent_state_nmse": sample.recurrent_state_nmse,
        "convolution_state_pcc": sample.convolution_state_pcc,
        "convolution_state_nmse": sample.convolution_state_nmse,
        "paged_kv_state_pcc": sample.paged_kv_state_pcc,
        "paged_kv_state_nmse": sample.paged_kv_state_nmse,
    }


def _load_decode_profile(path: Path) -> DecodeProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return DecodeProfile.from_mapping(json.load(profile_file))
    except (DecodePolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"decode profile is invalid: {error}") from error


def _load_probe_profile(path: Path) -> MtpProbeProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return MtpProbeProfile.from_mapping(json.load(profile_file))
    except (MtpProbePolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"MTP probe profile is invalid: {error}") from error


def _existing_absolute_directory(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise SystemExit(f"{name} must be an existing absolute directory")
    return path


def _existing_absolute_file(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise SystemExit(f"{name} must be an existing absolute file")
    return path


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary_suffix = ".tmp"
    temporary_path = path.with_name(path.name + temporary_suffix)
    with temporary_path.open("x", encoding="utf-8") as receipt_file:
        json.dump(receipt, receipt_file, indent=2, sort_keys=True)
        receipt_file.write("\n")
        receipt_file.flush()
        os.fsync(receipt_file.fileno())
    os.replace(temporary_path, path)


if __name__ == "__main__":
    main()
