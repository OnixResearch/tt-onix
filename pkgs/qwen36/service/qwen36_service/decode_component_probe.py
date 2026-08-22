"""Run diagnostic Qwen3.8 decode component replay on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.decode_component_attribution]

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from transformers import AutoTokenizer

from .decode_adapter import TracedDecodeAdapter
from .decode_component_adapter import DecodeComponentProbeAdapter
from .decode_component_core import (
    DecodeComponentEvidence,
    DecodeComponentIdentity,
    DecodeComponentPolicyError,
    DecodeComponentProfile,
    evaluate_decode_components,
)
from .decode_core import DecodePolicyError, DecodeProfile

RECEIPT_SCHEMA_VERSION = 1
MAXIMUM_BATCH_SIZE = 1
L1_SMALL_SIZE_BYTES = 24_576
COMMAND_QUEUE_COUNT = 1
MODEL_ALIAS = "Qwen3.8-27B"
DECODE_SLOT_ID = "decode-component-probe-slot"
INITIAL_BUFFER_GENERATION = 1
MEASUREMENT_MODE = "diagnostic-decode-component-trace-replay"


class DecodeComponentCommandError(RuntimeError):
    """The component probe command cannot produce trustworthy evidence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--component-profile", required=True)
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
    component_profile_path = _existing_absolute_file(
        args.component_profile, "--component-profile"
    )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        raise SystemExit("--output must be an absolute path")
    if output_path.exists():
        raise SystemExit("--output must not already exist")
    if not output_path.parent.is_dir():
        raise SystemExit("--output parent directory does not exist")

    decode_profile = _load_decode_profile(decode_profile_path)
    component_profile = _load_component_profile(component_profile_path)
    _validate_profile_binding(model_path, decode_profile, component_profile)
    os.environ["HF_MODEL"] = str(model_path)
    os.environ["TT_CACHE_PATH"] = str(model_path)

    pcie_device_ids = tuple(int(device_id) for device_id in ttnn.get_pcie_device_ids())
    if pcie_device_ids != component_profile.expected_pcie_device_ids:
        raise SystemExit(
            "physical PCIe device IDs differ from the component profile: "
            f"expected {component_profile.expected_pcie_device_ids}, got {pcie_device_ids}"
        )

    identity = DecodeComponentIdentity(
        runtime_identity=args.runtime_identity,
        package_identity=args.package_identity,
        model_identity=f"{MODEL_ALIAS}:{component_profile.checkpoint_revision}",
        authority_blake3=component_profile.authority_blake3,
        checkpoint_revision=component_profile.checkpoint_revision,
        model_source_revision=component_profile.model_source_revision,
        tokenizer_revision=component_profile.tokenizer_revision,
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
    probe_run = None
    cleanup_complete = False
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*component_profile.mesh_shape),
            l1_small_size=L1_SMALL_SIZE_BYTES,
            num_command_queues=COMMAND_QUEUE_COUNT,
            trace_region_size=component_profile.trace_region_size_bytes,
        )
        if mesh_device.get_num_devices() != len(
            component_profile.expected_pcie_device_ids
        ):
            raise DecodeComponentCommandError(
                "component probe mesh did not open all expected devices"
            )
        mesh_device.enable_program_cache()
        model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=MAXIMUM_BATCH_SIZE,
            max_seq_len=component_profile.maximum_sequence_length,
        )
        if len(model.layers) != component_profile.layer_count:
            raise DecodeComponentCommandError(
                f"expected {component_profile.layer_count} layers, got {len(model.layers)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_tokens = tokenizer(
            component_profile.prompt,
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
        probe = DecodeComponentProbeAdapter(
            model=model,
            decode_adapter=decode_adapter,
            profile=component_profile,
            identity=identity,
            physical_pcie_device_ids=pcie_device_ids,
        )
        probe_run = probe.run(prompt_tokens)
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

    if probe_run is None:
        raise DecodeComponentCommandError("component probe produced no evidence")
    evidence = replace(probe_run.evidence, cleanup_complete=cleanup_complete)
    decision = evaluate_decode_components(component_profile, evidence)
    receipt = _receipt_mapping(
        profile=component_profile,
        evidence=evidence,
        decision=decision.to_mapping(),
        trace_captures=[capture.to_mapping() for capture in probe_run.trace_captures],
        prompt_token_count=probe_run.prompt_token_count,
        first_decode_token=probe_run.first_decode_token,
    )
    _write_receipt(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _receipt_mapping(
    *,
    profile: DecodeComponentProfile,
    evidence: DecodeComponentEvidence,
    decision: dict[str, Any],
    trace_captures: list[dict[str, Any]],
    prompt_token_count: int,
    first_decode_token: int,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "measurement_mode": MEASUREMENT_MODE,
        "diagnostic_only": True,
        "admission_eligible": False,
        "simulated": False,
        "cleanup_complete": evidence.cleanup_complete,
        "profile_id": profile.profile_id,
        "execution_path": profile.execution_path,
        "identity": {
            "runtime": evidence.identity.runtime_identity,
            "package": evidence.identity.package_identity,
            "model": evidence.identity.model_identity,
            "authority_blake3": evidence.identity.authority_blake3,
            "checkpoint_revision": evidence.identity.checkpoint_revision,
            "model_source_revision": evidence.identity.model_source_revision,
            "tokenizer_revision": evidence.identity.tokenizer_revision,
        },
        "physical_pcie_device_ids": list(evidence.physical_pcie_device_ids),
        "mesh_shape": list(evidence.mesh_shape),
        "host_cpu_affinity": list(evidence.host_cpu_affinity),
        "prompt": profile.prompt,
        "prompt_token_count": prompt_token_count,
        "first_decode_token": first_decode_token,
        "samples_per_group": profile.samples_per_group,
        "warmup_replays": profile.warmup_replays,
        "layer_types": [
            {
                "layer_index": observation.layer_index,
                "mixer_type": observation.mixer_type,
            }
            for observation in evidence.layer_types
        ],
        "trace_captures": trace_captures,
        "component_samples": [
            {
                "component": sample.component,
                "layer_index": sample.layer_index,
                "sample_index": sample.sample_index,
                "seconds": sample.seconds,
            }
            for sample in evidence.component_samples
        ],
        "full_samples": [
            {
                "sample_index": sample.sample_index,
                "device_seconds": sample.device_seconds,
                "token_id": sample.token_id,
            }
            for sample in evidence.full_samples
        ],
        "host_samples": [
            {
                "sample_index": sample.sample_index,
                "input_update_seconds": sample.input_update_seconds,
                "winner_readback_seconds": sample.winner_readback_seconds,
            }
            for sample in evidence.host_samples
        ],
        "decision": decision,
        "non_claims": [
            "Component replay is diagnostic and is ineligible for production admission.",
            "Independent component traces do not prove full-model optimization speedup.",
            "The receipt makes no batch, concurrency, long-context, thermal, or TP4 claim.",
            "Production remains eager.",
        ],
    }


def _load_decode_profile(path: Path) -> DecodeProfile:
    try:
        source = json.loads(path.read_text())
        return DecodeProfile.from_mapping(source)
    except (DecodePolicyError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid decode profile: {error}") from error


def _load_component_profile(path: Path) -> DecodeComponentProfile:
    try:
        source = json.loads(path.read_text())
        return DecodeComponentProfile.from_mapping(source)
    except (
        DecodeComponentPolicyError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"invalid component profile: {error}") from error


def _validate_profile_binding(
    model_path: Path,
    decode_profile: DecodeProfile,
    component_profile: DecodeComponentProfile,
) -> None:
    if model_path.name != component_profile.checkpoint_revision:
        raise SystemExit("model path does not match the component checkpoint")
    if decode_profile.authority_blake3 != component_profile.authority_blake3:
        raise SystemExit("decode and component authority identities differ")
    if decode_profile.checkpoint_revision != component_profile.checkpoint_revision:
        raise SystemExit("decode and component checkpoint identities differ")
    if (
        decode_profile.runtime_source_revision
        != component_profile.model_source_revision
    ):
        raise SystemExit("decode and component source identities differ")
    if decode_profile.tokenizer_revision != component_profile.tokenizer_revision:
        raise SystemExit("decode and component tokenizer identities differ")
    if (
        decode_profile.maximum_sequence_length
        != component_profile.maximum_sequence_length
    ):
        raise SystemExit("decode and component sequence bounds differ")
    if decode_profile.kv_block_size != component_profile.kv_block_size:
        raise SystemExit("decode and component KV block sizes differ")
    if (
        decode_profile.trace_region_size_bytes
        != component_profile.trace_region_size_bytes
    ):
        raise SystemExit("decode and component trace-region sizes differ")


def _existing_absolute_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise SystemExit(f"{label} must name an existing absolute directory")
    return path


def _existing_absolute_file(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise SystemExit(f"{label} must name an existing absolute file")
    return path


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
