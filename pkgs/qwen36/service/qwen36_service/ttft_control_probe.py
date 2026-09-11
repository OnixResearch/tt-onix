"""Run the diagnostic Qwen3.8 same-process TTFT control on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.traced_ttft_control]

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from transformers import AutoTokenizer

from .decode_adapter import TracedDecodeAdapter
from .decode_core import DecodePolicyError, DecodeProfile
from .ttft_control_adapter import TtftControlAdapter, TtftControlRun
from .ttft_control_core import (
    TtftControlDecision,
    TtftControlIdentity,
    TtftControlPolicyError,
    TtftControlProfile,
    evaluate_ttft_control,
)

RECEIPT_SCHEMA_VERSION = 1
MAXIMUM_BATCH_SIZE = 1
L1_SMALL_SIZE_BYTES = 24_576
COMMAND_QUEUE_COUNT = 1
MODEL_ALIAS = "Qwen3.8-27B"
DECODE_SLOT_ID = "ttft-control-slot"
INITIAL_BUFFER_GENERATION = 1
MEASUREMENT_MODE = "diagnostic-same-process-ttft-control"
JSON_INDENT_SPACES = 2


class TtftControlCommandError(RuntimeError):
    """The TTFT control command cannot produce trustworthy evidence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--control-profile", required=True)
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
    control_profile_path = _existing_absolute_file(
        args.control_profile, "--control-profile"
    )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        raise SystemExit("--output must be an absolute path")
    if output_path.exists():
        raise SystemExit("--output must not already exist")
    if not output_path.parent.is_dir():
        raise SystemExit("--output parent directory does not exist")

    decode_profile = _load_decode_profile(decode_profile_path)
    control_profile = _load_control_profile(control_profile_path)
    _validate_profile_binding(model_path, decode_profile, control_profile)
    os.environ["HF_MODEL"] = str(model_path)
    os.environ["TT_CACHE_PATH"] = str(model_path)

    pcie_device_ids = tuple(int(device_id) for device_id in ttnn.get_pcie_device_ids())
    if pcie_device_ids != control_profile.expected_pcie_device_ids:
        raise SystemExit(
            "physical PCIe device IDs differ from the TTFT control profile: "
            f"expected {control_profile.expected_pcie_device_ids}, got {pcie_device_ids}"
        )
    host_cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if host_cpu_affinity != control_profile.expected_cpu_affinity:
        raise SystemExit(
            "host CPU affinity differs from the TTFT control profile: "
            f"expected {control_profile.expected_cpu_affinity}, got {host_cpu_affinity}"
        )

    identity = TtftControlIdentity(
        runtime_identity=args.runtime_identity,
        package_identity=args.package_identity,
        model_identity=f"{MODEL_ALIAS}:{control_profile.checkpoint_revision}",
        authority_blake3=control_profile.authority_blake3,
        checkpoint_revision=control_profile.checkpoint_revision,
        model_source_revision=control_profile.model_source_revision,
        tokenizer_revision=control_profile.tokenizer_revision,
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
    control_adapter = None
    control_run = None
    cleanup_complete = False
    cleanup_errors: list[str] = []
    try:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*control_profile.mesh_shape),
            l1_small_size=L1_SMALL_SIZE_BYTES,
            num_command_queues=COMMAND_QUEUE_COUNT,
            trace_region_size=control_profile.trace_region_size_bytes,
        )
        if mesh_device.get_num_devices() != len(
            control_profile.expected_pcie_device_ids
        ):
            raise TtftControlCommandError(
                "TTFT control mesh did not open all expected devices"
            )
        mesh_device.enable_program_cache()
        model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=MAXIMUM_BATCH_SIZE,
            max_seq_len=control_profile.maximum_sequence_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_tokens = tokenizer(
            control_profile.prompt,
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
        if (
            decode_adapter.decode_page_table_blocks
            != control_profile.trace_capacity_blocks
        ):
            raise TtftControlCommandError("decode trace capacity changed")
        control_adapter = TtftControlAdapter(
            model=model,
            decode_adapter=decode_adapter,
            profile=control_profile,
            identity=identity,
            physical_pcie_device_ids=pcie_device_ids,
            host_cpu_affinity=host_cpu_affinity,
        )
        control_run = control_adapter.run(prompt_tokens)
    finally:
        if control_adapter is not None:
            try:
                control_adapter.close()
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                cleanup_errors.append(f"control-adapter:{type(error).__name__}")
            control_adapter = None
        if decode_adapter is not None:
            try:
                decode_adapter.close()
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                cleanup_errors.append(f"decode-adapter:{type(error).__name__}")
            decode_adapter = None
        if model is not None:
            try:
                model.free_kv_caches()
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                cleanup_errors.append(f"model-cache:{type(error).__name__}")
            model = None
            gc.collect()
        if mesh_device is not None:
            try:
                for submesh in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
            except (AssertionError, OSError, RuntimeError, ValueError) as error:
                cleanup_errors.append(f"mesh:{type(error).__name__}")
            mesh_device = None
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except (AssertionError, OSError, RuntimeError, ValueError) as error:
            cleanup_errors.append(f"fabric:{type(error).__name__}")
        cleanup_complete = not cleanup_errors

    if cleanup_errors:
        raise TtftControlCommandError(
            "TTFT control cleanup failed: " + ",".join(cleanup_errors)
        )
    if control_run is None:
        raise TtftControlCommandError("TTFT control produced no evidence")

    evidence = replace(control_run.evidence, cleanup_complete=cleanup_complete)
    decision = evaluate_ttft_control(control_profile, evidence)
    receipt = _receipt_mapping(
        profile=control_profile,
        run=replace(control_run, evidence=evidence),
        decision=decision,
    )
    _write_receipt(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _receipt_mapping(
    *,
    profile: TtftControlProfile,
    run: TtftControlRun,
    decision: TtftControlDecision,
) -> dict[str, Any]:
    evidence = run.evidence
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "measurement_mode": MEASUREMENT_MODE,
        "diagnostic_only": True,
        "admission_eligible": False,
        "simulated": False,
        "cleanup_complete": evidence.cleanup_complete,
        "profile_id": profile.profile_id,
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
        "prompt_token_count": run.prompt_token_count,
        "generated_tokens": profile.generated_tokens,
        "page_table_blocks": profile.page_table_blocks,
        "request_cache_blocks": profile.request_cache_blocks,
        "trace_capacity_blocks": profile.trace_capacity_blocks,
        "route_schedule": list(profile.route_schedule),
        "warmup_samples": profile.warmup_samples,
        "retained_samples": profile.retained_samples,
        "trace_capture_seconds": run.trace_capture_seconds,
        "samples": [sample.to_mapping() for sample in evidence.samples],
        "decision": decision.to_mapping(),
        "non_claims": [
            "same-process control evidence does not admit traced decode",
            "cache binding control does not reproduce fresh process allocation",
            "the receipt does not apply to TP4, batching, concurrency, long context, or thermal behavior",
        ],
    }


def _load_decode_profile(path: Path) -> DecodeProfile:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
        return DecodeProfile.from_mapping(source)
    except (DecodePolicyError, json.JSONDecodeError, OSError, TypeError) as error:
        raise SystemExit(f"invalid decode profile: {error}") from error


def _load_control_profile(path: Path) -> TtftControlProfile:
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
        return TtftControlProfile.from_mapping(source)
    except (TtftControlPolicyError, json.JSONDecodeError, OSError, TypeError) as error:
        raise SystemExit(f"invalid TTFT control profile: {error}") from error


def _validate_profile_binding(
    model_path: Path,
    decode_profile: DecodeProfile,
    control_profile: TtftControlProfile,
) -> None:
    if model_path.name != control_profile.checkpoint_revision:
        raise SystemExit(
            "model path must end with the pinned checkpoint revision: "
            f"expected {control_profile.checkpoint_revision}, got {model_path.name}"
        )
    if decode_profile.authority_blake3 != control_profile.authority_blake3:
        raise SystemExit("decode and TTFT control authority identities differ")
    if decode_profile.checkpoint_revision != control_profile.checkpoint_revision:
        raise SystemExit("decode and TTFT control checkpoint identities differ")
    if decode_profile.runtime_source_revision != control_profile.model_source_revision:
        raise SystemExit("decode and TTFT control source identities differ")
    if decode_profile.tokenizer_revision != control_profile.tokenizer_revision:
        raise SystemExit("decode and TTFT control tokenizer identities differ")
    if (
        decode_profile.maximum_sequence_length
        != control_profile.maximum_sequence_length
    ):
        raise SystemExit("decode and TTFT control sequence limits differ")
    if decode_profile.kv_block_size != control_profile.kv_block_size:
        raise SystemExit("decode and TTFT control KV block sizes differ")
    if (
        decode_profile.trace_region_size_bytes
        != control_profile.trace_region_size_bytes
    ):
        raise SystemExit("decode and TTFT control trace regions differ")


def _existing_absolute_file(value: str, flag: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise SystemExit(f"{flag} must name an existing absolute file")
    return path


def _existing_absolute_directory(value: str, flag: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise SystemExit(f"{flag} must name an existing absolute directory")
    return path


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(
            receipt,
            indent=JSON_INDENT_SPACES,
            sort_keys=True,
        )
        + "\n"
    )
    path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
