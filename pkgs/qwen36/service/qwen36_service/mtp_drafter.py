"""Run the diagnostic Qwen3.8 MTP drafter smoke on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.mtp_drafter_probe]
# r[impl tenstorrent.qwen38_p150x2.mtp_recovery_stability]

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from transformers import AutoTokenizer

from . import mtp_stage_core as stage_core
from .decode_adapter import TracedDecodeAdapter
from .decode_core import DecodePolicyError, DecodeProfile
from .mtp_drafter_adapter import MtpDrafterAdapter, MtpDrafterPilotResult
from .mtp_drafter_core import (
    MtpDrafterIdentity,
    MtpDrafterPolicyError,
    MtpDrafterProfile,
    MtpDrafterSample,
    evaluate_mtp_drafter,
)
from .mtp_rollback_adapter import recovery_state_components_equal
from .mtp_rollback_core import (
    REPREFILL_RECOVERY,
    SNAPSHOT_RECOVERY,
    MtpRollbackPolicyError,
    MtpRollbackProfile,
    MtpRollbackSample,
    evaluate_mtp_rollback,
)
from .mtp_selection_core import (
    FULL_LOGIT_READBACK,
    SHARD_LOCAL_TOP1,
    MtpSelectionPolicyError,
    MtpSelectionProfile,
    MtpSelectionSample,
    evaluate_mtp_selection,
)
from .mtp_target_prewarm_core import (
    MtpTargetValidationSignature,
    make_target_validation_signature,
    target_validation_signatures_match,
)
from .mtp_trace_core import (
    EAGER_DRAFT,
    TRACED_DRAFT,
    MtpTracePolicyError,
    MtpTraceProfile,
    MtpTraceSample,
    evaluate_mtp_trace,
)

RECEIPT_SCHEMA_VERSION = 1
EXPECTED_PCIE_DEVICE_IDS = [0, 1]
MESH_ROWS = 1
MESH_COLUMNS = 2
MAXIMUM_BATCH_SIZE = 1
EXPECTED_LAYER_COUNT = 64
L1_SMALL_SIZE_BYTES = 24_576
COMMAND_QUEUE_COUNT = 1
TRACE_REGION_SIZE_BYTES = 1_073_741_824
DECODE_SLOT_ID = "mtp-drafter-probe-slot"
INITIAL_BUFFER_GENERATION = 1
MODEL_ALIAS = "Qwen3.8-27B"
SMOKE_MEASUREMENT_MODE = "diagnostic-mtp-drafter-smoke"
PILOT_MEASUREMENT_MODE = "diagnostic-mtp-drafter-pilot"
SELECTION_MEASUREMENT_MODE = "diagnostic-mtp-draft-selection"
TRACE_MEASUREMENT_MODE = "diagnostic-mtp-draft-trace"
ROLLBACK_MEASUREMENT_MODE = "diagnostic-mtp-kv-rollback"
STAGE_MEASUREMENT_MODE = "diagnostic-mtp-recovery-stage"
SMOKE_MODE = "smoke"
PILOT_MODE = "pilot"
SELECTION_MODE = "selection"
TRACE_MODE = "trace"
ROLLBACK_MODE = "rollback"
STAGE_MODE = "stage"
VLLM_REFERENCE_REVISION = "7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6"
SGLANG_REFERENCE_REVISION = "4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3"


class MtpDrafterCommandError(RuntimeError):
    """The physical drafter command cannot produce trustworthy evidence."""


class _MtpStageProgressWriter:
    """Persist each pure stage transition with an atomic replacement."""

    def __init__(
        self,
        *,
        path: Path,
        prompt_index: int,
        route: str,
        prewarm_target_validation: bool,
    ) -> None:
        self.path = path
        self.started = time.monotonic()
        self._program_cache_reader: Callable[[], int] | None = None
        self._last_program_cache_entries: int | None = None
        self.progress = stage_core.new_mtp_stage_progress(
            prompt_index=prompt_index,
            route=route,
            prewarm_target_validation=prewarm_target_validation,
        )

    def bind_program_cache(self, mesh_device: object) -> None:
        if self._program_cache_reader is not None:
            raise MtpDrafterCommandError("program-cache observation is already bound")
        reader = getattr(mesh_device, "num_program_cache_entries", None)
        if reader is None or not callable(reader):
            raise MtpDrafterCommandError("program-cache observation is unavailable")
        self._program_cache_reader = reader

    def freeze_program_cache(self) -> None:
        if self._program_cache_reader is not None:
            self._last_program_cache_entries = self._program_cache_reader()
            self._program_cache_reader = None

    def record(self, stage: str) -> None:
        if self._program_cache_reader is not None:
            self._last_program_cache_entries = self._program_cache_reader()
        self.progress = stage_core.advance_mtp_stage_progress(
            self.progress,
            stage=stage,
            elapsed_seconds=time.monotonic() - self.started,
            program_cache_entries=self._last_program_cache_entries,
        )
        _replace_json(self.path, self.progress.to_mapping())

    def record_prewarm(self, stage: str) -> None:
        self.record(stage_core.target_validation_prewarm_stage(stage))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--mtp-drafter-profile", required=True)
    parser.add_argument("--mtp-selection-profile")
    parser.add_argument("--mtp-trace-profile")
    parser.add_argument("--mtp-rollback-profile")
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--package-identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            SMOKE_MODE,
            PILOT_MODE,
            SELECTION_MODE,
            TRACE_MODE,
            ROLLBACK_MODE,
            STAGE_MODE,
        ),
        default=SMOKE_MODE,
    )
    parser.add_argument(
        "--selector-mode",
        choices=(FULL_LOGIT_READBACK, SHARD_LOCAL_TOP1),
        default=FULL_LOGIT_READBACK,
    )
    parser.add_argument("--progress-output")
    parser.add_argument(
        "--stage-route",
        choices=(REPREFILL_RECOVERY, SNAPSHOT_RECOVERY),
    )
    parser.add_argument("--stage-prompt-index", type=int)
    parser.add_argument("--stage-prewarm-target-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = _existing_absolute_directory(args.model_path, "--model-path")
    decode_profile_path = _existing_absolute_file(
        args.decode_profile, "--decode-profile"
    )
    drafter_profile_path = _existing_absolute_file(
        args.mtp_drafter_profile,
        "--mtp-drafter-profile",
    )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        raise SystemExit("--output must be absolute")
    if output_path.exists():
        raise SystemExit("--output must not exist")
    if not output_path.parent.is_dir():
        raise SystemExit("--output parent directory does not exist")
    progress_path = None
    if args.mode == STAGE_MODE:
        if args.progress_output is None:
            raise SystemExit("--progress-output is required in stage mode")
        if args.stage_route is None:
            raise SystemExit("--stage-route is required in stage mode")
        if args.stage_prompt_index is None:
            raise SystemExit("--stage-prompt-index is required in stage mode")
        progress_path = Path(args.progress_output).expanduser()
        if not progress_path.is_absolute():
            raise SystemExit("--progress-output must be absolute")
        if progress_path.exists():
            raise SystemExit("--progress-output must not exist")
        if not progress_path.parent.is_dir():
            raise SystemExit("--progress-output parent directory does not exist")
        if progress_path.resolve() == output_path.resolve():
            raise SystemExit("--progress-output and --output must differ")
    elif args.stage_prewarm_target_validation or any(
        value is not None
        for value in (
            args.progress_output,
            args.stage_route,
            args.stage_prompt_index,
        )
    ):
        raise SystemExit("stage arguments require stage mode")

    decode_profile = _load_decode_profile(decode_profile_path)
    drafter_profile = _load_drafter_profile(drafter_profile_path)
    selection_profile = None
    trace_profile = None
    rollback_profile = None
    if args.mode == SELECTION_MODE:
        if args.mtp_selection_profile is None:
            raise SystemExit("--mtp-selection-profile is required in selection mode")
        selection_profile = _load_selection_profile(
            _existing_absolute_file(
                args.mtp_selection_profile,
                "--mtp-selection-profile",
            )
        )
        _require_selection_profile_match(drafter_profile, selection_profile)
    elif args.mtp_selection_profile is not None:
        raise SystemExit("--mtp-selection-profile requires selection mode")
    if args.mode in (TRACE_MODE, ROLLBACK_MODE, STAGE_MODE):
        if args.mtp_trace_profile is None:
            raise SystemExit(
                "--mtp-trace-profile is required in trace, rollback, or stage mode"
            )
        trace_profile = _load_trace_profile(
            _existing_absolute_file(
                args.mtp_trace_profile,
                "--mtp-trace-profile",
            )
        )
        _require_trace_profile_match(drafter_profile, trace_profile)
    elif args.mtp_trace_profile is not None:
        raise SystemExit("--mtp-trace-profile requires trace or rollback mode")
    if args.mode in (ROLLBACK_MODE, STAGE_MODE):
        if args.mtp_rollback_profile is None:
            raise SystemExit(
                "--mtp-rollback-profile is required in rollback or stage mode"
            )
        rollback_profile = _load_rollback_profile(
            _existing_absolute_file(
                args.mtp_rollback_profile,
                "--mtp-rollback-profile",
            )
        )
        _require_rollback_profile_match(drafter_profile, rollback_profile)
        if args.mode == STAGE_MODE and not (
            stage_core.MINIMUM_PROMPT_INDEX
            <= args.stage_prompt_index
            <= len(rollback_profile.prompts)
        ):
            raise SystemExit("--stage-prompt-index is outside the rollback prompts")
    elif args.mtp_rollback_profile is not None:
        raise SystemExit("--mtp-rollback-profile requires rollback or stage mode")
    if model_path.name != drafter_profile.checkpoint_revision:
        raise SystemExit("model path does not match the MTP drafter checkpoint")
    if decode_profile.authority_blake3 != drafter_profile.authority_blake3:
        raise SystemExit("decode and MTP drafter authority identities differ")
    if decode_profile.checkpoint_revision != drafter_profile.checkpoint_revision:
        raise SystemExit("decode and MTP drafter checkpoint identities differ")
    if decode_profile.runtime_source_revision != drafter_profile.model_source_revision:
        raise SystemExit("decode and MTP drafter source identities differ")
    if decode_profile.tokenizer_revision != drafter_profile.tokenizer_revision:
        raise SystemExit("decode and MTP drafter tokenizer identities differ")

    stage_writer = None
    if args.mode == STAGE_MODE:
        assert progress_path is not None
        assert args.stage_prompt_index is not None
        assert args.stage_route is not None
        stage_writer = _MtpStageProgressWriter(
            path=progress_path,
            prompt_index=args.stage_prompt_index,
            route=args.stage_route,
            prewarm_target_validation=args.stage_prewarm_target_validation,
        )
        stage_writer.record(stage_core.COMMAND_START)

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
    adapter = None
    smoke_result = None
    pilot_results = None
    selection_results = None
    trace_results = None
    rollback_results = None
    prewarm_result = None
    prewarm_signature = None
    measured_signature = None
    stage_result = None
    cleanup_complete = False
    identity = MtpDrafterIdentity(
        runtime_identity=args.runtime_identity,
        package_identity=args.package_identity,
        model_identity=f"{MODEL_ALIAS}:{drafter_profile.checkpoint_revision}",
    )
    try:
        _record_stage(stage_writer, stage_core.MESH_OPEN_START)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(MESH_ROWS, MESH_COLUMNS),
            l1_small_size=L1_SMALL_SIZE_BYTES,
            num_command_queues=COMMAND_QUEUE_COUNT,
            trace_region_size=TRACE_REGION_SIZE_BYTES,
        )
        if mesh_device.get_num_devices() != len(EXPECTED_PCIE_DEVICE_IDS):
            raise MtpDrafterCommandError("P150x2 mesh did not open both devices")
        mesh_device.enable_program_cache()
        if stage_writer is not None:
            stage_writer.bind_program_cache(mesh_device)
        _record_stage(stage_writer, stage_core.MESH_OPEN_COMPLETE)
        _record_stage(stage_writer, stage_core.MODEL_LOAD_START)
        model = Qwen36Model.from_pretrained(
            mesh_device,
            max_batch_size=MAXIMUM_BATCH_SIZE,
            max_seq_len=decode_profile.maximum_sequence_length,
        )
        _record_stage(stage_writer, stage_core.MODEL_LOAD_COMPLETE)
        if len(model.layers) != EXPECTED_LAYER_COUNT:
            raise MtpDrafterCommandError(
                f"expected {EXPECTED_LAYER_COUNT} layers, got {len(model.layers)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_tokens = tokenizer(
            drafter_profile.prompts[0],
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(torch.int32)
        _record_stage(stage_writer, stage_core.ADAPTER_CREATE_START)
        decode_adapter = TracedDecodeAdapter(
            model=model,
            profile=decode_profile,
            runtime_identity=identity.runtime_identity,
            package_identity=identity.package_identity,
            model_identity=identity.model_identity,
            slot_id=DECODE_SLOT_ID,
            buffer_generation=INITIAL_BUFFER_GENERATION,
        )
        adapter = MtpDrafterAdapter(
            model=model,
            decode_adapter=decode_adapter,
            model_path=model_path,
            profile=drafter_profile,
            trace_profile=trace_profile,
            rollback_profile=rollback_profile,
        )
        _record_stage(stage_writer, stage_core.ADAPTER_CREATE_COMPLETE)
        if args.mode == SMOKE_MODE:
            smoke_result = adapter.run_smoke(
                prompt_tokens,
                selector_mode=args.selector_mode,
            )
        elif args.mode == PILOT_MODE:
            prompt_token_rows = tuple(
                tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(torch.int32)
                for prompt in drafter_profile.prompts
            )
            # Warm every measured prompt and rejection path, then discard them.
            for tokens in prompt_token_rows:
                adapter.run_pilot(tokens, selector_mode=args.selector_mode)
            pilot_results = tuple(
                adapter.run_pilot(tokens, selector_mode=args.selector_mode)
                for tokens in prompt_token_rows
            )
        elif args.mode == SELECTION_MODE:
            assert selection_profile is not None
            prompt_token_rows = tuple(
                tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(torch.int32)
                for prompt in selection_profile.prompts
            )
            for tokens in prompt_token_rows:
                adapter.run_pilot(tokens, selector_mode=FULL_LOGIT_READBACK)
                adapter.run_pilot(tokens, selector_mode=SHARD_LOCAL_TOP1)
            measured = []
            for prompt_index, tokens in enumerate(prompt_token_rows, start=1):
                for root_index, route in enumerate(
                    selection_profile.route_schedule,
                    start=1,
                ):
                    measured.append((
                        prompt_index,
                        root_index,
                        route,
                        adapter.run_pilot(tokens, selector_mode=route),
                    ))
            selection_results = tuple(measured)
        elif args.mode == TRACE_MODE:
            assert trace_profile is not None
            prompt_token_rows = tuple(
                tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(torch.int32)
                for prompt in trace_profile.prompts
            )
            for tokens in prompt_token_rows:
                adapter.run_pilot(
                    tokens,
                    selector_mode=SHARD_LOCAL_TOP1,
                    draft_execution_mode=EAGER_DRAFT,
                )
                adapter.run_pilot(
                    tokens,
                    selector_mode=SHARD_LOCAL_TOP1,
                    draft_execution_mode=TRACED_DRAFT,
                )
            measured = []
            for prompt_index, tokens in enumerate(prompt_token_rows, start=1):
                for root_index, route in enumerate(
                    trace_profile.route_schedule,
                    start=1,
                ):
                    measured.append((
                        prompt_index,
                        root_index,
                        route,
                        adapter.run_pilot(
                            tokens,
                            selector_mode=SHARD_LOCAL_TOP1,
                            draft_execution_mode=route,
                        ),
                    ))
            trace_results = tuple(measured)
        elif args.mode == ROLLBACK_MODE:
            assert trace_profile is not None
            assert rollback_profile is not None
            prompt_token_rows = tuple(
                tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).input_ids.to(torch.int32)
                for prompt in rollback_profile.prompts
            )
            for tokens in prompt_token_rows:
                adapter.run_pilot(
                    tokens,
                    selector_mode=SHARD_LOCAL_TOP1,
                    draft_execution_mode=TRACED_DRAFT,
                    mtp_recovery_mode=REPREFILL_RECOVERY,
                )
                adapter.run_pilot(
                    tokens,
                    selector_mode=SHARD_LOCAL_TOP1,
                    draft_execution_mode=TRACED_DRAFT,
                    mtp_recovery_mode=SNAPSHOT_RECOVERY,
                )
            measured = []
            for prompt_index, tokens in enumerate(prompt_token_rows, start=1):
                for root_index, route in enumerate(
                    rollback_profile.route_schedule,
                    start=1,
                ):
                    measured.append((
                        prompt_index,
                        root_index,
                        route,
                        adapter.run_pilot(
                            tokens,
                            selector_mode=SHARD_LOCAL_TOP1,
                            draft_execution_mode=TRACED_DRAFT,
                            mtp_recovery_mode=route,
                        ),
                    ))
            rollback_results = tuple(measured)
        else:
            assert args.mode == STAGE_MODE
            assert rollback_profile is not None
            assert args.stage_prompt_index is not None
            assert args.stage_route is not None
            assert stage_writer is not None
            prompt = rollback_profile.prompts[args.stage_prompt_index - 1]
            stage_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(torch.int32)
            if args.stage_prewarm_target_validation:
                stage_writer.record(stage_core.TARGET_VALIDATION_PREWARM_START)
                prewarm_result = adapter.run_pilot(
                    stage_tokens,
                    selector_mode=SHARD_LOCAL_TOP1,
                    draft_execution_mode=TRACED_DRAFT,
                    progress=stage_writer.record_prewarm,
                )
                prewarm_signature = _target_validation_signature(prewarm_result)
                stage_writer.record(stage_core.TARGET_VALIDATION_PREWARM_COMPLETE)
            stage_writer.record(stage_core.PILOT_START)
            stage_result = adapter.run_pilot(
                stage_tokens,
                selector_mode=SHARD_LOCAL_TOP1,
                draft_execution_mode=TRACED_DRAFT,
                mtp_recovery_mode=args.stage_route,
                progress=stage_writer.record,
            )
            if args.stage_prewarm_target_validation:
                assert prewarm_signature is not None
                stage_writer.record(stage_core.PREWARM_SIGNATURE_COMPARE_START)
                measured_signature = _target_validation_signature(stage_result)
                if not target_validation_signatures_match(
                    prewarm_signature,
                    measured_signature,
                ):
                    raise MtpDrafterCommandError(
                        "prewarm and measured target-validation signatures differ"
                    )
                stage_writer.record(stage_core.PREWARM_SIGNATURE_COMPARE_COMPLETE)
    except Exception:
        _record_stage(stage_writer, stage_core.FAILURE_RECORDED)
        raise
    finally:
        _record_stage(stage_writer, stage_core.CLEANUP_START)
        if adapter is not None:
            adapter.close()
            adapter = None
        if decode_adapter is not None:
            decode_adapter.close()
            decode_adapter = None
        if model is not None:
            model.free_kv_caches()
            model = None
            gc.collect()
        try:
            if mesh_device is not None:
                if stage_writer is not None:
                    stage_writer.freeze_program_cache()
                for submesh in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
                mesh_device = None
        finally:
            if stage_writer is not None:
                stage_writer.freeze_program_cache()
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        cleanup_complete = True
        _record_stage(stage_writer, stage_core.CLEANUP_COMPLETE)

    common_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "diagnostic_only": True,
        "admission_eligible": False,
        "simulated": False,
        "cleanup_complete": cleanup_complete,
        "identity": {
            "runtime": identity.runtime_identity,
            "package": identity.package_identity,
            "model": identity.model_identity,
            "authority_blake3": drafter_profile.authority_blake3,
            "checkpoint_revision": drafter_profile.checkpoint_revision,
            "model_source_revision": drafter_profile.model_source_revision,
            "tokenizer_revision": drafter_profile.tokenizer_revision,
        },
        "reference_revisions": {
            "vllm": VLLM_REFERENCE_REVISION,
            "sglang": SGLANG_REFERENCE_REVISION,
        },
        "physical_pcie_device_ids": EXPECTED_PCIE_DEVICE_IDS,
        "mesh_shape": [MESH_ROWS, MESH_COLUMNS],
        "host_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "profile_id": (
            rollback_profile.profile_id
            if rollback_profile is not None
            else (
                trace_profile.profile_id
                if trace_profile is not None
                else (
                    selection_profile.profile_id
                    if selection_profile is not None
                    else drafter_profile.profile_id
                )
            )
        ),
    }
    if args.mode == SMOKE_MODE:
        if smoke_result is None:
            raise MtpDrafterCommandError("MTP drafter smoke produced no result")
        receipt = {
            **common_receipt,
            "measurement_mode": SMOKE_MEASUREMENT_MODE,
            "smoke_only": True,
            "prompt_index": 1,
            "anchor_token_id": smoke_result.anchor_token_id,
            "draft_token_ids": list(smoke_result.draft_token_ids),
            "target_token_ids": list(smoke_result.target_token_ids),
            "accepted_prefix_length": smoke_result.accepted_prefix_length,
            "mtp_prefill_seconds": smoke_result.mtp_prefill_seconds,
            "draft_decode_seconds": smoke_result.draft_decode_seconds,
            "draft_selection_seconds": smoke_result.draft_selection_seconds,
            "selector_mode": args.selector_mode,
            "target_sequential_seconds": smoke_result.target_sequential_seconds,
            "weight_shapes": {
                name: list(shape)
                for name, shape in sorted(smoke_result.weight_shapes.items())
            },
            "mtp_selected": False,
            "blockers": [
                "five-prompt-acceptance-missing",
                "target-prefix-rollback-missing",
                "end-to-end-economics-missing",
                "production-admission-missing",
            ],
            "non_claims": [
                "One prompt does not establish local acceptance.",
                "No target GDN prefix rollback is established.",
                "No speculative speedup or production admission is established.",
            ],
        }
    elif args.mode == PILOT_MODE:
        if pilot_results is None or len(pilot_results) != len(drafter_profile.prompts):
            raise MtpDrafterCommandError(
                "MTP drafter pilot produced incomplete results"
            )
        samples = tuple(
            MtpDrafterSample(
                prompt_index=prompt_index,
                identity=identity,
                physical_pcie_device_ids=drafter_profile.expected_pcie_device_ids,
                draft_token_ids=result.draft_token_ids,
                target_token_ids=result.target_token_ids,
                accepted_prefix_length=result.accepted_prefix_length,
                mtp_prefill_seconds=result.mtp_prefill_seconds,
                draft_seconds=(
                    result.mtp_prefill_seconds + result.draft_decode_seconds
                ),
                target_sequential_seconds=result.target_sequential_seconds,
                target_verification_seconds=result.target_verification_seconds,
                target_verification_readback_seconds=(
                    result.target_verification_readback_seconds
                ),
                rollback_seconds=result.rollback_seconds,
                mtp_cache_reprefill_seconds=result.mtp_cache_reprefill_seconds,
                rollback_valid=result.rollback_valid,
                target_output_valid=result.target_output_valid,
            )
            for prompt_index, result in enumerate(pilot_results, start=1)
        )
        decision = evaluate_mtp_drafter(drafter_profile, identity, samples)
        receipt = {
            **common_receipt,
            "measurement_mode": PILOT_MEASUREMENT_MODE,
            "smoke_only": False,
            "warmup_discarded": True,
            "samples": [
                _pilot_result_mapping(prompt_index, result)
                for prompt_index, result in enumerate(pilot_results, start=1)
            ],
            "weight_shapes": {
                name: list(shape)
                for name, shape in sorted(pilot_results[0].weight_shapes.items())
            },
            "decision": decision.to_mapping(),
            "selector_mode": args.selector_mode,
            "mtp_selected": decision.mtp_selected,
            "blockers": list(decision.blockers),
            "non_claims": [
                "The pilot does not admit production behavior.",
                "Rejected-prefix target or MTP-cache rollback is not established.",
                "Trace-capture setup is diagnostic and excluded from replay economics.",
            ],
        }
    elif args.mode == SELECTION_MODE:
        assert selection_profile is not None
        expected_result_count = len(selection_profile.prompts) * len(
            selection_profile.route_schedule
        )
        if selection_results is None or len(selection_results) != expected_result_count:
            raise MtpDrafterCommandError(
                "MTP selector control produced incomplete results"
            )
        selection_samples = tuple(
            MtpSelectionSample(
                prompt_index=prompt_index,
                root_index=root_index,
                route=route,
                identity=identity,
                physical_pcie_device_ids=(selection_profile.expected_pcie_device_ids),
                draft_token_ids=result.draft_token_ids,
                target_token_ids=result.target_token_ids,
                accepted_prefix_length=result.accepted_prefix_length,
                mtp_prefill_seconds=result.mtp_prefill_seconds,
                draft_decode_seconds=result.draft_decode_seconds,
                draft_selection_seconds=result.draft_selection_seconds,
                target_sequential_seconds=result.target_sequential_seconds,
                target_verification_seconds=result.target_verification_seconds,
                target_verification_readback_seconds=(
                    result.target_verification_readback_seconds
                ),
                rollback_seconds=result.rollback_seconds,
                mtp_cache_reprefill_seconds=result.mtp_cache_reprefill_seconds,
                rollback_valid=result.rollback_valid,
                target_output_valid=result.target_output_valid,
            )
            for prompt_index, root_index, route, result in selection_results
        )
        selection_decision = evaluate_mtp_selection(
            selection_profile,
            identity,
            selection_samples,
        )
        first_result = selection_results[0][3]
        receipt = {
            **common_receipt,
            "measurement_mode": SELECTION_MEASUREMENT_MODE,
            "smoke_only": False,
            "warmup_discarded": True,
            "route_schedule": list(selection_profile.route_schedule),
            "samples": [
                _pilot_result_mapping(
                    prompt_index,
                    result,
                    root_index=root_index,
                    route=route,
                )
                for prompt_index, root_index, route, result in selection_results
            ],
            "weight_shapes": {
                name: list(shape)
                for name, shape in sorted(first_result.weight_shapes.items())
            },
            "decision": selection_decision.to_mapping(),
            "bounded_selector_selected": (selection_decision.bounded_selector_selected),
            "mtp_selected": False,
            "blockers": list(selection_decision.blockers),
            "non_claims": [
                "Selector evidence does not admit MTP or traced decode.",
                "The control does not establish sustained multi-round economics.",
                "The control does not apply to batching, TP4, or long context.",
            ],
        }
    elif args.mode == TRACE_MODE:
        assert trace_profile is not None
        expected_result_count = len(trace_profile.prompts) * len(
            trace_profile.route_schedule
        )
        if trace_results is None or len(trace_results) != expected_result_count:
            raise MtpDrafterCommandError(
                "MTP trace control produced incomplete results"
            )
        trace_samples = tuple(
            MtpTraceSample(
                prompt_index=prompt_index,
                root_index=root_index,
                route=route,
                identity=identity,
                physical_pcie_device_ids=trace_profile.expected_pcie_device_ids,
                draft_token_ids=result.draft_token_ids,
                target_token_ids=result.target_token_ids,
                accepted_prefix_length=result.accepted_prefix_length,
                mtp_prefill_seconds=result.mtp_prefill_seconds,
                draft_decode_seconds=result.draft_decode_seconds,
                draft_selection_seconds=result.draft_selection_seconds,
                trace_capture_seconds=result.draft_trace_capture_seconds,
                target_sequential_seconds=result.target_sequential_seconds,
                target_verification_seconds=result.target_verification_seconds,
                target_verification_readback_seconds=(
                    result.target_verification_readback_seconds
                ),
                rollback_seconds=result.rollback_seconds,
                mtp_cache_recovery_seconds=result.mtp_cache_reprefill_seconds,
                trace_binding_valid=result.trace_binding_valid,
                capture_idempotence_valid=result.capture_idempotence_valid,
                mtp_kv_prefix_valid=result.mtp_kv_prefix_valid,
                rollback_valid=result.rollback_valid,
                target_output_valid=result.target_output_valid,
            )
            for prompt_index, root_index, route, result in trace_results
        )
        trace_decision = evaluate_mtp_trace(trace_profile, identity, trace_samples)
        first_result = trace_results[0][3]
        receipt = {
            **common_receipt,
            "measurement_mode": TRACE_MEASUREMENT_MODE,
            "smoke_only": False,
            "warmup_discarded": True,
            "selector_mode": SHARD_LOCAL_TOP1,
            "route_schedule": list(trace_profile.route_schedule),
            "minimum_samples_per_route": (trace_profile.minimum_samples_per_route),
            "samples": [
                _pilot_result_mapping(
                    prompt_index,
                    result,
                    root_index=root_index,
                    route=route,
                )
                for prompt_index, root_index, route, result in trace_results
            ],
            "weight_shapes": {
                name: list(shape)
                for name, shape in sorted(first_result.weight_shapes.items())
            },
            "decision": trace_decision.to_mapping(),
            "traced_drafting_selected": trace_decision.traced_drafting_selected,
            "mtp_selected": False,
            "blockers": list(trace_decision.blockers),
            "non_claims": [
                "Trace evidence does not admit MTP or traced target decode.",
                "Capture time is recorded and excluded from replay economics.",
                "The control does not establish rejected-row overwrite or multi-round economics.",
            ],
        }
    elif args.mode == ROLLBACK_MODE:
        assert rollback_profile is not None
        expected_result_count = len(rollback_profile.prompts) * len(
            rollback_profile.route_schedule
        )
        if rollback_results is None or len(rollback_results) != expected_result_count:
            raise MtpDrafterCommandError(
                "MTP rollback control produced incomplete results"
            )
        reference_states = {}
        for prompt_index, _root_index, route, result in rollback_results:
            if route != REPREFILL_RECOVERY or prompt_index in reference_states:
                continue
            if result.mtp_cache_recovery_state is None:
                raise MtpDrafterCommandError("MTP rollback reference state is missing")
            reference_states[prompt_index] = result.mtp_cache_recovery_state
        if len(reference_states) != len(rollback_profile.prompts):
            raise MtpDrafterCommandError("MTP rollback references are incomplete")

        rollback_samples = []
        sample_mappings = []
        for prompt_index, root_index, route, result in rollback_results:
            state = result.mtp_cache_recovery_state
            if state is None:
                raise MtpDrafterCommandError("MTP rollback state is missing")
            token_valid, feedback_valid, kv_valid = recovery_state_components_equal(
                reference_states[prompt_index],
                state,
            )
            committed_token_id = result.target_verification_token_ids[
                result.accepted_prefix_length
            ]
            rollback_samples.append(
                MtpRollbackSample(
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                    identity=identity,
                    physical_pcie_device_ids=(
                        rollback_profile.expected_pcie_device_ids
                    ),
                    draft_token_ids=result.draft_token_ids,
                    target_token_ids=result.target_token_ids,
                    accepted_prefix_length=result.accepted_prefix_length,
                    committed_token_id=committed_token_id,
                    next_draft_token_id=state.next_draft_token_id,
                    recovery_seconds=result.mtp_cache_recovery_seconds,
                    recovery_execution_mode=(
                        result.mtp_cache_recovery_execution_mode or ""
                    ),
                    process_trace_capture_count=(
                        result.mtp_cache_process_trace_capture_count
                    ),
                    trace_binding_identity=(
                        json.dumps(
                            result.draft_trace_binding,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if result.draft_trace_binding is not None
                        else ""
                    ),
                    trace_binding_valid=result.mtp_cache_trace_binding_valid,
                    snapshot_row_count=result.mtp_cache_snapshot_row_count,
                    committed_advance_count=(result.mtp_cache_committed_advance_count),
                    next_draft_valid=token_valid,
                    feedback_state_valid=feedback_valid,
                    kv_prefix_valid=kv_valid,
                    target_output_valid=result.target_output_valid,
                    target_rollback_valid=result.rollback_valid,
                )
            )
            sample_mappings.append({
                **_pilot_result_mapping(
                    prompt_index,
                    result,
                    root_index=root_index,
                    route=route,
                ),
                "committed_token_id": committed_token_id,
                "next_draft_token_id": state.next_draft_token_id,
                "mtp_cache_recovery_seconds": (result.mtp_cache_recovery_seconds),
                "mtp_cache_queue_drain_seconds": (
                    result.mtp_cache_queue_drain_seconds
                ),
                "mtp_cache_recovery_execution_mode": (
                    result.mtp_cache_recovery_execution_mode
                ),
                "mtp_cache_process_trace_capture_count": (
                    result.mtp_cache_process_trace_capture_count
                ),
                "mtp_cache_trace_binding_valid": (
                    result.mtp_cache_trace_binding_valid
                ),
                "mtp_cache_snapshot_row_count": (result.mtp_cache_snapshot_row_count),
                "mtp_cache_committed_advance_count": (
                    result.mtp_cache_committed_advance_count
                ),
                "next_draft_valid": token_valid,
                "feedback_state_valid": feedback_valid,
                "kv_prefix_valid": kv_valid,
            })
        rollback_decision = evaluate_mtp_rollback(
            rollback_profile,
            identity,
            tuple(rollback_samples),
        )
        first_result = rollback_results[0][3]
        receipt = {
            **common_receipt,
            "measurement_mode": ROLLBACK_MEASUREMENT_MODE,
            "smoke_only": False,
            "warmup_discarded": True,
            "selector_mode": SHARD_LOCAL_TOP1,
            "draft_execution_mode": TRACED_DRAFT,
            "recovery_execution_mode": rollback_profile.recovery_execution_mode,
            "process_trace_capture_count": (
                first_result.mtp_cache_process_trace_capture_count
            ),
            "route_schedule": list(rollback_profile.route_schedule),
            "minimum_samples_per_route": (rollback_profile.minimum_samples_per_route),
            "samples": sample_mappings,
            "weight_shapes": {
                name: list(shape)
                for name, shape in sorted(first_result.weight_shapes.items())
            },
            "decision": rollback_decision.to_mapping(),
            "snapshot_rollback_selected": (
                rollback_decision.snapshot_rollback_selected
            ),
            "mtp_selected": False,
            "blockers": list(rollback_decision.blockers),
            "non_claims": [
                "Rollback evidence does not admit MTP or traced target decode.",
                "The control covers natural acceptance lengths zero and three only.",
                "The control does not establish sustained multi-round economics.",
            ],
        }
    else:
        assert args.mode == STAGE_MODE
        assert stage_result is not None
        assert stage_writer is not None
        assert progress_path is not None
        assert args.stage_prompt_index is not None
        assert args.stage_route is not None
        receipt = {
            **common_receipt,
            "measurement_mode": STAGE_MEASUREMENT_MODE,
            "smoke_only": True,
            "warmup_discarded": args.stage_prewarm_target_validation,
            "target_validation_prewarm": args.stage_prewarm_target_validation,
            "selector_mode": SHARD_LOCAL_TOP1,
            "draft_execution_mode": TRACED_DRAFT,
            "prompt_index": args.stage_prompt_index,
            "route": args.stage_route,
            "sample": _pilot_result_mapping(
                args.stage_prompt_index,
                stage_result,
                root_index=stage_core.MINIMUM_PROMPT_INDEX,
                route=args.stage_route,
            ),
            "progress_output": str(progress_path),
            "weight_shapes": {
                name: list(shape)
                for name, shape in sorted(stage_result.weight_shapes.items())
            },
            "mtp_selected": False,
            "blockers": [
                "single-stage-diagnostic",
                "matched-roots-missing",
                "production-admission-missing",
            ],
            "non_claims": [
                "Stage completion does not establish timing stability.",
                "One prompt and route do not establish rollback selection.",
                "Staged evidence does not admit MTP or traced target decode.",
            ],
        }
        if args.stage_prewarm_target_validation:
            assert prewarm_result is not None
            assert prewarm_signature is not None
            assert measured_signature is not None
            receipt.update({
                "prewarm_sample": _pilot_result_mapping(
                    args.stage_prompt_index,
                    prewarm_result,
                ),
                "prewarm_signature": prewarm_signature.to_mapping(),
                "measured_signature": measured_signature.to_mapping(),
                "prewarm_signature_match": True,
            })
    _record_stage(stage_writer, stage_core.RECEIPT_WRITE_START)
    if stage_writer is not None:
        receipt["progress"] = stage_writer.progress.to_mapping()
    _write_receipt(output_path, receipt)
    _record_stage(stage_writer, stage_core.COMPLETE)
    print(json.dumps(receipt, sort_keys=True))


def _target_validation_signature(
    result: MtpDrafterPilotResult,
) -> MtpTargetValidationSignature:
    return make_target_validation_signature(
        draft_token_ids=result.draft_token_ids,
        target_token_ids=result.target_token_ids,
        target_verification_token_ids=result.target_verification_token_ids,
        accepted_prefix_length=result.accepted_prefix_length,
        rollback_applied=result.rollback_applied,
        target_output_valid=result.target_output_valid,
        rollback_valid=result.rollback_valid,
    )


def _pilot_result_mapping(
    prompt_index: int,
    result: MtpDrafterPilotResult,
    *,
    root_index: int | None = None,
    route: str | None = None,
) -> dict[str, object]:
    if (root_index is None) != (route is None):
        raise MtpDrafterCommandError("paired root metadata is incomplete")
    total_draft_seconds = result.mtp_prefill_seconds + result.draft_decode_seconds
    mapping: dict[str, object] = {
        "prompt_index": prompt_index,
        "anchor_token_id": result.anchor_token_id,
        "draft_token_ids": list(result.draft_token_ids),
        "target_token_ids": list(result.target_token_ids),
        "target_continuation_token_id": result.target_continuation_token_id,
        "target_verification_token_ids": list(result.target_verification_token_ids),
        "accepted_prefix_length": result.accepted_prefix_length,
        "mtp_prefill_seconds": result.mtp_prefill_seconds,
        "draft_decode_seconds": result.draft_decode_seconds,
        "draft_selection_seconds": result.draft_selection_seconds,
        "draft_compute_seconds": total_draft_seconds - result.draft_selection_seconds,
        "total_draft_seconds": total_draft_seconds,
        "draft_execution_mode": result.draft_execution_mode,
        "draft_trace_capture_seconds": result.draft_trace_capture_seconds,
        "draft_trace_binding": result.draft_trace_binding,
        "trace_binding_valid": result.trace_binding_valid,
        "capture_idempotence_valid": result.capture_idempotence_valid,
        "mtp_kv_prefix_valid": result.mtp_kv_prefix_valid,
        "target_sequential_seconds": result.target_sequential_seconds,
        "target_verification_trace_capture_seconds": (
            result.target_verification_trace_capture_seconds
        ),
        "target_verification_seconds": result.target_verification_seconds,
        "target_verification_readback_seconds": (
            result.target_verification_readback_seconds
        ),
        "target_output_valid": result.target_output_valid,
        "rollback_applied": result.rollback_applied,
        "rollback_valid": result.rollback_valid,
        "rollback_seconds": result.rollback_seconds,
        "mtp_cache_reprefill_seconds": result.mtp_cache_reprefill_seconds,
        "mtp_cache_recovery_seconds": result.mtp_cache_recovery_seconds,
        "mtp_cache_queue_drain_seconds": result.mtp_cache_queue_drain_seconds,
        "mtp_cache_recovery_execution_mode": (
            result.mtp_cache_recovery_execution_mode
        ),
        "mtp_cache_process_trace_capture_count": (
            result.mtp_cache_process_trace_capture_count
        ),
        "mtp_cache_trace_binding_valid": result.mtp_cache_trace_binding_valid,
        "rollback_recurrent_state_pcc": result.rollback_recurrent_state_pcc,
        "rollback_recurrent_state_nmse": result.rollback_recurrent_state_nmse,
        "rollback_convolution_state_pcc": result.rollback_convolution_state_pcc,
        "rollback_convolution_state_nmse": result.rollback_convolution_state_nmse,
        "rollback_paged_kv_state_pcc": result.rollback_paged_kv_state_pcc,
        "rollback_paged_kv_state_nmse": result.rollback_paged_kv_state_nmse,
    }
    if root_index is not None and route is not None:
        mapping["root_index"] = root_index
        mapping["route"] = route
    return mapping


def _load_decode_profile(path: Path) -> DecodeProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return DecodeProfile.from_mapping(json.load(profile_file))
    except (DecodePolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"decode profile is invalid: {error}") from error


def _load_drafter_profile(path: Path) -> MtpDrafterProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return MtpDrafterProfile.from_mapping(json.load(profile_file))
    except (MtpDrafterPolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"MTP drafter profile is invalid: {error}") from error


def _load_selection_profile(path: Path) -> MtpSelectionProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return MtpSelectionProfile.from_mapping(json.load(profile_file))
    except (MtpSelectionPolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"MTP selection profile is invalid: {error}") from error


def _load_trace_profile(path: Path) -> MtpTraceProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return MtpTraceProfile.from_mapping(json.load(profile_file))
    except (MtpTracePolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"MTP trace profile is invalid: {error}") from error


def _load_rollback_profile(path: Path) -> MtpRollbackProfile:
    try:
        with path.open(encoding="utf-8") as profile_file:
            return MtpRollbackProfile.from_mapping(json.load(profile_file))
    except (MtpRollbackPolicyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"MTP rollback profile is invalid: {error}") from error


def _require_selection_profile_match(
    drafter: MtpDrafterProfile,
    selection: MtpSelectionProfile,
) -> None:
    comparisons = (
        (drafter.authority_blake3, selection.authority_blake3, "authority"),
        (drafter.checkpoint_revision, selection.checkpoint_revision, "checkpoint"),
        (drafter.model_source_revision, selection.model_source_revision, "source"),
        (drafter.tokenizer_revision, selection.tokenizer_revision, "tokenizer"),
        (
            drafter.expected_pcie_device_ids,
            selection.expected_pcie_device_ids,
            "physical devices",
        ),
        (drafter.draft_token_count, selection.draft_token_count, "draft count"),
        (drafter.prompts, selection.prompts, "prompts"),
    )
    for drafter_value, selection_value, label in comparisons:
        if drafter_value != selection_value:
            raise SystemExit(f"MTP drafter and selection {label} differ")


def _require_trace_profile_match(
    drafter: MtpDrafterProfile,
    trace: MtpTraceProfile,
) -> None:
    comparisons = (
        (drafter.authority_blake3, trace.authority_blake3, "authority"),
        (drafter.checkpoint_revision, trace.checkpoint_revision, "checkpoint"),
        (drafter.model_source_revision, trace.model_source_revision, "source"),
        (drafter.tokenizer_revision, trace.tokenizer_revision, "tokenizer"),
        (
            drafter.expected_pcie_device_ids,
            trace.expected_pcie_device_ids,
            "physical devices",
        ),
        (drafter.draft_token_count, trace.draft_token_count, "draft count"),
        (drafter.prompts, trace.prompts, "prompts"),
    )
    for drafter_value, trace_value, label in comparisons:
        if drafter_value != trace_value:
            raise SystemExit(f"MTP drafter and trace {label} differ")


def _require_rollback_profile_match(
    drafter: MtpDrafterProfile,
    rollback: MtpRollbackProfile,
) -> None:
    comparisons = (
        (drafter.authority_blake3, rollback.authority_blake3, "authority"),
        (drafter.checkpoint_revision, rollback.checkpoint_revision, "checkpoint"),
        (drafter.model_source_revision, rollback.model_source_revision, "source"),
        (drafter.tokenizer_revision, rollback.tokenizer_revision, "tokenizer"),
        (
            drafter.expected_pcie_device_ids,
            rollback.expected_pcie_device_ids,
            "physical devices",
        ),
        (drafter.draft_token_count, rollback.draft_token_count, "draft count"),
        (drafter.prompts, rollback.prompts, "prompts"),
    )
    for drafter_value, rollback_value, label in comparisons:
        if drafter_value != rollback_value:
            raise SystemExit(f"MTP drafter and rollback {label} differ")


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


def _record_stage(
    writer: _MtpStageProgressWriter | None,
    stage: str,
) -> None:
    if writer is not None:
        writer.record(stage)


def _replace_json(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    with temporary_path.open("x", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("x", encoding="utf-8") as receipt_file:
        json.dump(receipt, receipt_file, indent=2, sort_keys=True)
        receipt_file.write("\n")
        receipt_file.flush()
        os.fsync(receipt_file.fileno())
    os.replace(temporary_path, path)


if __name__ == "__main__":
    main()
