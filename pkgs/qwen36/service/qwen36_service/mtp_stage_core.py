"""Pure ordered progress state for the staged Qwen MTP recovery probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_recovery_stage_probe]
# r[impl tenstorrent.qwen38_p150x2.mtp_target_validation_prewarm]

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .mtp_rollback_core import REPREFILL_RECOVERY, SNAPSHOT_RECOVERY

STAGE_SCHEMA_VERSION = 2
MINIMUM_PROMPT_INDEX = 1
MINIMUM_PROGRAM_CACHE_ENTRIES = 0

COMMAND_START = "command-start"
MESH_OPEN_START = "mesh-open-start"
MESH_OPEN_COMPLETE = "mesh-open-complete"
MODEL_LOAD_START = "model-load-start"
MODEL_LOAD_COMPLETE = "model-load-complete"
ADAPTER_CREATE_START = "adapter-create-start"
ADAPTER_CREATE_COMPLETE = "adapter-create-complete"
TARGET_VALIDATION_PREWARM_START = "target-validation-prewarm-start"
TARGET_VALIDATION_PREWARM_COMPLETE = "target-validation-prewarm-complete"
TARGET_VALIDATION_PREWARM_STAGE_PREFIX = "target-validation-prewarm-"
PILOT_START = "pilot-start"
TARGET_PREFILL_START = "target-prefill-start"
TARGET_PREFILL_COMPLETE = "target-prefill-complete"
DRAFT_START = "draft-start"
MTP_PREFILL_START = "mtp-prefill-start"
MTP_PREFILL_COMPLETE = "mtp-prefill-complete"
ROLLBACK_CONTEXT_START = "rollback-context-start"
ROLLBACK_CONTEXT_COMPLETE = "rollback-context-complete"
TRACE_RUN_START = "trace-run-start"
TRACE_INPUT_ALLOCATION_START = "trace-input-allocation-start"
TRACE_INPUT_ALLOCATION_COMPLETE = "trace-input-allocation-complete"
TRACE_CAPTURE_START = "trace-capture-start"
TRACE_CAPTURE_COMPLETE = "trace-capture-complete"
TRACE_DRAFT_REPLAY_STEP_1_START = "trace-draft-replay-step-1-start"
TRACE_DRAFT_REPLAY_STEP_1_COMPLETE = "trace-draft-replay-step-1-complete"
TRACE_DRAFT_REPLAY_STEP_2_START = "trace-draft-replay-step-2-start"
TRACE_DRAFT_REPLAY_STEP_2_COMPLETE = "trace-draft-replay-step-2-complete"
TRACE_DRAFT_REPLAY_STEP_3_START = "trace-draft-replay-step-3-start"
TRACE_DRAFT_REPLAY_STEP_3_COMPLETE = "trace-draft-replay-step-3-complete"
TRACE_DRAFT_REPLAY_STEP_4_START = "trace-draft-replay-step-4-start"
TRACE_DRAFT_REPLAY_STEP_4_COMPLETE = "trace-draft-replay-step-4-complete"
TRACE_DRAFT_REPLAY_STEP_5_START = "trace-draft-replay-step-5-start"
TRACE_DRAFT_REPLAY_STEP_5_COMPLETE = "trace-draft-replay-step-5-complete"
TRACE_RUN_COMPLETE = "trace-run-complete"
DRAFT_COMPLETE = "draft-complete"
TARGET_TRACE_START = "target-trace-start"
TARGET_TRACE_COMPLETE = "target-trace-complete"
TARGET_SEQUENTIAL_START = "target-sequential-start"
TARGET_SEQUENTIAL_COMPLETE = "target-sequential-complete"
TARGET_TRACE_RELEASE_COMPLETE = "target-trace-release-complete"
TARGET_REFERENCE_STATE_START = "target-reference-state-start"
TARGET_REFERENCE_STATE_COMPLETE = "target-reference-state-complete"
VERIFICATION_PREFILL_START = "verification-prefill-start"
VERIFICATION_PREFILL_COMPLETE = "verification-prefill-complete"
RECOVERY_START = "recovery-start"
RECOVERY_QUEUE_DRAIN_START = "recovery-queue-drain-start"
RECOVERY_QUEUE_DRAIN_COMPLETE = "recovery-queue-drain-complete"
RECOVERY_RESTORE_START = "recovery-restore-start"
RECOVERY_RESTORE_COMPLETE = "recovery-restore-complete"
RECOVERY_MTP_PREFILL_START = "recovery-mtp-prefill-start"
RECOVERY_MTP_PREFILL_COMPLETE = "recovery-mtp-prefill-complete"
CONTROLLED_REPLAY_START = "controlled-replay-start"
CONTROLLED_REPLAY_STEP_1_START = "controlled-replay-step-1-start"
CONTROLLED_REPLAY_STEP_1_COMPLETE = "controlled-replay-step-1-complete"
CONTROLLED_REPLAY_STEP_2_START = "controlled-replay-step-2-start"
CONTROLLED_REPLAY_STEP_2_COMPLETE = "controlled-replay-step-2-complete"
CONTROLLED_REPLAY_STEP_3_START = "controlled-replay-step-3-start"
CONTROLLED_REPLAY_STEP_3_COMPLETE = "controlled-replay-step-3-complete"
CONTROLLED_REPLAY_STEP_4_START = "controlled-replay-step-4-start"
CONTROLLED_REPLAY_STEP_4_COMPLETE = "controlled-replay-step-4-complete"
CONTROLLED_REPLAY_COMPLETE = "controlled-replay-complete"
RECOVERY_SYNCHRONIZE_START = "recovery-synchronize-start"
RECOVERY_SYNCHRONIZE_COMPLETE = "recovery-synchronize-complete"
RECOVERY_STATE_READBACK_START = "recovery-state-readback-start"
RECOVERY_STATE_READBACK_COMPLETE = "recovery-state-readback-complete"
RECOVERY_COMPLETE = "recovery-complete"
NATIVE_VERIFIER_START = "native-verifier-start"
NATIVE_VERIFIER_COMPLETE = "native-verifier-complete"
TARGET_STATE_COMPARE_START = "target-state-compare-start"
TARGET_STATE_COMPARE_COMPLETE = "target-state-compare-complete"
PILOT_COMPLETE = "pilot-complete"
PREWARM_SIGNATURE_COMPARE_START = "target-prewarm-signature-compare-start"
PREWARM_SIGNATURE_COMPARE_COMPLETE = "target-prewarm-signature-compare-complete"
FAILURE_RECORDED = "failure-recorded"
CLEANUP_START = "cleanup-start"
CLEANUP_COMPLETE = "cleanup-complete"
RECEIPT_WRITE_START = "receipt-write-start"
COMPLETE = "complete"

TRACE_DRAFT_REPLAY_STEP_START_STAGES = (
    TRACE_DRAFT_REPLAY_STEP_1_START,
    TRACE_DRAFT_REPLAY_STEP_2_START,
    TRACE_DRAFT_REPLAY_STEP_3_START,
    TRACE_DRAFT_REPLAY_STEP_4_START,
    TRACE_DRAFT_REPLAY_STEP_5_START,
)
TRACE_DRAFT_REPLAY_STEP_COMPLETE_STAGES = (
    TRACE_DRAFT_REPLAY_STEP_1_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_2_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_3_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_4_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_5_COMPLETE,
)
CONTROLLED_REPLAY_STEP_START_STAGES = (
    CONTROLLED_REPLAY_STEP_1_START,
    CONTROLLED_REPLAY_STEP_2_START,
    CONTROLLED_REPLAY_STEP_3_START,
    CONTROLLED_REPLAY_STEP_4_START,
)
CONTROLLED_REPLAY_STEP_COMPLETE_STAGES = (
    CONTROLLED_REPLAY_STEP_1_COMPLETE,
    CONTROLLED_REPLAY_STEP_2_COMPLETE,
    CONTROLLED_REPLAY_STEP_3_COMPLETE,
    CONTROLLED_REPLAY_STEP_4_COMPLETE,
)

_BOOTSTRAP_STAGES = (
    COMMAND_START,
    MESH_OPEN_START,
    MESH_OPEN_COMPLETE,
    MODEL_LOAD_START,
    MODEL_LOAD_COMPLETE,
    ADAPTER_CREATE_START,
    ADAPTER_CREATE_COMPLETE,
)
_PILOT_PREFIX = (
    PILOT_START,
    TARGET_PREFILL_START,
    TARGET_PREFILL_COMPLETE,
    DRAFT_START,
    MTP_PREFILL_START,
    MTP_PREFILL_COMPLETE,
)
_TRACE_STAGES = (
    TRACE_RUN_START,
    TRACE_INPUT_ALLOCATION_START,
    TRACE_INPUT_ALLOCATION_COMPLETE,
    TRACE_CAPTURE_START,
    TRACE_CAPTURE_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_1_START,
    TRACE_DRAFT_REPLAY_STEP_1_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_2_START,
    TRACE_DRAFT_REPLAY_STEP_2_COMPLETE,
    TRACE_RUN_COMPLETE,
    DRAFT_COMPLETE,
    TARGET_TRACE_START,
    TARGET_TRACE_COMPLETE,
    TARGET_SEQUENTIAL_START,
    TARGET_SEQUENTIAL_COMPLETE,
    TARGET_TRACE_RELEASE_COMPLETE,
)
_CONTROLLED_REPLAY_STAGES = (
    CONTROLLED_REPLAY_START,
    CONTROLLED_REPLAY_STEP_1_START,
    CONTROLLED_REPLAY_STEP_1_COMPLETE,
    CONTROLLED_REPLAY_STEP_2_START,
    CONTROLLED_REPLAY_STEP_2_COMPLETE,
    CONTROLLED_REPLAY_STEP_3_START,
    CONTROLLED_REPLAY_STEP_3_COMPLETE,
    CONTROLLED_REPLAY_STEP_4_START,
    CONTROLLED_REPLAY_STEP_4_COMPLETE,
    CONTROLLED_REPLAY_COMPLETE,
)
_RECOVERY_TAIL = (
    RECOVERY_SYNCHRONIZE_START,
    RECOVERY_SYNCHRONIZE_COMPLETE,
    RECOVERY_STATE_READBACK_START,
    RECOVERY_STATE_READBACK_COMPLETE,
    RECOVERY_COMPLETE,
)
_VALIDATION_TAIL = (
    NATIVE_VERIFIER_START,
    NATIVE_VERIFIER_COMPLETE,
    TARGET_STATE_COMPARE_START,
    TARGET_STATE_COMPARE_COMPLETE,
    PILOT_COMPLETE,
)
_PROCESS_TAIL = (
    FAILURE_RECORDED,
    CLEANUP_START,
    CLEANUP_COMPLETE,
    RECEIPT_WRITE_START,
    COMPLETE,
)
_REPREFILL_PILOT_ORDER = (
    *_PILOT_PREFIX,
    *_TRACE_STAGES,
    TARGET_REFERENCE_STATE_START,
    TARGET_REFERENCE_STATE_COMPLETE,
    VERIFICATION_PREFILL_START,
    VERIFICATION_PREFILL_COMPLETE,
    RECOVERY_START,
    RECOVERY_QUEUE_DRAIN_START,
    RECOVERY_QUEUE_DRAIN_COMPLETE,
    RECOVERY_MTP_PREFILL_START,
    RECOVERY_MTP_PREFILL_COMPLETE,
    *_CONTROLLED_REPLAY_STAGES,
    *_RECOVERY_TAIL,
    *_VALIDATION_TAIL,
)
_SNAPSHOT_PILOT_ORDER = (
    *_PILOT_PREFIX,
    ROLLBACK_CONTEXT_START,
    ROLLBACK_CONTEXT_COMPLETE,
    *_TRACE_STAGES,
    RECOVERY_START,
    RECOVERY_QUEUE_DRAIN_START,
    RECOVERY_QUEUE_DRAIN_COMPLETE,
    RECOVERY_RESTORE_START,
    RECOVERY_RESTORE_COMPLETE,
    *_CONTROLLED_REPLAY_STAGES,
    *_RECOVERY_TAIL,
    TARGET_REFERENCE_STATE_START,
    TARGET_REFERENCE_STATE_COMPLETE,
    VERIFICATION_PREFILL_START,
    VERIFICATION_PREFILL_COMPLETE,
    *_VALIDATION_TAIL,
)
_PREWARM_SOURCE_STAGES = (
    TARGET_PREFILL_START,
    TARGET_PREFILL_COMPLETE,
    DRAFT_START,
    MTP_PREFILL_START,
    MTP_PREFILL_COMPLETE,
    *_TRACE_STAGES,
    TARGET_REFERENCE_STATE_START,
    TARGET_REFERENCE_STATE_COMPLETE,
    VERIFICATION_PREFILL_START,
    VERIFICATION_PREFILL_COMPLETE,
    *_VALIDATION_TAIL,
)
_PREWARM_SOURCE_STAGE_SET = frozenset(_PREWARM_SOURCE_STAGES)
_PREWARM_STAGE_ORDER = tuple(
    TARGET_VALIDATION_PREWARM_STAGE_PREFIX + stage for stage in _PREWARM_SOURCE_STAGES
)
_PILOT_ORDER_BY_ROUTE = {
    REPREFILL_RECOVERY: _REPREFILL_PILOT_ORDER,
    SNAPSHOT_RECOVERY: _SNAPSHOT_PILOT_ORDER,
}
_STAGE_ORDER_BY_ROUTE_AND_PREWARM = {
    (route, False): (*_BOOTSTRAP_STAGES, *pilot_order, *_PROCESS_TAIL)
    for route, pilot_order in _PILOT_ORDER_BY_ROUTE.items()
}
_STAGE_ORDER_BY_ROUTE_AND_PREWARM.update({
    (route, True): (
        *_BOOTSTRAP_STAGES,
        TARGET_VALIDATION_PREWARM_START,
        *_PREWARM_STAGE_ORDER,
        TARGET_VALIDATION_PREWARM_COMPLETE,
        *pilot_order,
        PREWARM_SIGNATURE_COMPARE_START,
        PREWARM_SIGNATURE_COMPARE_COMPLETE,
        *_PROCESS_TAIL,
    )
    for route, pilot_order in _PILOT_ORDER_BY_ROUTE.items()
})
_STAGE_INDEX_BY_ROUTE_AND_PREWARM = {
    key: {stage: index for index, stage in enumerate(stages)}
    for key, stages in _STAGE_ORDER_BY_ROUTE_AND_PREWARM.items()
}


class MtpStageProgressError(ValueError):
    """The staged progress ledger is invalid or ambiguous."""


@dataclass(frozen=True)
class MtpStageEvent:
    """One completed host boundary and its elapsed process time."""

    stage: str
    elapsed_seconds: float
    program_cache_entries: int | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_seconds": self.elapsed_seconds,
            "program_cache_entries": self.program_cache_entries,
        }


@dataclass(frozen=True)
class MtpStageProgress:
    """Immutable ordered progress for one prompt and recovery route."""

    prompt_index: int
    route: str
    prewarm_target_validation: bool
    events: tuple[MtpStageEvent, ...] = ()

    @property
    def last_stage(self) -> str | None:
        return self.events[-1].stage if self.events else None

    @property
    def complete(self) -> bool:
        return self.last_stage == COMPLETE

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "admission_eligible": False,
            "prompt_index": self.prompt_index,
            "route": self.route,
            "prewarm_target_validation": self.prewarm_target_validation,
            "last_stage": self.last_stage,
            "complete": self.complete,
            "events": [event.to_mapping() for event in self.events],
        }


def target_validation_prewarm_stage(stage: str) -> str:
    """Map one known pilot boundary into the prewarm namespace."""
    if stage not in _PREWARM_SOURCE_STAGE_SET:
        raise MtpStageProgressError("MTP target prewarm boundary is unsupported")
    return TARGET_VALIDATION_PREWARM_STAGE_PREFIX + stage


def new_mtp_stage_progress(
    *,
    prompt_index: int,
    route: str,
    prewarm_target_validation: bool = False,
) -> MtpStageProgress:
    """Create an empty, validated staged progress value."""
    if isinstance(prompt_index, bool) or prompt_index < MINIMUM_PROMPT_INDEX:
        raise MtpStageProgressError("MTP stage prompt index must be positive")
    if route not in _PILOT_ORDER_BY_ROUTE:
        raise MtpStageProgressError("MTP stage recovery route is unsupported")
    if not isinstance(prewarm_target_validation, bool):
        raise MtpStageProgressError("MTP stage prewarm flag must be boolean")
    return MtpStageProgress(
        prompt_index=prompt_index,
        route=route,
        prewarm_target_validation=prewarm_target_validation,
    )


def advance_mtp_stage_progress(
    progress: MtpStageProgress,
    *,
    stage: str,
    elapsed_seconds: float,
    program_cache_entries: int | None = None,
) -> MtpStageProgress:
    """Return progress advanced to one later known boundary."""
    if progress.complete:
        raise MtpStageProgressError("MTP stage progress is already complete")
    stage_indices = _STAGE_INDEX_BY_ROUTE_AND_PREWARM.get((
        progress.route,
        progress.prewarm_target_validation,
    ))
    if stage_indices is None:
        raise MtpStageProgressError("MTP stage recovery route is unsupported")
    if stage not in stage_indices:
        raise MtpStageProgressError(
            "MTP stage boundary is unknown or route-incompatible"
        )
    if (
        isinstance(elapsed_seconds, bool)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
    ):
        raise MtpStageProgressError("MTP stage elapsed time is invalid")
    if program_cache_entries is not None and (
        isinstance(program_cache_entries, bool)
        or not isinstance(program_cache_entries, int)
        or program_cache_entries < MINIMUM_PROGRAM_CACHE_ENTRIES
    ):
        raise MtpStageProgressError("MTP stage program-cache count is invalid")
    if progress.events:
        previous = progress.events[-1]
        if elapsed_seconds < previous.elapsed_seconds:
            raise MtpStageProgressError("MTP stage elapsed time regressed")
        if stage_indices[stage] <= stage_indices[previous.stage]:
            raise MtpStageProgressError("MTP stage boundary repeated or regressed")
        if previous.program_cache_entries is not None:
            if program_cache_entries is None:
                raise MtpStageProgressError(
                    "MTP stage program-cache observation disappeared"
                )
            if program_cache_entries < previous.program_cache_entries:
                raise MtpStageProgressError("MTP stage program-cache count regressed")
    if stage == COMPLETE and progress.last_stage != RECEIPT_WRITE_START:
        raise MtpStageProgressError("MTP stage completed before receipt writing")
    event = MtpStageEvent(
        stage=stage,
        elapsed_seconds=elapsed_seconds,
        program_cache_entries=program_cache_entries,
    )
    return replace(progress, events=(*progress.events, event))
