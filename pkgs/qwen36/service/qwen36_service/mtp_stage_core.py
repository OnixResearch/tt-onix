"""Pure ordered progress state for the staged Qwen MTP recovery probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_recovery_stage_probe]

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .mtp_rollback_core import REPREFILL_RECOVERY, SNAPSHOT_RECOVERY

STAGE_SCHEMA_VERSION = 1
MINIMUM_PROMPT_INDEX = 1

COMMAND_START = "command-start"
MESH_OPEN_START = "mesh-open-start"
MESH_OPEN_COMPLETE = "mesh-open-complete"
MODEL_LOAD_START = "model-load-start"
MODEL_LOAD_COMPLETE = "model-load-complete"
ADAPTER_CREATE_START = "adapter-create-start"
ADAPTER_CREATE_COMPLETE = "adapter-create-complete"
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
FAILURE_RECORDED = "failure-recorded"
CLEANUP_START = "cleanup-start"
CLEANUP_COMPLETE = "cleanup-complete"
RECEIPT_WRITE_START = "receipt-write-start"
COMPLETE = "complete"

TRACE_DRAFT_REPLAY_STEP_START_STAGES = (
    TRACE_DRAFT_REPLAY_STEP_1_START,
    TRACE_DRAFT_REPLAY_STEP_2_START,
)
TRACE_DRAFT_REPLAY_STEP_COMPLETE_STAGES = (
    TRACE_DRAFT_REPLAY_STEP_1_COMPLETE,
    TRACE_DRAFT_REPLAY_STEP_2_COMPLETE,
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

_COMMON_PREFIX = (
    COMMAND_START,
    MESH_OPEN_START,
    MESH_OPEN_COMPLETE,
    MODEL_LOAD_START,
    MODEL_LOAD_COMPLETE,
    ADAPTER_CREATE_START,
    ADAPTER_CREATE_COMPLETE,
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
_COMMAND_TAIL = (
    NATIVE_VERIFIER_START,
    NATIVE_VERIFIER_COMPLETE,
    TARGET_STATE_COMPARE_START,
    TARGET_STATE_COMPARE_COMPLETE,
    PILOT_COMPLETE,
    FAILURE_RECORDED,
    CLEANUP_START,
    CLEANUP_COMPLETE,
    RECEIPT_WRITE_START,
    COMPLETE,
)

_REPREFILL_STAGE_ORDER = (
    *_COMMON_PREFIX,
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
    *_COMMAND_TAIL,
)
_SNAPSHOT_STAGE_ORDER = (
    *_COMMON_PREFIX,
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
    *_COMMAND_TAIL,
)
_STAGE_ORDER_BY_ROUTE = {
    REPREFILL_RECOVERY: _REPREFILL_STAGE_ORDER,
    SNAPSHOT_RECOVERY: _SNAPSHOT_STAGE_ORDER,
}
_STAGE_INDEX_BY_ROUTE = {
    route: {stage: index for index, stage in enumerate(stages)}
    for route, stages in _STAGE_ORDER_BY_ROUTE.items()
}


class MtpStageProgressError(ValueError):
    """The staged progress ledger is invalid or ambiguous."""


@dataclass(frozen=True)
class MtpStageEvent:
    """One completed host boundary and its elapsed process time."""

    stage: str
    elapsed_seconds: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class MtpStageProgress:
    """Immutable ordered progress for one prompt and recovery route."""

    prompt_index: int
    route: str
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
            "last_stage": self.last_stage,
            "complete": self.complete,
            "events": [event.to_mapping() for event in self.events],
        }


def new_mtp_stage_progress(*, prompt_index: int, route: str) -> MtpStageProgress:
    """Create an empty, validated staged progress value."""
    if isinstance(prompt_index, bool) or prompt_index < MINIMUM_PROMPT_INDEX:
        raise MtpStageProgressError("MTP stage prompt index must be positive")
    if route not in _STAGE_ORDER_BY_ROUTE:
        raise MtpStageProgressError("MTP stage recovery route is unsupported")
    return MtpStageProgress(prompt_index=prompt_index, route=route)


def advance_mtp_stage_progress(
    progress: MtpStageProgress,
    *,
    stage: str,
    elapsed_seconds: float,
) -> MtpStageProgress:
    """Return progress advanced to one later known boundary."""
    if progress.complete:
        raise MtpStageProgressError("MTP stage progress is already complete")
    stage_indices = _STAGE_INDEX_BY_ROUTE.get(progress.route)
    if stage_indices is None:
        raise MtpStageProgressError("MTP stage recovery route is unsupported")
    if stage not in stage_indices:
        raise MtpStageProgressError("MTP stage boundary is unknown or route-incompatible")
    if (
        isinstance(elapsed_seconds, bool)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
    ):
        raise MtpStageProgressError("MTP stage elapsed time is invalid")
    if progress.events:
        previous = progress.events[-1]
        if elapsed_seconds < previous.elapsed_seconds:
            raise MtpStageProgressError("MTP stage elapsed time regressed")
        if stage_indices[stage] <= stage_indices[previous.stage]:
            raise MtpStageProgressError("MTP stage boundary repeated or regressed")
    if stage == COMPLETE and progress.last_stage != RECEIPT_WRITE_START:
        raise MtpStageProgressError("MTP stage completed before receipt writing")
    event = MtpStageEvent(stage=stage, elapsed_seconds=elapsed_seconds)
    return replace(progress, events=(*progress.events, event))
