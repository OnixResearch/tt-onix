"""Pure ordered progress for the prewarmed persistent MTP recovery control."""

# r[impl tenstorrent.qwen38_p150x2.mtp_recovery_prewarmed_control]

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .mtp_drafter_core import EXPECTED_MINIMUM_PROMPT_COUNT
from .mtp_rollback_core import (
    EXPECTED_ROUTE_SCHEDULE,
    REPREFILL_RECOVERY,
    SNAPSHOT_RECOVERY,
)
from .mtp_target_prewarm_core import MtpTargetValidationSignature

PROGRESS_SCHEMA_VERSION = 1
MINIMUM_PROMPT_INDEX = 1
MINIMUM_ROOT_INDEX = 1
MINIMUM_PROGRAM_CACHE_ENTRIES = 0
ZERO_ELAPSED_SECONDS = 0.0

COMMAND_START = "command-start"
MESH_OPEN_START = "mesh-open-start"
MESH_OPEN_COMPLETE = "mesh-open-complete"
MODEL_LOAD_START = "model-load-start"
MODEL_LOAD_COMPLETE = "model-load-complete"
ADAPTER_CREATE_START = "adapter-create-start"
ADAPTER_CREATE_COMPLETE = "adapter-create-complete"
PREWARM_START = "prewarm-start"
PREWARM_PROMPT_START = "prewarm-prompt-start"
PREWARM_PROMPT_COMPLETE = "prewarm-prompt-complete"
PREWARM_COMPLETE = "prewarm-complete"
PERSISTENT_WARMUP_START = "persistent-warmup-start"
WARMUP_ROOT_START = "warmup-root-start"
WARMUP_ROOT_COMPLETE = "warmup-root-complete"
PERSISTENT_WARMUP_COMPLETE = "persistent-warmup-complete"
MEASUREMENT_START = "measurement-start"
MEASURED_ROOT_START = "measured-root-start"
MEASURED_ROOT_COMPLETE = "measured-root-complete"
MEASUREMENT_COMPLETE = "measurement-complete"
FAILURE_RECORDED = "failure-recorded"
CLEANUP_START = "cleanup-start"
CLEANUP_COMPLETE = "cleanup-complete"
EVALUATION_START = "evaluation-start"
EVALUATION_COMPLETE = "evaluation-complete"
RECEIPT_WRITE_START = "receipt-write-start"
COMPLETE = "complete"

WARMUP_ROUTES = (REPREFILL_RECOVERY, SNAPSHOT_RECOVERY)
_STARTUP_STAGES = (
    COMMAND_START,
    MESH_OPEN_START,
    MESH_OPEN_COMPLETE,
    MODEL_LOAD_START,
    MODEL_LOAD_COMPLETE,
    ADAPTER_CREATE_START,
    ADAPTER_CREATE_COMPLETE,
)
_SIGNATURE_STAGE = PREWARM_PROMPT_COMPLETE
_SIGNATURE_MATCH_STAGES = (WARMUP_ROOT_COMPLETE, MEASURED_ROOT_COMPLETE)


class MtpRecoveryControlProgressError(ValueError):
    """The repeated recovery progress is invalid or out of order."""


@dataclass(frozen=True)
class _ExpectedEvent:
    stage: str
    prompt_index: int | None = None
    root_index: int | None = None
    route: str | None = None


@dataclass(frozen=True)
class MtpRecoveryControlEvent:
    """One durable repeated-control boundary."""

    stage: str
    elapsed_seconds: float
    program_cache_entries: int | None
    prompt_index: int | None
    root_index: int | None
    route: str | None
    signature: MtpTargetValidationSignature | None
    signature_match: bool | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_seconds": self.elapsed_seconds,
            "program_cache_entries": self.program_cache_entries,
            "prompt_index": self.prompt_index,
            "root_index": self.root_index,
            "route": self.route,
            "signature": (
                self.signature.to_mapping() if self.signature is not None else None
            ),
            "signature_match": self.signature_match,
        }


@dataclass(frozen=True)
class MtpRecoveryControlProgress:
    """Immutable progress for one full prewarmed recovery control."""

    prompt_count: int
    route_schedule: tuple[str, ...]
    events: tuple[MtpRecoveryControlEvent, ...]
    failed: bool

    @property
    def complete(self) -> bool:
        return bool(self.events) and self.events[-1].stage == COMPLETE

    @property
    def last_stage(self) -> str | None:
        return self.events[-1].stage if self.events else None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "diagnostic_only": True,
            "admission_eligible": False,
            "complete": self.complete,
            "failed": self.failed,
            "prompt_count": self.prompt_count,
            "route_schedule": list(self.route_schedule),
            "measurement_program_cache_entries": (
                measurement_program_cache_entries(self)
            ),
            "last_stage": self.last_stage,
            "events": [event.to_mapping() for event in self.events],
        }


def new_mtp_recovery_control_progress(
    *,
    prompt_count: int,
    route_schedule: tuple[str, ...],
) -> MtpRecoveryControlProgress:
    """Create empty progress for the accepted repeated-control shape."""
    if (
        isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count != EXPECTED_MINIMUM_PROMPT_COUNT
    ):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control prompt count changed"
        )
    if (
        not isinstance(route_schedule, tuple)
        or route_schedule != EXPECTED_ROUTE_SCHEDULE
    ):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control route schedule changed"
        )
    return MtpRecoveryControlProgress(
        prompt_count=prompt_count,
        route_schedule=route_schedule,
        events=(),
        failed=False,
    )


def advance_mtp_recovery_control_progress(
    progress: MtpRecoveryControlProgress,
    *,
    stage: str,
    elapsed_seconds: float,
    program_cache_entries: int | None,
    prompt_index: int | None = None,
    root_index: int | None = None,
    route: str | None = None,
    signature: MtpTargetValidationSignature | None = None,
) -> MtpRecoveryControlProgress:
    """Validate and append one ordered repeated-control boundary."""
    if not isinstance(progress, MtpRecoveryControlProgress):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control progress type is invalid"
        )
    if progress.complete:
        raise MtpRecoveryControlProgressError(
            "MTP recovery control is already complete"
        )
    _validate_elapsed(progress, elapsed_seconds)
    _validate_program_cache(progress, stage, program_cache_entries)

    if stage == FAILURE_RECORDED:
        return _record_failure(
            progress,
            elapsed_seconds=elapsed_seconds,
            program_cache_entries=program_cache_entries,
        )

    expected = _next_expected_event(progress)
    observed = _ExpectedEvent(
        stage=stage,
        prompt_index=prompt_index,
        root_index=root_index,
        route=route,
    )
    if observed != expected:
        raise MtpRecoveryControlProgressError(
            f"MTP recovery control expected {expected}, got {observed}"
        )
    signature_match = _validate_signature_fields(
        progress,
        stage=stage,
        prompt_index=prompt_index,
        signature=signature,
    )
    event = MtpRecoveryControlEvent(
        stage=stage,
        elapsed_seconds=float(elapsed_seconds),
        program_cache_entries=program_cache_entries,
        prompt_index=prompt_index,
        root_index=root_index,
        route=route,
        signature=signature,
        signature_match=signature_match,
    )
    return MtpRecoveryControlProgress(
        prompt_count=progress.prompt_count,
        route_schedule=progress.route_schedule,
        events=(*progress.events, event),
        failed=progress.failed,
    )


def measurement_program_cache_entries(
    progress: MtpRecoveryControlProgress,
) -> int | None:
    """Return the cache count frozen at measurement start."""
    for event in progress.events:
        if event.stage == MEASUREMENT_START:
            return event.program_cache_entries
    return None


def _success_plan(progress: MtpRecoveryControlProgress) -> tuple[_ExpectedEvent, ...]:
    plan = [_ExpectedEvent(stage) for stage in _STARTUP_STAGES]
    plan.append(_ExpectedEvent(PREWARM_START))
    for prompt_index in range(
        MINIMUM_PROMPT_INDEX,
        progress.prompt_count + MINIMUM_PROMPT_INDEX,
    ):
        plan.extend((
            _ExpectedEvent(PREWARM_PROMPT_START, prompt_index=prompt_index),
            _ExpectedEvent(PREWARM_PROMPT_COMPLETE, prompt_index=prompt_index),
        ))
    plan.extend((
        _ExpectedEvent(PREWARM_COMPLETE),
        _ExpectedEvent(PERSISTENT_WARMUP_START),
    ))
    for prompt_index in range(
        MINIMUM_PROMPT_INDEX,
        progress.prompt_count + MINIMUM_PROMPT_INDEX,
    ):
        for root_index, route in enumerate(WARMUP_ROUTES, start=MINIMUM_ROOT_INDEX):
            plan.extend((
                _ExpectedEvent(
                    WARMUP_ROOT_START,
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                ),
                _ExpectedEvent(
                    WARMUP_ROOT_COMPLETE,
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                ),
            ))
    plan.extend((
        _ExpectedEvent(PERSISTENT_WARMUP_COMPLETE),
        _ExpectedEvent(MEASUREMENT_START),
    ))
    for prompt_index in range(
        MINIMUM_PROMPT_INDEX,
        progress.prompt_count + MINIMUM_PROMPT_INDEX,
    ):
        for root_index, route in enumerate(
            progress.route_schedule,
            start=MINIMUM_ROOT_INDEX,
        ):
            plan.extend((
                _ExpectedEvent(
                    MEASURED_ROOT_START,
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                ),
                _ExpectedEvent(
                    MEASURED_ROOT_COMPLETE,
                    prompt_index=prompt_index,
                    root_index=root_index,
                    route=route,
                ),
            ))
    plan.extend((
        _ExpectedEvent(MEASUREMENT_COMPLETE),
        _ExpectedEvent(CLEANUP_START),
        _ExpectedEvent(CLEANUP_COMPLETE),
        _ExpectedEvent(EVALUATION_START),
        _ExpectedEvent(EVALUATION_COMPLETE),
        _ExpectedEvent(RECEIPT_WRITE_START),
        _ExpectedEvent(COMPLETE),
    ))
    return tuple(plan)


def _next_expected_event(progress: MtpRecoveryControlProgress) -> _ExpectedEvent:
    if progress.failed:
        if progress.last_stage == FAILURE_RECORDED:
            return _ExpectedEvent(CLEANUP_START)
        if progress.last_stage == CLEANUP_START:
            return _ExpectedEvent(CLEANUP_COMPLETE)
        raise MtpRecoveryControlProgressError(
            "failed MTP recovery control has no further transition"
        )
    plan = _success_plan(progress)
    if len(progress.events) >= len(plan):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control has no further transition"
        )
    return plan[len(progress.events)]


def _record_failure(
    progress: MtpRecoveryControlProgress,
    *,
    elapsed_seconds: float,
    program_cache_entries: int | None,
) -> MtpRecoveryControlProgress:
    if progress.failed:
        raise MtpRecoveryControlProgressError(
            "MTP recovery control failure was already recorded"
        )
    expected = _next_expected_event(progress)
    event = MtpRecoveryControlEvent(
        stage=FAILURE_RECORDED,
        elapsed_seconds=float(elapsed_seconds),
        program_cache_entries=program_cache_entries,
        prompt_index=expected.prompt_index,
        root_index=expected.root_index,
        route=expected.route,
        signature=None,
        signature_match=None,
    )
    return MtpRecoveryControlProgress(
        prompt_count=progress.prompt_count,
        route_schedule=progress.route_schedule,
        events=(*progress.events, event),
        failed=True,
    )


def _validate_elapsed(
    progress: MtpRecoveryControlProgress,
    elapsed_seconds: float,
) -> None:
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < ZERO_ELAPSED_SECONDS
    ):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control elapsed time is invalid"
        )
    if progress.events and elapsed_seconds < progress.events[-1].elapsed_seconds:
        raise MtpRecoveryControlProgressError(
            "MTP recovery control elapsed time regressed"
        )


def _validate_program_cache(
    progress: MtpRecoveryControlProgress,
    stage: str,
    program_cache_entries: int | None,
) -> None:
    if program_cache_entries is not None and (
        isinstance(program_cache_entries, bool)
        or not isinstance(program_cache_entries, int)
        or program_cache_entries < MINIMUM_PROGRAM_CACHE_ENTRIES
    ):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control program-cache count is invalid"
        )
    if stage == FAILURE_RECORDED or progress.failed:
        return
    previous = next(
        (
            event.program_cache_entries
            for event in reversed(progress.events)
            if event.program_cache_entries is not None
        ),
        None,
    )
    if previous is not None:
        if program_cache_entries is None:
            raise MtpRecoveryControlProgressError(
                "MTP recovery control program-cache observation disappeared"
            )
        if program_cache_entries < previous:
            raise MtpRecoveryControlProgressError(
                "MTP recovery control program-cache count regressed"
            )
    if (
        not progress.failed
        and stage == MESH_OPEN_COMPLETE
        and program_cache_entries is None
    ):
        raise MtpRecoveryControlProgressError(
            "MTP recovery control mesh lacks a program-cache observation"
        )
    measurement_entries = measurement_program_cache_entries(progress)
    if measurement_entries is not None and program_cache_entries != measurement_entries:
        raise MtpRecoveryControlProgressError(
            "MTP recovery control measured program-cache count changed"
        )


def _validate_signature_fields(
    progress: MtpRecoveryControlProgress,
    *,
    stage: str,
    prompt_index: int | None,
    signature: MtpTargetValidationSignature | None,
) -> bool | None:
    if stage == _SIGNATURE_STAGE:
        if not isinstance(signature, MtpTargetValidationSignature):
            raise MtpRecoveryControlProgressError(
                "MTP recovery control prewarm signature is missing"
            )
        return None
    if stage in _SIGNATURE_MATCH_STAGES:
        if not isinstance(signature, MtpTargetValidationSignature):
            raise MtpRecoveryControlProgressError(
                "MTP recovery control measured signature is missing"
            )
        prewarm_signature = _prewarm_signature(progress, prompt_index)
        if signature != prewarm_signature:
            raise MtpRecoveryControlProgressError(
                "MTP recovery control target-validation signature changed"
            )
        return True
    if signature is not None:
        raise MtpRecoveryControlProgressError(
            "MTP recovery control signature metadata is unexpected"
        )
    return None


def _prewarm_signature(
    progress: MtpRecoveryControlProgress,
    prompt_index: int | None,
) -> MtpTargetValidationSignature:
    if prompt_index is None:
        raise MtpRecoveryControlProgressError(
            "MTP recovery control signature lacks a prompt"
        )
    for event in progress.events:
        if (
            event.stage == PREWARM_PROMPT_COMPLETE
            and event.prompt_index == prompt_index
            and event.signature is not None
        ):
            return event.signature
    raise MtpRecoveryControlProgressError(
        "MTP recovery control prompt prewarm signature is missing"
    )
