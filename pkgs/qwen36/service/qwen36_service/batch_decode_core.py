"""Pure active-batch planning for Qwen3.8-27B decode on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.batched_concurrency]

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

MINIMUM_ACTIVE_BATCH_WIDTH = 1
MAXIMUM_ACTIVE_BATCH_WIDTH = 16
MINIMUM_TOKEN_ID = 0
MINIMUM_POSITION = 0
MINIMUM_STATE_SLICE = 0
MAXIMUM_RESIDENT_TRACE_COUNT = 1
TRACE_REUSE_REASON_DISABLED = "reuse-disabled"
TRACE_REUSE_REASON_MISSING = "no-resident-trace"
TRACE_REUSE_REASON_MULTIPLE = "multiple-resident-traces"
TRACE_REUSE_REASON_WIDTH = "width-changed"
TRACE_REUSE_REASON_RESET = "reset-incomplete"
TRACE_REUSE_REASON_BINDING = "binding-drift"
TRACE_REUSE_REASON_EAGER_PREFILL = "eager-prefill-next"


class BatchDecodePolicyError(ValueError):
    """The active decode batch does not satisfy the isolation contract."""


@dataclass(frozen=True)
class ActiveDecodeUser:
    """One user that is ready for exactly one decode step."""

    request_id: str
    token_id: int
    position: int
    state_slice: int


@dataclass(frozen=True)
class PlannedDecodeRow:
    """One stable mapping from a batch row to a user state slice."""

    row_index: int
    request_id: str
    token_id: int
    position: int
    next_position: int
    state_slice: int


@dataclass(frozen=True)
class DecodeBatchPlan:
    """A deterministic active-width plan for one full-model decode step."""

    width: int
    rows: tuple[PlannedDecodeRow, ...]


# r[impl tenstorrent.qwen38_p150x2.same_width_trace_reuse]
class TracePrefillAction(Enum):
    """One adapter action before the next request group starts prefill."""

    RELEASE = "release-before-prefill"
    PRESERVE = "preserve-for-reuse"


@dataclass(frozen=True)
class TracePrefillPlan:
    """A pure decision about one resident trace before prefill."""

    action: TracePrefillAction
    reuse_after_prefill: bool
    reason: str | None


def plan_same_width_trace_prefill(
    *,
    requested_width: int,
    resident_widths: tuple[int, ...],
    reuse_enabled: bool,
    prior_reset_complete: bool,
    binding_stable: bool,
) -> TracePrefillPlan:
    """Preserve only one stable same-width trace after a complete reset."""
    if isinstance(requested_width, bool) or not isinstance(requested_width, int):
        raise BatchDecodePolicyError("requested trace width must be an integer")
    if not MINIMUM_ACTIVE_BATCH_WIDTH <= requested_width <= MAXIMUM_ACTIVE_BATCH_WIDTH:
        raise BatchDecodePolicyError("requested trace width is outside the batch bound")
    if not isinstance(resident_widths, tuple):
        raise BatchDecodePolicyError("resident trace widths must be a tuple")
    for resident_width in resident_widths:
        if isinstance(resident_width, bool) or not isinstance(resident_width, int):
            raise BatchDecodePolicyError("resident trace width must be an integer")
        if (
            not MINIMUM_ACTIVE_BATCH_WIDTH
            <= resident_width
            <= MAXIMUM_ACTIVE_BATCH_WIDTH
        ):
            raise BatchDecodePolicyError(
                "resident trace width is outside the batch bound"
            )
    for name, value in (
        ("reuse enabled", reuse_enabled),
        ("prior reset complete", prior_reset_complete),
        ("binding stable", binding_stable),
    ):
        if not isinstance(value, bool):
            raise BatchDecodePolicyError(f"{name} must be a boolean")

    if not reuse_enabled:
        reason = TRACE_REUSE_REASON_DISABLED
    elif not resident_widths:
        reason = TRACE_REUSE_REASON_MISSING
    elif len(resident_widths) > MAXIMUM_RESIDENT_TRACE_COUNT:
        reason = TRACE_REUSE_REASON_MULTIPLE
    elif resident_widths[0] != requested_width:
        reason = TRACE_REUSE_REASON_WIDTH
    elif not prior_reset_complete:
        reason = TRACE_REUSE_REASON_RESET
    elif not binding_stable:
        reason = TRACE_REUSE_REASON_BINDING
    else:
        return TracePrefillPlan(
            action=TracePrefillAction.PRESERVE,
            reuse_after_prefill=True,
            reason=None,
        )
    return TracePrefillPlan(
        action=TracePrefillAction.RELEASE,
        reuse_after_prefill=False,
        reason=reason,
    )


# r[impl tenstorrent.qwen38_p150x2.resident_slot_trace]
def plan_slot_group_trace(
    *,
    width: int,
    resident_widths: tuple[int, ...],
    binding_stable: bool,
    masked_prefill_ready: bool,
) -> TracePrefillPlan:
    """Keep the one resident decode trace for the next continuous-slot group.

    The group's refills replay the masked prefill traces beside it, as later refills in a
    group already do, and its first decode step restages every input. An eager prefill
    would allocate under the active trace, so without the masked traces the trace is
    released. Every row is reset before the group's first refill, so a kept trace never
    replays over the previous group's state.
    """
    if not isinstance(masked_prefill_ready, bool):
        raise BatchDecodePolicyError("masked prefill ready must be a boolean")
    plan = plan_same_width_trace_prefill(
        requested_width=width,
        resident_widths=resident_widths,
        reuse_enabled=True,
        prior_reset_complete=True,
        binding_stable=binding_stable,
    )
    if plan.reuse_after_prefill and not masked_prefill_ready:
        return TracePrefillPlan(
            action=TracePrefillAction.RELEASE,
            reuse_after_prefill=False,
            reason=TRACE_REUSE_REASON_EAGER_PREFILL,
        )
    return plan


def plan_decode_batch(
    users: Sequence[ActiveDecodeUser],
    *,
    vocabulary_size: int,
    maximum_sequence_length: int,
) -> DecodeBatchPlan:
    """Select the active width and advance each isolated user by one position."""
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
        raise BatchDecodePolicyError("vocabulary size must be an integer")
    if vocabulary_size <= MINIMUM_TOKEN_ID:
        raise BatchDecodePolicyError("vocabulary size must be positive")
    if isinstance(maximum_sequence_length, bool) or not isinstance(
        maximum_sequence_length, int
    ):
        raise BatchDecodePolicyError("maximum sequence length must be an integer")
    if maximum_sequence_length <= MINIMUM_POSITION:
        raise BatchDecodePolicyError("maximum sequence length must be positive")
    if isinstance(users, (str, bytes)) or not isinstance(users, Sequence):
        raise BatchDecodePolicyError("active users must be a sequence")

    width = len(users)
    if not MINIMUM_ACTIVE_BATCH_WIDTH <= width <= MAXIMUM_ACTIVE_BATCH_WIDTH:
        raise BatchDecodePolicyError(
            "active batch width must be between "
            f"{MINIMUM_ACTIVE_BATCH_WIDTH} and {MAXIMUM_ACTIVE_BATCH_WIDTH}"
        )

    request_ids: set[str] = set()
    state_slices: set[int] = set()
    rows = []
    for row_index, user in enumerate(users):
        if not isinstance(user, ActiveDecodeUser):
            raise BatchDecodePolicyError("active user has the wrong type")
        if not isinstance(user.request_id, str) or not user.request_id:
            raise BatchDecodePolicyError("request identity must be a non-empty string")
        if user.request_id in request_ids:
            raise BatchDecodePolicyError("request identity collision")
        request_ids.add(user.request_id)

        if isinstance(user.token_id, bool) or not isinstance(user.token_id, int):
            raise BatchDecodePolicyError("token identity must be an integer")
        if not MINIMUM_TOKEN_ID <= user.token_id < vocabulary_size:
            raise BatchDecodePolicyError("token identity is outside the vocabulary")
        if isinstance(user.position, bool) or not isinstance(user.position, int):
            raise BatchDecodePolicyError("decode position must be an integer")
        if not MINIMUM_POSITION <= user.position < maximum_sequence_length:
            raise BatchDecodePolicyError(
                "decode position is outside the sequence bound"
            )
        if isinstance(user.state_slice, bool) or not isinstance(user.state_slice, int):
            raise BatchDecodePolicyError("state slice must be an integer")
        if not MINIMUM_STATE_SLICE <= user.state_slice < MAXIMUM_ACTIVE_BATCH_WIDTH:
            raise BatchDecodePolicyError("state slice is outside the batch bound")
        if user.state_slice in state_slices:
            raise BatchDecodePolicyError("state slice collision")
        state_slices.add(user.state_slice)

        rows.append(
            PlannedDecodeRow(
                row_index=row_index,
                request_id=user.request_id,
                token_id=user.token_id,
                position=user.position,
                next_position=user.position + MINIMUM_ACTIVE_BATCH_WIDTH,
                state_slice=user.state_slice,
            )
        )

    if state_slices != set(range(width)):
        raise BatchDecodePolicyError("state slices must form the active prefix")
    return DecodeBatchPlan(width=width, rows=tuple(rows))
