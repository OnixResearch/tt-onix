"""Device-resident MTP K/V rollback for the diagnostic Qwen probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_kv_rollback]
# r[impl tenstorrent.qwen38_p150x2.mtp_recovery_stability]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.tt_transformers.tt.common import copy_host_to_device

from .mtp_rollback_core import (
    EXPECTED_COMMITTED_ADVANCE_COUNT,
    EXPECTED_SNAPSHOT_ROW_COUNT,
    PERSISTENT_TRACED_RECOVERY,
)
from .mtp_trace_adapter import TracedMtpDraftAdapter

PROBE_BATCH_SIZE = 1
EXPECTED_CACHE_TENSOR_COUNT = 4
CACHE_SEQUENCE_DIMENSION = 2
CACHE_HEAD_DIMENSION = 3
CACHE_RANK = 4
PADDED_CACHE_HEAD_COUNT = 32
MINIMUM_POSITION = 0
MINIMUM_TOKEN_ID = 0
ZERO_DURATION_SECONDS = 0.0


class MtpRollbackAdapterError(RuntimeError):
    """The MTP rollback adapter cannot preserve exact device state."""


@dataclass(frozen=True)
class MtpRecoveryState:
    """Host comparison state after one committed-token advance."""

    next_draft_token_id: int
    feedback: torch.Tensor
    kv_prefix: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class MtpRecoveryResult:
    """One timed recovery route and its resulting continuation state."""

    route: str
    recovery_seconds: float
    queue_drain_seconds: float
    recovery_execution_mode: str
    process_trace_capture_count: int
    trace_binding_valid: bool
    state: MtpRecoveryState
    snapshot_row_count: int
    committed_advance_count: int


class MtpRollbackContext:
    """Own bounded cache-row snapshots and preallocated prefix feedback."""

    def __init__(
        self,
        *,
        owner: Any,
        start_position: int,
        cache_identity: tuple[tuple[int, ...], ...],
        row_snapshots: tuple[tuple[Any, ...], ...],
        prefix_feedback: tuple[Any, ...],
    ) -> None:
        self.owner = owner
        self.mesh = owner.mesh
        self.start_position = start_position
        self.cache_identity = cache_identity
        self.row_snapshots = row_snapshots
        self.prefix_feedback = prefix_feedback
        self._closed = False

    @classmethod
    def capture(
        cls,
        *,
        owner: Any,
        start_position: int,
        initial_feedback: Any,
    ) -> MtpRollbackContext:
        """Allocate every rollback buffer before trace capture begins."""
        if start_position < MINIMUM_POSITION:
            raise MtpRollbackAdapterError("MTP rollback position is negative")
        caches = _mtp_caches(owner)
        cache_identity = tuple(_device_addresses(cache) for cache in caches)
        row_snapshots: list[tuple[Any, ...]] = []
        prefix_feedback: list[Any] = []
        try:
            for offset in range(EXPECTED_SNAPSHOT_ROW_COUNT):
                position = start_position + offset
                row_buffers = []
                try:
                    for cache in caches:
                        row_buffers.append(
                            _snapshot_update_input(owner, cache, position)
                        )
                except (
                    AssertionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    _deallocate_present(row_buffers)
                    raise
                row_snapshots.append(tuple(row_buffers))
            prefix_feedback.append(ttnn.clone(initial_feedback))
            for _index in range(EXPECTED_SNAPSHOT_ROW_COUNT):
                prefix_feedback.append(ttnn.clone(initial_feedback))
            return cls(
                owner=owner,
                start_position=start_position,
                cache_identity=cache_identity,
                row_snapshots=tuple(row_snapshots),
                prefix_feedback=tuple(prefix_feedback),
            )
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
            _deallocate_nested(row_snapshots)
            _deallocate_present(prefix_feedback)
            raise MtpRollbackAdapterError("MTP rollback snapshot failed") from error

    def record_prefix_feedback(self, prefix_length: int, feedback: Any) -> None:
        """Copy one trace output into its preallocated prefix buffer."""
        self._require_open()
        if not 0 < prefix_length < len(self.prefix_feedback):
            raise MtpRollbackAdapterError("MTP rollback feedback prefix is invalid")
        ttnn.copy(feedback, self.prefix_feedback[prefix_length])

    def restore_rejected_rows(self, accepted_prefix_length: int) -> None:
        """Restore only rows written beyond the accepted prefix."""
        self._require_open()
        if not 0 <= accepted_prefix_length <= self.owner.profile.draft_token_count:
            raise MtpRollbackAdapterError("MTP rollback accepted prefix is invalid")
        self._require_cache_identity()
        caches = _mtp_caches(self.owner)
        first_rejected_row = min(
            accepted_prefix_length,
            EXPECTED_SNAPSHOT_ROW_COUNT,
        )
        for offset in range(first_rejected_row, EXPECTED_SNAPSHOT_ROW_COUNT):
            position = self.start_position + offset
            position_tensor = _position_tensor(self.owner, position)
            try:
                for cache, snapshot in zip(
                    caches,
                    self.row_snapshots[offset],
                    strict=True,
                ):
                    ttnn.experimental.paged_update_cache(
                        cache,
                        snapshot,
                        update_idxs_tensor=position_tensor,
                    )
            finally:
                ttnn.deallocate(position_tensor)

    def feedback_for_prefix(self, accepted_prefix_length: int) -> tuple[Any, int]:
        """Return the newest feedback already represented by original drafting."""
        self._require_open()
        represented_prefix = min(
            accepted_prefix_length,
            EXPECTED_SNAPSHOT_ROW_COUNT,
        )
        return self.prefix_feedback[represented_prefix], represented_prefix

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _deallocate_nested(self.row_snapshots)
        _deallocate_present(self.prefix_feedback)

    def _require_cache_identity(self) -> None:
        observed = tuple(_device_addresses(cache) for cache in _mtp_caches(self.owner))
        if observed != self.cache_identity:
            raise MtpRollbackAdapterError("MTP cache identity changed before rollback")

    def _require_open(self) -> None:
        if self._closed:
            raise MtpRollbackAdapterError("MTP rollback context is closed")


def recover_from_snapshot(
    *,
    owner: Any,
    trace_adapter: TracedMtpDraftAdapter,
    context: MtpRollbackContext,
    draft_token_ids: tuple[int, ...],
    accepted_prefix_length: int,
    committed_token_id: int,
) -> MtpRecoveryResult:
    """Restore rejected rows and advance one committed target token."""
    _validate_recovery_inputs(
        owner=owner,
        draft_token_ids=draft_token_ids,
        accepted_prefix_length=accepted_prefix_length,
        committed_token_id=committed_token_id,
    )
    queue_drain_seconds = _drain_device(owner)
    recovery_started = time.monotonic()
    try:
        context.restore_rejected_rows(accepted_prefix_length)
        current_feedback, represented_prefix = context.feedback_for_prefix(
            accepted_prefix_length
        )
        token_inputs = (
            *draft_token_ids[represented_prefix:accepted_prefix_length],
            committed_token_id,
        )
        replay = trace_adapter.replay_token_inputs(
            token_inputs=token_inputs,
            initial_feedback=current_feedback,
            start_position=context.start_position + represented_prefix,
        )
        ttnn.synchronize_device(owner.mesh)
        recovery_seconds = time.monotonic() - recovery_started
        _validate_duration(recovery_seconds)
        state = _recovery_state(
            owner=owner,
            next_draft_token_id=replay.token_ids[-1],
            feedback=replay.feedback,
            stop_position=context.start_position + accepted_prefix_length + 1,
        )
        return MtpRecoveryResult(
            route="snapshot",
            recovery_seconds=recovery_seconds,
            queue_drain_seconds=queue_drain_seconds,
            recovery_execution_mode=PERSISTENT_TRACED_RECOVERY,
            process_trace_capture_count=trace_adapter.trace_capture_count,
            trace_binding_valid=replay.trace_binding_valid,
            state=state,
            snapshot_row_count=EXPECTED_SNAPSHOT_ROW_COUNT,
            committed_advance_count=EXPECTED_COMMITTED_ADVANCE_COUNT,
        )
    finally:
        context.close()


def recover_from_reprefill(
    *,
    owner: Any,
    trace_adapter: TracedMtpDraftAdapter,
    prompt_tokens: torch.Tensor,
    target_hidden: Any,
    anchor_token_id: int,
    prompt_length: int,
    draft_token_ids: tuple[int, ...],
    accepted_prefix_length: int,
    committed_token_id: int,
) -> MtpRecoveryResult:
    """Rebuild the prefix and advance one committed target token."""
    _validate_recovery_inputs(
        owner=owner,
        draft_token_ids=draft_token_ids,
        accepted_prefix_length=accepted_prefix_length,
        committed_token_id=committed_token_id,
    )
    queue_drain_seconds = _drain_device(owner)
    recovery_started = time.monotonic()
    feedback = None
    try:
        first_draft, feedback, _selection_seconds = owner._mtp_prefill(
            prompt_tokens,
            target_hidden,
            anchor_token_id,
            prompt_length,
            selector_mode=owner._rollback_selector_mode,
        )
        if first_draft != draft_token_ids[0]:
            raise MtpRollbackAdapterError("MTP re-prefill changed the first draft")
        token_inputs = (
            *draft_token_ids[:accepted_prefix_length],
            committed_token_id,
        )
        replay = trace_adapter.replay_token_inputs(
            token_inputs=token_inputs,
            initial_feedback=feedback,
            start_position=prompt_length,
        )
        for step in range(accepted_prefix_length):
            expected_next = (
                draft_token_ids[step + 1] if step + 1 < len(draft_token_ids) else None
            )
            if expected_next is not None and replay.token_ids[step] != expected_next:
                raise MtpRollbackAdapterError(
                    "MTP re-prefill changed an accepted draft"
                )
        ttnn.synchronize_device(owner.mesh)
        recovery_seconds = time.monotonic() - recovery_started
        _validate_duration(recovery_seconds)
        state = _recovery_state(
            owner=owner,
            next_draft_token_id=replay.token_ids[-1],
            feedback=replay.feedback,
            stop_position=prompt_length + accepted_prefix_length + 1,
        )
        return MtpRecoveryResult(
            route="reprefill",
            recovery_seconds=recovery_seconds,
            queue_drain_seconds=queue_drain_seconds,
            recovery_execution_mode=PERSISTENT_TRACED_RECOVERY,
            process_trace_capture_count=trace_adapter.trace_capture_count,
            trace_binding_valid=replay.trace_binding_valid,
            state=state,
            snapshot_row_count=0,
            committed_advance_count=EXPECTED_COMMITTED_ADVANCE_COUNT,
        )
    finally:
        if feedback is not None:
            ttnn.deallocate(feedback)


def recovery_state_components_equal(
    left: MtpRecoveryState,
    right: MtpRecoveryState,
) -> tuple[bool, bool, bool]:
    """Compare the next token, feedback, and bounded K/V prefix exactly."""
    token_valid = left.next_draft_token_id == right.next_draft_token_id
    feedback_valid = torch.equal(left.feedback, right.feedback)
    kv_valid = len(left.kv_prefix) == len(right.kv_prefix) and all(
        torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(
            left.kv_prefix,
            right.kv_prefix,
            strict=True,
        )
    )
    return token_valid, feedback_valid, kv_valid


def recovery_states_equal(left: MtpRecoveryState, right: MtpRecoveryState) -> bool:
    """Compare complete bounded continuation state exactly."""
    return all(recovery_state_components_equal(left, right))


def _drain_device(owner: Any) -> float:
    drain_started = time.monotonic()
    ttnn.synchronize_device(owner.mesh)
    drain_seconds = time.monotonic() - drain_started
    _validate_duration(drain_seconds)
    return drain_seconds


def _recovery_state(
    *,
    owner: Any,
    next_draft_token_id: int,
    feedback: Any,
    stop_position: int,
) -> MtpRecoveryState:
    if next_draft_token_id < MINIMUM_TOKEN_ID:
        raise MtpRollbackAdapterError("MTP recovery returned an invalid token")
    mesh_reader = ttnn.ConcatMeshToTensor(owner.mesh, dim=0)
    feedback_host = ttnn.to_torch(feedback, mesh_composer=mesh_reader).clone()
    cache_rows = []
    for cache in _mtp_caches(owner):
        prefix = ttnn.slice(
            cache,
            (0, 0, 0, 0),
            (
                int(cache.shape[0]),
                int(cache.shape[1]),
                stop_position,
                int(cache.shape[CACHE_HEAD_DIMENSION]),
            ),
        )
        try:
            cache_rows.append(ttnn.to_torch(prefix, mesh_composer=mesh_reader).clone())
        finally:
            ttnn.deallocate(prefix)
    return MtpRecoveryState(
        next_draft_token_id=next_draft_token_id,
        feedback=feedback_host,
        kv_prefix=tuple(cache_rows),
    )


def _snapshot_update_input(owner: Any, cache: Any, position: int) -> Any:
    if len(cache.shape) != CACHE_RANK:
        raise MtpRollbackAdapterError("MTP cache rank changed")
    batch_size = int(cache.shape[0])
    if batch_size != PROBE_BATCH_SIZE or int(cache.shape[1]) != PROBE_BATCH_SIZE:
        raise MtpRollbackAdapterError(
            "MTP rollback requires batch one and one K/V head"
        )
    stop_position = position + 1
    if stop_position > int(cache.shape[CACHE_SEQUENCE_DIMENSION]):
        raise MtpRollbackAdapterError("MTP rollback snapshot exceeds cache capacity")
    row = ttnn.slice(
        cache,
        (0, 0, position, 0),
        (
            batch_size,
            PROBE_BATCH_SIZE,
            stop_position,
            int(cache.shape[CACHE_HEAD_DIMENSION]),
        ),
    )
    padded = None
    try:
        padded = ttnn.pad(
            row,
            [
                PROBE_BATCH_SIZE,
                batch_size,
                PADDED_CACHE_HEAD_COUNT,
                int(cache.shape[CACHE_HEAD_DIMENSION]),
            ],
            [0, 0, 0, 0],
            0.0,
        )
        return ttnn.to_memory_config(
            padded,
            owner.mtp_layer.attention._kv_shard_cfg(batch_size),
        )
    finally:
        ttnn.deallocate(row)
        if padded is not None:
            ttnn.deallocate(padded)


def _position_tensor(owner: Any, position: int) -> Any:
    host_inputs = owner.model.prepare_decode_inputs_host(
        torch.tensor([[MINIMUM_TOKEN_ID]], dtype=torch.int32),
        torch.tensor([position], dtype=torch.int32),
        page_table=None,
    )
    device_inputs = copy_host_to_device(host_inputs, mesh_device=owner.mesh)
    if len(device_inputs) != CACHE_RANK:
        _deallocate_present(device_inputs)
        raise MtpRollbackAdapterError("MTP rollback position inputs are incomplete")
    token_tensor, position_tensor, rope_tensor, page_table = device_inputs
    _deallocate_present((token_tensor, rope_tensor, page_table))
    return position_tensor


def _mtp_caches(owner: Any) -> tuple[Any, ...]:
    attention = owner.mtp_layer.attention
    caches = tuple(attention.k_caches or ()) + tuple(attention.v_caches or ())
    if len(caches) != EXPECTED_CACHE_TENSOR_COUNT:
        raise MtpRollbackAdapterError("MTP rollback cache count changed")
    return caches


def _device_addresses(value: Any) -> tuple[int, ...]:
    try:
        tensors = tuple(ttnn.get_device_tensors(value))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        tensors = (value,)
    try:
        return tuple(int(tensor.buffer_address()) for tensor in tensors)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise MtpRollbackAdapterError(
            "MTP rollback cache has no device address"
        ) from error


def _validate_recovery_inputs(
    *,
    owner: Any,
    draft_token_ids: tuple[int, ...],
    accepted_prefix_length: int,
    committed_token_id: int,
) -> None:
    if len(draft_token_ids) != owner.profile.draft_token_count:
        raise MtpRollbackAdapterError("MTP recovery draft count changed")
    if not 0 <= accepted_prefix_length <= owner.profile.draft_token_count:
        raise MtpRollbackAdapterError("MTP recovery accepted prefix is invalid")
    if committed_token_id < MINIMUM_TOKEN_ID:
        raise MtpRollbackAdapterError("MTP recovery committed token is invalid")


def _validate_duration(value: float) -> None:
    if not math.isfinite(value) or value <= ZERO_DURATION_SECONDS:
        raise MtpRollbackAdapterError("MTP recovery duration is invalid")


def _deallocate_nested(values: Any) -> None:
    for value in values:
        if isinstance(value, (tuple, list)):
            _deallocate_nested(value)
        elif value is not None:
            ttnn.deallocate(value)


def _deallocate_present(values: Any) -> None:
    for value in values:
        if value is not None:
            ttnn.deallocate(value)
