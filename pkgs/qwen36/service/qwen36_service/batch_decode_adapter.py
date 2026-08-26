"""TTNN adapter for isolated batched Qwen decode on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.batched_concurrency]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.tt_transformers.tt.common import copy_host_to_device

from .batch_decode_core import (
    MAXIMUM_ACTIVE_BATCH_WIDTH,
    MINIMUM_ACTIVE_BATCH_WIDTH,
    TRACE_REUSE_REASON_BINDING,
    TracePrefillPlan,
    plan_same_width_trace_prefill,
)
from .decode_adapter import TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS
from .decode_core import DecodeProfile, required_page_table_blocks

MAX_REDUCTION_COLUMNS = 32
TILE_EDGE_ELEMENTS = 32
PADDED_LOGIT_SENTINEL = -1.0e30
TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = False
MUTABLE_INPUT_COUNT = 3
EXPECTED_DECODE_INPUT_COUNT = 4
MINIMUM_POSITION = 0


class BatchDecodeAdapterError(RuntimeError):
    """The TTNN batch adapter cannot preserve user-state isolation."""


@dataclass(frozen=True)
class BatchPrefillResult:
    first_token_ids: tuple[int, ...]
    elapsed_seconds: float
    first_token_ready_seconds: float
    page_table_blocks: int
    state_install_seconds: float
    trace_reuse_planned: bool = False
    trace_transition_reason: str | None = None


@dataclass(frozen=True)
class TracePreparationResult:
    capture_seconds: float
    capture_count: int
    reused: bool
    resident_trace_count: int
    binding_stable: bool
    transition_reason: str | None


@dataclass
class _BatchTrace:
    width: int
    trace_id: Any
    device_inputs: tuple[Any, ...]
    logits: Any
    indices: Any
    values: Any
    binding: tuple[str, ...]
    capture_seconds: float


@dataclass(frozen=True)
class _GdnSnapshot:
    layer: Any
    recurrent: Any
    convolution: tuple[Any, ...]


@dataclass(frozen=True)
class _GdnZeroSources:
    layer: Any
    recurrent: Any
    convolution: Any
    convolution_carry: Any


class BatchedTracedDecodeAdapter:
    """Own fixed-capacity user state and exact-width decode traces."""

    def __init__(
        self,
        *,
        model: Any,
        profile: DecodeProfile,
        maximum_width: int = MAXIMUM_ACTIVE_BATCH_WIDTH,
    ) -> None:
        if model.num_devices <= MINIMUM_ACTIVE_BATCH_WIDTH:
            raise BatchDecodeAdapterError(
                "batched TP decode requires more than one device"
            )
        if (
            not MINIMUM_ACTIVE_BATCH_WIDTH
            <= maximum_width
            <= MAXIMUM_ACTIVE_BATCH_WIDTH
        ):
            raise BatchDecodeAdapterError("batch capacity is outside policy")
        if int(model.args.max_batch_size) < maximum_width:
            raise BatchDecodeAdapterError(
                "model batch capacity is smaller than adapter capacity"
            )
        if model.args.vocab_size % model.num_devices != 0:
            raise BatchDecodeAdapterError(
                "vocabulary size is not divisible by the device count"
            )
        self.model = model
        self.profile = profile
        self.maximum_width = maximum_width
        self.mesh = model.mesh_device
        self.vocabulary_size = int(model.args.vocab_size)
        self.device_count = int(model.num_devices)
        self.vocabulary_per_shard = self.vocabulary_size // self.device_count
        self.blocks_per_user = self._aligned_blocks_per_user()
        self.page_table = self._make_page_table()
        self._mesh_reader = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        self._traces: dict[int, _BatchTrace] = {}
        self._pending_trace_plan: TracePrefillPlan | None = None
        self._prior_reset_complete = True
        self._closed = False
        self._allocate_persistent_state()
        self._gdn_zero_sources = self._capture_gdn_zero_sources()

    def _aligned_blocks_per_user(self) -> int:
        block_count = (
            self.profile.maximum_sequence_length + self.profile.kv_block_size - 1
        ) // self.profile.kv_block_size
        return (
            (block_count + TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS - 1)
            // TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS
        ) * TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS

    def _make_page_table(self) -> torch.Tensor:
        physical_blocks = self.blocks_per_user * self.maximum_width
        return torch.arange(physical_blocks, dtype=torch.int32).reshape(
            self.maximum_width,
            self.blocks_per_user,
        )

    def _allocate_persistent_state(self) -> None:
        cache_shape = [
            int(self.page_table.numel()),
            int(self.model.args.n_local_kv_heads),
            self.profile.kv_block_size,
            int(self.model.args.head_dim),
        ]
        self.model.allocate_kv_caches(
            cache_shape,
            ttnn.bfloat16,
            batch_size=self.maximum_width,
        )

    def _capture_gdn_zero_sources(self) -> tuple[_GdnZeroSources, ...]:
        sources = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            attention = layer.attention
            if (
                attention.rec_state is None
                or attention.conv_states is None
                or attention.conv_carry is None
                or attention._zero_rec is None
                or attention._zero_conv0 is None
                or attention._zero_conv_carry is None
            ):
                raise BatchDecodeAdapterError("GDN zero state is not allocated")
            if tuple(attention._zero_rec.shape) != tuple(attention.rec_state.shape):
                raise BatchDecodeAdapterError(
                    "GDN recurrent zero shape differs from decode state"
                )
            if tuple(attention._zero_conv0.shape) != tuple(
                attention.conv_states[0].shape
            ):
                raise BatchDecodeAdapterError(
                    "GDN convolution zero shape differs from decode state"
                )
            sources.append(
                _GdnZeroSources(
                    layer=attention,
                    recurrent=attention._zero_rec,
                    convolution=attention._zero_conv0,
                    convolution_carry=attention._zero_conv_carry,
                )
            )
        if not sources:
            raise BatchDecodeAdapterError("model has no GDN zero state")
        return tuple(sources)

    def prepare_width(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> TracePreparationResult:
        """Reuse one safe trace or capture one exact-width replacement."""
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        if len(positions) != width:
            raise BatchDecodeAdapterError("trace token and position widths differ")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocabulary_size
            for token in token_ids
        ):
            raise BatchDecodeAdapterError("trace token is outside the vocabulary")
        if any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not MINIMUM_POSITION <= position < self.profile.maximum_sequence_length
            for position in positions
        ):
            raise BatchDecodeAdapterError("trace position is outside the profile bound")
        pending_plan = self._pending_trace_plan
        self._pending_trace_plan = None
        transition_reason = pending_plan.reason if pending_plan is not None else None
        if pending_plan is not None and pending_plan.reuse_after_prefill:
            resident_trace = self._traces.get(width)
            if resident_trace is not None and self._binding_matches(resident_trace):
                return TracePreparationResult(
                    capture_seconds=0.0,
                    capture_count=0,
                    reused=True,
                    resident_trace_count=len(self._traces),
                    binding_stable=True,
                    transition_reason=None,
                )
            transition_reason = TRACE_REUSE_REASON_BINDING
            self._release_traces()
        elif self._traces:
            self._release_traces()

        tokens = torch.tensor(token_ids, dtype=torch.int32).reshape(
            width,
            MINIMUM_ACTIVE_BATCH_WIDTH,
        )
        position_tensor = torch.tensor(positions, dtype=torch.int32)
        host_inputs = self.model.prepare_decode_inputs_host(
            tokens,
            position_tensor,
            page_table=self.page_table[:width].contiguous(),
        )
        device_inputs = tuple(copy_host_to_device(host_inputs, mesh_device=self.mesh))
        if len(device_inputs) != EXPECTED_DECODE_INPUT_COUNT:
            raise BatchDecodeAdapterError("batch trace inputs are incomplete")

        trace_id = None
        logits = None
        indices = None
        values = None
        snapshots = self._clone_gdn_state()
        capture_started = time.monotonic()
        try:
            warm_logits = self._forward_sharded(device_inputs)
            warm_indices, warm_values = self._reduce_shards(warm_logits, width)
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(warm_indices)
            ttnn.deallocate(warm_values)
            ttnn.deallocate(warm_logits)
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.mesh)

            trace_id = ttnn.begin_trace_capture(
                self.mesh,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            logits = self._forward_sharded(device_inputs)
            indices, values = self._reduce_shards(logits, width)
            ttnn.end_trace_capture(
                self.mesh,
                trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            ttnn.synchronize_device(self.mesh)
            capture_seconds = time.monotonic() - capture_started
            if (
                not math.isfinite(capture_seconds)
                or capture_seconds <= 0
                or capture_seconds > self.profile.trace_capture_timeout_seconds
            ):
                raise BatchDecodeAdapterError("batch trace capture exceeded its bound")
            trace = _BatchTrace(
                width=width,
                trace_id=trace_id,
                device_inputs=device_inputs,
                logits=logits,
                indices=indices,
                values=values,
                binding=self._observe_binding(device_inputs, logits),
                capture_seconds=capture_seconds,
            )
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.mesh)
            self._traces[width] = trace
            return TracePreparationResult(
                capture_seconds=capture_seconds,
                capture_count=1,
                reused=False,
                resident_trace_count=len(self._traces),
                binding_stable=True,
                transition_reason=transition_reason,
            )
        except (AssertionError, OSError, RuntimeError, ValueError) as error:
            self._restore_gdn_state(snapshots)
            ttnn.synchronize_device(self.mesh)
            if trace_id is not None:
                try:
                    ttnn.release_trace(self.mesh, trace_id)
                except (AssertionError, OSError, RuntimeError, ValueError):
                    pass
            for tensor in (indices, values, logits):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            for tensor in device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)
            raise BatchDecodeAdapterError("batch trace capture failed") from error
        finally:
            self._deallocate_gdn_state(snapshots)

    def prefill_users(
        self,
        token_ids: tuple[torch.Tensor, ...],
        *,
        generation_tokens: int,
        trace_reuse_enabled: bool = False,
    ) -> BatchPrefillResult:
        """Prefill isolated state after a pure one-trace transition decision."""
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        trace_plan = plan_same_width_trace_prefill(
            requested_width=width,
            resident_widths=tuple(sorted(self._traces)),
            reuse_enabled=trace_reuse_enabled,
            prior_reset_complete=self._prior_reset_complete,
            binding_stable=self._single_trace_binding_stable(),
        )
        if not trace_plan.reuse_after_prefill:
            self._release_traces()
        self._pending_trace_plan = trace_plan
        self.reset_all()
        self._prior_reset_complete = False
        required_blocks = []
        valid_lengths = []
        for user_tokens in token_ids:
            if not isinstance(user_tokens, torch.Tensor) or user_tokens.ndim != 2:
                raise BatchDecodeAdapterError("prefill tokens must be rank-two tensors")
            if user_tokens.shape[0] != MINIMUM_ACTIVE_BATCH_WIDTH:
                raise BatchDecodeAdapterError("prefill tokens must contain one user")
            prompt_tokens = int(user_tokens.shape[1])
            required_blocks.append(
                required_page_table_blocks(
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                    kv_block_size=self.profile.kv_block_size,
                    capacity_blocks=self.blocks_per_user,
                )
            )
            valid_lengths.append(prompt_tokens)
        page_table_blocks = max(required_blocks)
        bounded_page_table = self.page_table[
            :width,
            :page_table_blocks,
        ].contiguous()
        started = time.monotonic()
        observed_first_tokens: tuple[int, ...] | None = None
        first_token_ready_seconds: float | None = None

        def observe_first_tokens(host_logits: list[torch.Tensor]) -> None:
            nonlocal observed_first_tokens, first_token_ready_seconds
            if (
                observed_first_tokens is not None
                or first_token_ready_seconds is not None
            ):
                raise BatchDecodeAdapterError(
                    "prefill reported first tokens more than once"
                )
            if len(host_logits) != width:
                raise BatchDecodeAdapterError(
                    "prefill output count differs from active width"
                )
            first_token_ids = []
            for user_logits in host_logits:
                rows = user_logits.float().reshape(-1, self.vocabulary_size)
                if rows.shape[0] != MINIMUM_ACTIVE_BATCH_WIDTH:
                    raise BatchDecodeAdapterError(
                        "prefill logits have the wrong row count"
                    )
                first_token_ids.append(int(torch.argmax(rows[0]).item()))
            observed_first_tokens = tuple(first_token_ids)
            first_token_ready_seconds = time.monotonic() - started

        logits = self.model.prefill_paged_slots(
            list(token_ids),
            bounded_page_table,
            list(range(width)),
            valid_lens=valid_lengths,
            first_token_callback=observe_first_tokens,
        )
        ttnn.synchronize_device(self.mesh)
        elapsed_seconds = time.monotonic() - started
        if len(logits) != width:
            raise BatchDecodeAdapterError(
                "prefill output count differs from active width"
            )
        if observed_first_tokens is None or first_token_ready_seconds is None:
            raise BatchDecodeAdapterError(
                "prefill did not report first-token readiness"
            )
        state_install_seconds = elapsed_seconds - first_token_ready_seconds
        if state_install_seconds < 0:
            raise BatchDecodeAdapterError("prefill phase timing regressed")
        return BatchPrefillResult(
            first_token_ids=observed_first_tokens,
            elapsed_seconds=elapsed_seconds,
            first_token_ready_seconds=first_token_ready_seconds,
            page_table_blocks=page_table_blocks,
            state_install_seconds=state_install_seconds,
            trace_reuse_planned=trace_plan.reuse_after_prefill,
            trace_transition_reason=trace_plan.reason,
        )

    def decode_traced(
        self,
        *,
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Advance every active-prefix user by exactly one traced decode step."""
        width = len(token_ids)
        self._require_width(width)
        self._require_open()
        if len(positions) != width:
            raise BatchDecodeAdapterError("token and position widths differ")
        trace = self._traces.get(width)
        if trace is None:
            raise BatchDecodeAdapterError("batch width has no prepared trace")
        self._require_binding(trace)
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocabulary_size
            for token in token_ids
        ):
            raise BatchDecodeAdapterError("batch token is outside the vocabulary")
        if any(
            isinstance(position, bool)
            or not isinstance(position, int)
            or not MINIMUM_POSITION <= position < self.profile.maximum_sequence_length
            for position in positions
        ):
            raise BatchDecodeAdapterError("batch position is outside the profile bound")

        host_inputs = self.model.prepare_decode_inputs_host(
            torch.tensor(token_ids, dtype=torch.int32).reshape(
                width,
                MINIMUM_ACTIVE_BATCH_WIDTH,
            ),
            torch.tensor(positions, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(
            host_inputs[:MUTABLE_INPUT_COUNT],
            device_tensors=trace.device_inputs[:MUTABLE_INPUT_COUNT],
        )
        ttnn.execute_trace(
            self.mesh,
            trace.trace_id,
            cq_id=TRACE_COMMAND_QUEUE_ID,
            blocking=TRACE_BLOCKING_REPLAY,
        )
        ttnn.synchronize_device(self.mesh)
        return self._read_global_winners(trace.indices, trace.values, width)

    def reset_all(self) -> None:
        """Zero all persistent GDN rows without using the B=1 prefill scratch zeros."""
        self._require_open()
        for sources in self._gdn_zero_sources:
            layer = sources.layer
            if layer.rec_state is None or layer.conv_states is None:
                raise BatchDecodeAdapterError("GDN reset target is not allocated")
            if tuple(layer.rec_state.shape) != tuple(sources.recurrent.shape):
                raise BatchDecodeAdapterError("GDN recurrent reset shape changed")
            ttnn.copy(sources.recurrent, layer.rec_state)
            for state in layer.conv_states:
                if tuple(state.shape) != tuple(sources.convolution.shape):
                    raise BatchDecodeAdapterError("GDN convolution reset shape changed")
                ttnn.copy(sources.convolution, state)
            if layer.conv_carry is not None:
                ttnn.copy(sources.convolution_carry, layer.conv_carry)
        ttnn.synchronize_device(self.mesh)
        self._prior_reset_complete = True

    def _clone_gdn_state(self) -> tuple[_GdnSnapshot, ...]:
        snapshots = []
        for layer in self.model.layers:
            if layer.is_full_attention:
                continue
            attention = layer.attention
            if attention.rec_state is None or attention.conv_states is None:
                raise BatchDecodeAdapterError("GDN state is not allocated")
            snapshots.append(
                _GdnSnapshot(
                    layer=attention,
                    recurrent=ttnn.clone(attention.rec_state),
                    convolution=tuple(
                        ttnn.clone(state) for state in attention.conv_states
                    ),
                )
            )
        if not snapshots:
            raise BatchDecodeAdapterError("model has no GDN state")
        return tuple(snapshots)

    @staticmethod
    def _restore_gdn_state(snapshots: tuple[_GdnSnapshot, ...]) -> None:
        for snapshot in snapshots:
            ttnn.copy(snapshot.recurrent, snapshot.layer.rec_state)
            for source, target in zip(
                snapshot.convolution,
                snapshot.layer.conv_states,
                strict=True,
            ):
                ttnn.copy(source, target)

    @staticmethod
    def _deallocate_gdn_state(snapshots: tuple[_GdnSnapshot, ...]) -> None:
        for snapshot in snapshots:
            ttnn.deallocate(snapshot.recurrent)
            for state in snapshot.convolution:
                ttnn.deallocate(state)

    def _forward_sharded(self, device_inputs: tuple[Any, ...]) -> Any:
        token_buffer, position_buffer, rope_buffer, page_table_buffer = device_inputs
        cos, sin = unpack_rope(rope_buffer)
        return self.model._forward_decode(
            token_buffer,
            cos,
            sin,
            position_buffer,
            page_table_buffer,
            sharded_lm_head=True,
        )

    def _reduce_shards(self, sharded_logits: Any, width: int) -> tuple[Any, Any]:
        logits_row_major = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
        indices = ttnn.argmax(logits_row_major, dim=-1, keepdim=False)
        ttnn.deallocate(logits_row_major)
        padded_rows = (
            (
                (self.vocabulary_per_shard + MAX_REDUCTION_COLUMNS - 1)
                // MAX_REDUCTION_COLUMNS
                + TILE_EDGE_ELEMENTS
                - 1
            )
            // TILE_EDGE_ELEMENTS
        ) * TILE_EDGE_ELEMENTS
        padded_elements = padded_rows * MAX_REDUCTION_COLUMNS
        padded = ttnn.pad(
            sharded_logits,
            [
                (0, 0),
                (0, 0),
                (0, 0),
                (0, padded_elements - self.vocabulary_per_shard),
            ],
            value=PADDED_LOGIT_SENTINEL,
        )
        grid = ttnn.reshape(
            padded,
            (MINIMUM_ACTIVE_BATCH_WIDTH, width, padded_rows, MAX_REDUCTION_COLUMNS),
        )
        partial = ttnn.max(grid, dim=-1)
        partial_row = ttnn.reshape(
            partial,
            (
                MINIMUM_ACTIVE_BATCH_WIDTH,
                MINIMUM_ACTIVE_BATCH_WIDTH,
                width,
                padded_rows,
            ),
        )
        values = ttnn.max(partial_row, dim=-1)
        ttnn.deallocate(padded)
        ttnn.deallocate(grid)
        ttnn.deallocate(partial)
        ttnn.deallocate(partial_row)
        return indices, values

    def _read_global_winners(
        self,
        indices: Any,
        values: Any,
        width: int,
    ) -> tuple[int, ...]:
        host_indices = ttnn.to_torch(
            indices,
            mesh_composer=self._mesh_reader,
        ).reshape(self.device_count, width)
        host_values = ttnn.to_torch(
            values,
            mesh_composer=self._mesh_reader,
        ).reshape(self.device_count, width)
        winning_devices = torch.argmax(host_values, dim=0)
        columns = torch.arange(width)
        tokens = winning_devices * self.vocabulary_per_shard + host_indices[
            winning_devices, columns
        ].to(torch.int64)
        result = tuple(int(token) for token in tokens.tolist())
        if any(not 0 <= token < self.vocabulary_size for token in result):
            raise BatchDecodeAdapterError("batch winner is outside the vocabulary")
        return result

    def _observe_binding(
        self,
        device_inputs: tuple[Any, ...],
        logits: Any,
    ) -> tuple[str, ...]:
        gdn_state = tuple(
            (layer.attention.rec_state, tuple(layer.attention.conv_states))
            for layer in self.model.layers
            if not layer.is_full_attention
        )
        return (
            *(_tensor_identity(tensor) for tensor in device_inputs),
            _nested_identity(self.model._paged_kv_caches),
            _nested_identity(gdn_state),
            _tensor_identity(logits),
        )

    def _binding_matches(self, trace: _BatchTrace) -> bool:
        try:
            observed = self._observe_binding(trace.device_inputs, trace.logits)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return observed == trace.binding

    def _single_trace_binding_stable(self) -> bool:
        if len(self._traces) != 1:
            return False
        return self._binding_matches(next(iter(self._traces.values())))

    def _require_binding(self, trace: _BatchTrace) -> None:
        if not self._binding_matches(trace):
            raise BatchDecodeAdapterError("batch trace binding changed")

    def _require_width(self, width: int) -> None:
        if isinstance(width, bool) or not isinstance(width, int):
            raise BatchDecodeAdapterError("active width must be an integer")
        if not MINIMUM_ACTIVE_BATCH_WIDTH <= width <= self.maximum_width:
            raise BatchDecodeAdapterError("active width is outside adapter capacity")

    def _require_open(self) -> None:
        if self._closed:
            raise BatchDecodeAdapterError("batch decode adapter is closed")

    def _release_traces(self) -> None:
        for trace in self._traces.values():
            ttnn.release_trace(self.mesh, trace.trace_id)
            for tensor in (trace.indices, trace.values, trace.logits):
                ttnn.deallocate(tensor)
            for tensor in trace.device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)
        self._traces.clear()
        self._pending_trace_plan = None

    def close(self) -> None:
        if self._closed:
            return
        self._release_traces()
        self._closed = True


def _tensor_identity(value: Any) -> str:
    try:
        device_tensors = tuple(ttnn.get_device_tensors(value))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        device_tensors = (value,)
    addresses = []
    for tensor in device_tensors:
        try:
            addresses.append(str(int(tensor.buffer_address())))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return f"{type(value).__name__}:host:{id(value)}"
    return f"{type(value).__name__}:device:[{','.join(addresses)}]"


def _nested_identity(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_nested_identity(item) for item in value) + "]"
    return _tensor_identity(value)
