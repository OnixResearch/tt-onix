"""TTNN adapter for the Qwen P150x2 traced-decode core."""

# r[impl tenstorrent.qwen38_p150x2.traced_decode]
# r[impl tenstorrent.qwen38_p150x2.persistent_decode_buffers]
# r[impl tenstorrent.qwen38_p150x2.shard_local_argmax]

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.tt_transformers.tt.common import copy_host_to_device

from .decode_core import (
    DecodeProfile,
    ShardWinner,
    TraceBinding,
    select_global_winner,
    validate_trace_binding,
)

TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS = 32
MAX_REDUCTION_COLUMNS = 32
TILE_EDGE_ELEMENTS = 32
PADDED_LOGIT_SENTINEL = -1.0e30
TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = False


class DecodeAdapterError(RuntimeError):
    """The TTNN decode adapter cannot continue safely."""


@dataclass(frozen=True)
class TraceEligibility:
    eligible: bool
    reason: str


class TracedDecodeAdapter:
    """Own stable TTNN buffers and one fail-closed batch-one decode trace."""

    def __init__(
        self,
        *,
        model: Any,
        profile: DecodeProfile,
        runtime_identity: str,
        package_identity: str,
        model_identity: str,
        slot_id: str,
        buffer_generation: int,
    ) -> None:
        if model.num_devices <= 1:
            raise DecodeAdapterError("traced TP2 decode requires more than one device")
        if model.args.vocab_size % model.num_devices != 0:
            raise DecodeAdapterError("vocabulary size is not divisible by the device count")
        self.model = model
        self.profile = profile
        self.runtime_identity = runtime_identity
        self.package_identity = package_identity
        self.model_identity = model_identity
        self.slot_id = slot_id
        self.buffer_generation = buffer_generation
        self.mesh = model.mesh_device
        self.vocabulary_size = int(model.args.vocab_size)
        self.device_count = int(model.num_devices)
        self.vocabulary_per_shard = self.vocabulary_size // self.device_count
        self.page_table = self._make_page_table()
        self._allocate_persistent_model_state()
        self.decode_inputs = model.prepare_inputs_decode(
            torch.zeros((1, 1), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            page_table=self.page_table,
        )
        self._mesh_reader = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        self._trace_id: Any | None = None
        self._trace_logits: Any | None = None
        self._trace_indices: Any | None = None
        self._trace_values: Any | None = None
        self._expected_binding: TraceBinding | None = None
        self._closed = False

    def _make_page_table(self) -> torch.Tensor:
        block_count = (
            self.profile.maximum_sequence_length + self.profile.kv_block_size - 1
        ) // self.profile.kv_block_size
        aligned_block_count = (
            (block_count + TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS - 1)
            // TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS
        ) * TRACE_PAGE_TABLE_ALIGNMENT_BLOCKS
        return torch.arange(aligned_block_count, dtype=torch.int32).reshape(
            1, aligned_block_count
        )

    def _allocate_persistent_model_state(self) -> None:
        cache_shape = [
            int(self.page_table.shape[1]),
            int(self.model.args.n_local_kv_heads),
            self.profile.kv_block_size,
            int(self.model.args.head_dim),
        ]
        self.model.allocate_kv_caches(
            cache_shape,
            ttnn.bfloat16,
            batch_size=self.profile.slot_count,
        )

    def reset_for_request(self) -> None:
        """Clear logical recurrent state while preserving every device address."""
        self._require_open()
        if self._expected_binding is not None:
            mismatches = validate_trace_binding(
                self._expected_binding, self._observe_binding()
            )
            if mismatches:
                raise DecodeAdapterError(
                    "reset found a stale trace binding: " + ",".join(mismatches)
                )
        self.model._reset_gdn_state_for_new_sequence()
        ttnn.synchronize_device(self.mesh)

    def ensure_trace(self, *, token: int, position: int) -> TraceEligibility:
        """Capture once after prefill, or validate the existing trace binding."""
        self._require_open()
        if not self.profile.trace_enabled:
            return TraceEligibility(False, "profile-disabled")
        if self._trace_id is not None:
            assert self._expected_binding is not None
            mismatches = validate_trace_binding(
                self._expected_binding, self._observe_binding()
            )
            if mismatches:
                return TraceEligibility(False, "binding-mismatch:" + ",".join(mismatches))
            return TraceEligibility(True, "trace-ready")
        self._capture_trace(token=token, position=position)
        return TraceEligibility(True, "trace-captured")

    def _capture_trace(self, *, token: int, position: int) -> None:
        self._update_inputs(token=token, position=position)
        saved_state = self.model._save_deltanet_states()
        capture_started = time.monotonic()
        candidate_trace_id: Any | None = None
        try:
            warm_logits = self._decode_forward_sharded()
            warm_indices, warm_values = self._reduce_shards(warm_logits)
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(warm_indices)
            ttnn.deallocate(warm_values)
            ttnn.deallocate(warm_logits)

            candidate_trace_id = ttnn.begin_trace_capture(
                self.mesh, cq_id=TRACE_COMMAND_QUEUE_ID
            )
            trace_logits = self._decode_forward_sharded()
            trace_indices, trace_values = self._reduce_shards(trace_logits)
            ttnn.end_trace_capture(
                self.mesh,
                candidate_trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            ttnn.synchronize_device(self.mesh)
            capture_elapsed = time.monotonic() - capture_started
            if capture_elapsed > self.profile.trace_capture_timeout_seconds:
                raise DecodeAdapterError("decode trace capture exceeded the profile timeout")
            self._trace_id = candidate_trace_id
            self._trace_logits = trace_logits
            self._trace_indices = trace_indices
            self._trace_values = trace_values
            self._expected_binding = self._observe_binding()
        except (
            AssertionError,
            DecodeAdapterError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            if candidate_trace_id is not None and self._trace_id is None:
                try:
                    ttnn.release_trace(self.mesh, candidate_trace_id)
                except (AssertionError, OSError, RuntimeError, ValueError):
                    pass
            raise DecodeAdapterError("decode trace capture failed") from error
        finally:
            self.model._restore_deltanet_states(saved_state, self.mesh)
            ttnn.synchronize_device(self.mesh)

    def decode_traced(self, *, token: int, position: int) -> int:
        """Refresh stable inputs, replay once, and read only bounded shard winners."""
        eligibility = self.ensure_trace(token=token, position=position)
        if not eligibility.eligible:
            raise DecodeAdapterError(eligibility.reason)
        assert self._trace_id is not None
        assert self._trace_indices is not None
        assert self._trace_values is not None
        self._update_inputs(token=token, position=position)
        ttnn.execute_trace(
            self.mesh,
            self._trace_id,
            cq_id=TRACE_COMMAND_QUEUE_ID,
            blocking=TRACE_BLOCKING_REPLAY,
        )
        ttnn.synchronize_device(self.mesh)
        return self._read_global_winner(self._trace_indices, self._trace_values)

    def decode_eager(self, *, token: int, position: int) -> int:
        """Use the named full-logit fallback with the same persistent model state."""
        self._require_open()
        self._update_inputs(token=token, position=position)
        self.model._ondev_argmax = False
        output, _ = self.model.ttnn_decode_forward(
            self.decode_inputs[0],
            self.decode_inputs[1],
            rot_mat_idxs=self.decode_inputs[2],
            page_table=self.decode_inputs[3],
        )
        logits = self.model.process_output_decode(output, self.profile.slot_count)
        token_index = int(
            torch.argmax(logits.reshape(-1)[: self.vocabulary_size].float()).item()
        )
        if not 0 <= token_index < self.vocabulary_size:
            raise DecodeAdapterError("eager fallback produced an out-of-range token")
        return token_index

    def _update_inputs(self, *, token: int, position: int) -> None:
        if not 0 <= token < self.vocabulary_size:
            raise DecodeAdapterError("decode token is outside the vocabulary")
        if not 0 <= position < self.profile.maximum_sequence_length:
            raise DecodeAdapterError("decode position is outside the profile bound")
        host_inputs = self.model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=None,
        )
        mutable_input_count = 3
        copy_host_to_device(
            host_inputs[:mutable_input_count],
            device_tensors=self.decode_inputs[:mutable_input_count],
        )

    def _decode_forward_sharded(self) -> Any:
        self.model._ondev_argmax = True
        output, _ = self.model.ttnn_decode_forward(
            self.decode_inputs[0],
            self.decode_inputs[1],
            rot_mat_idxs=self.decode_inputs[2],
            page_table=self.decode_inputs[3],
        )
        return output

    def _reduce_shards(self, sharded_logits: Any) -> tuple[Any, Any]:
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
            padded, (1, 1, padded_rows, MAX_REDUCTION_COLUMNS)
        )
        partial = ttnn.max(grid, dim=-1)
        partial_row = ttnn.reshape(partial, (1, 1, 1, padded_rows))
        values = ttnn.max(partial_row, dim=-1)
        ttnn.deallocate(padded)
        ttnn.deallocate(grid)
        ttnn.deallocate(partial)
        ttnn.deallocate(partial_row)
        return indices, values

    def _read_global_winner(self, indices: Any, values: Any) -> int:
        host_indices = ttnn.to_torch(
            indices, mesh_composer=self._mesh_reader
        ).reshape(-1)
        host_values = ttnn.to_torch(
            values, mesh_composer=self._mesh_reader
        ).reshape(-1)
        if len(host_indices) != self.device_count or len(host_values) != self.device_count:
            raise DecodeAdapterError("shard winner readback has the wrong device count")
        winners = []
        for device_index in range(self.device_count):
            shard_start = device_index * self.vocabulary_per_shard
            local_token = int(host_indices[device_index].item())
            winners.append(
                ShardWinner(
                    shard_start=shard_start,
                    shard_stop=shard_start + self.vocabulary_per_shard,
                    token_index=shard_start + local_token,
                    maximum_logit=float(host_values[device_index].item()),
                )
            )
        return select_global_winner(winners, self.vocabulary_size)

    def _observe_binding(self) -> TraceBinding:
        if self._trace_logits is None:
            raise DecodeAdapterError("trace output is not available")
        return TraceBinding(
            runtime_identity=self.runtime_identity,
            package_identity=self.package_identity,
            model_identity=self.model_identity,
            profile_id=self.profile.profile_id,
            slot_id=self.slot_id,
            buffer_generation=self.buffer_generation,
            token_buffer_identity=_identity(self.decode_inputs[0]),
            position_buffer_identity=_identity(self.decode_inputs[1]),
            rope_buffer_identity=_identity(self.decode_inputs[2]),
            page_table_buffer_identity=_identity(self.decode_inputs[3]),
            kv_buffer_identity=_nested_identity(self.model._paged_kv_caches),
            gdn_buffer_identity=_nested_identity(self.model._deltanet_external_states),
            output_buffer_identity=_identity(self._trace_logits),
            output_shape=tuple(int(dimension) for dimension in self._trace_logits.shape),
            output_dtype=str(self._trace_logits.dtype),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh, self._trace_id)
            self._trace_id = None
        for tensor_name in ("_trace_indices", "_trace_values", "_trace_logits"):
            tensor = getattr(self, tensor_name)
            if tensor is not None:
                ttnn.deallocate(tensor)
                setattr(self, tensor_name, None)
        self.model._ondev_argmax = False

    def _require_open(self) -> None:
        if self._closed:
            raise DecodeAdapterError("decode adapter is closed")


def _identity(value: Any) -> str:
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
    return _identity(value)
