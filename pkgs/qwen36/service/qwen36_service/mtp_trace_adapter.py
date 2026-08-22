"""TTNN trace owner for one fixed-shape Qwen MTP draft step."""

# r[impl tenstorrent.qwen38_p150x2.mtp_draft_trace]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import ttnn
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.tt_transformers.tt.common import copy_host_to_device

from .mtp_trace_core import MtpTraceBinding, validate_mtp_trace_binding

TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = True
EXPECTED_DEVICE_INPUT_COUNT = 4
MUTABLE_DEVICE_INPUT_COUNT = 3
EXPECTED_DECODE_STEP_COUNT = 2
MINIMUM_TOKEN_ID = 0
MINIMUM_POSITION = 0
ZERO_DURATION_SECONDS = 0.0


class MtpTraceAdapterError(RuntimeError):
    """The MTP draft trace cannot continue safely."""


@dataclass(frozen=True)
class MtpTraceReplayResult:
    token_ids: tuple[int, ...]
    trace_capture_seconds: float
    draft_decode_seconds: float
    draft_selection_seconds: float
    binding: MtpTraceBinding
    trace_binding_valid: bool
    capture_idempotence_valid: bool
    mtp_kv_prefix_valid: bool


class TracedMtpDraftAdapter:
    """Own one short-lived trace with stable MTP draft inputs and outputs."""

    def __init__(
        self,
        *,
        owner: Any,
        maximum_trace_capture_seconds: float,
    ) -> None:
        if (
            not math.isfinite(maximum_trace_capture_seconds)
            or maximum_trace_capture_seconds <= ZERO_DURATION_SECONDS
        ):
            raise MtpTraceAdapterError("MTP trace capture bound is invalid")
        self.owner = owner
        self.model = owner.model
        self.decode_adapter = owner.decode_adapter
        self.mesh = owner.mesh
        self.maximum_trace_capture_seconds = maximum_trace_capture_seconds
        self._mesh_reader = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        self._trace_id: Any | None = None
        self._token_buffer: Any | None = None
        self._position_buffer: Any | None = None
        self._rope_buffer: Any | None = None
        self._feedback_buffer: Any | None = None
        self._trace_feedback: Any | None = None
        self._trace_indices: Any | None = None
        self._trace_values: Any | None = None
        self._expected_binding: MtpTraceBinding | None = None
        self._closed = False

    def run(
        self,
        *,
        initial_token: int,
        initial_feedback: Any,
        start_position: int,
        step_count: int = EXPECTED_DECODE_STEP_COUNT,
    ) -> MtpTraceReplayResult:
        """Capture once, replay each draft step, and return bounded winners."""
        self._require_open()
        if step_count != EXPECTED_DECODE_STEP_COUNT:
            raise MtpTraceAdapterError("MTP trace requires exactly two decode steps")
        self._validate_token(initial_token)
        if start_position < MINIMUM_POSITION:
            raise MtpTraceAdapterError("MTP trace start position is negative")
        if start_position + step_count > self.model.args.max_seq_len:
            raise MtpTraceAdapterError("MTP trace exceeds the sequence bound")

        self._allocate_inputs(
            token=initial_token,
            position=start_position,
            feedback=initial_feedback,
        )
        capture_token, capture_seconds, capture_exact, cache_exact = self._capture(
            token=initial_token,
            position=start_position,
            feedback=initial_feedback,
        )
        assert self._expected_binding is not None

        tokens: list[int] = []
        selection_seconds = ZERO_DURATION_SECONDS
        current_token = initial_token
        current_feedback = initial_feedback
        replay_started = time.monotonic()
        try:
            for step in range(step_count):
                position = start_position + step
                self._update_inputs(token=current_token, position=position)
                assert self._feedback_buffer is not None
                ttnn.copy(current_feedback, self._feedback_buffer)
                self._require_stable_binding("before-replay")
                assert self._trace_id is not None
                ttnn.execute_trace(
                    self.mesh,
                    self._trace_id,
                    cq_id=TRACE_COMMAND_QUEUE_ID,
                    blocking=TRACE_BLOCKING_REPLAY,
                )
                ttnn.synchronize_device(self.mesh)
                selection_started = time.monotonic()
                assert self._trace_indices is not None
                assert self._trace_values is not None
                current_token = self.decode_adapter._read_global_winner(
                    self._trace_indices,
                    self._trace_values,
                )
                selection_seconds += time.monotonic() - selection_started
                self._validate_token(current_token)
                self._require_stable_binding("after-replay")
                if step == 0 and current_token != capture_token:
                    capture_exact = False
                tokens.append(current_token)
                assert self._trace_feedback is not None
                current_feedback = self._trace_feedback
            draft_decode_seconds = time.monotonic() - replay_started
            if (
                not math.isfinite(draft_decode_seconds)
                or draft_decode_seconds <= ZERO_DURATION_SECONDS
            ):
                raise MtpTraceAdapterError("MTP trace replay duration is invalid")
            if (
                not math.isfinite(selection_seconds)
                or selection_seconds <= ZERO_DURATION_SECONDS
                or selection_seconds > draft_decode_seconds
            ):
                raise MtpTraceAdapterError("MTP trace readback duration is invalid")
            return MtpTraceReplayResult(
                token_ids=tuple(tokens),
                trace_capture_seconds=capture_seconds,
                draft_decode_seconds=draft_decode_seconds,
                draft_selection_seconds=selection_seconds,
                binding=self._expected_binding,
                trace_binding_valid=True,
                capture_idempotence_valid=capture_exact,
                mtp_kv_prefix_valid=cache_exact,
            )
        finally:
            self.close()

    def _allocate_inputs(self, *, token: int, position: int, feedback: Any) -> None:
        try:
            host_inputs = self.model.prepare_decode_inputs_host(
                torch.tensor([[token]], dtype=torch.int32),
                torch.tensor([position], dtype=torch.int32),
                page_table=None,
            )
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh)
            if len(device_inputs) != EXPECTED_DEVICE_INPUT_COUNT:
                _deallocate_present(device_inputs)
                raise MtpTraceAdapterError("MTP trace model inputs are incomplete")
            (
                self._token_buffer,
                self._position_buffer,
                self._rope_buffer,
                unused_page_table,
            ) = device_inputs
            if unused_page_table is not None:
                ttnn.deallocate(unused_page_table)
            self._feedback_buffer = ttnn.clone(feedback)
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.close()
            if isinstance(error, MtpTraceAdapterError):
                raise
            raise MtpTraceAdapterError("MTP trace input allocation failed") from error

    def _capture(
        self,
        *,
        token: int,
        position: int,
        feedback: Any,
    ) -> tuple[int, float, bool, bool]:
        capture_started = time.monotonic()
        candidate_trace_id: Any | None = None
        warm_outputs: tuple[Any, Any, Any] | None = None
        try:
            self._update_inputs(token=token, position=position)
            assert self._feedback_buffer is not None
            ttnn.copy(feedback, self._feedback_buffer)
            warm_outputs = self._forward_device()
            ttnn.synchronize_device(self.mesh)
            warm_feedback, warm_indices, warm_values = warm_outputs
            warm_token = self.decode_adapter._read_global_winner(
                warm_indices,
                warm_values,
            )
            warm_feedback_host = self._to_host(warm_feedback)
            warm_cache_row = self._snapshot_cache_row(position)
            self._deallocate_outputs(warm_outputs)
            warm_outputs = None

            candidate_trace_id = ttnn.begin_trace_capture(
                self.mesh,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            (
                self._trace_feedback,
                self._trace_indices,
                self._trace_values,
            ) = self._forward_device()
            ttnn.end_trace_capture(
                self.mesh,
                candidate_trace_id,
                cq_id=TRACE_COMMAND_QUEUE_ID,
            )
            ttnn.synchronize_device(self.mesh)
            self._trace_id = candidate_trace_id
            candidate_trace_id = None
            capture_seconds = time.monotonic() - capture_started
            if (
                not math.isfinite(capture_seconds)
                or capture_seconds <= ZERO_DURATION_SECONDS
                or capture_seconds > self.maximum_trace_capture_seconds
            ):
                raise MtpTraceAdapterError("MTP trace capture exceeded its bound")

            assert self._trace_feedback is not None
            assert self._trace_indices is not None
            assert self._trace_values is not None
            capture_token = self.decode_adapter._read_global_winner(
                self._trace_indices,
                self._trace_values,
            )
            capture_feedback_host = self._to_host(self._trace_feedback)
            capture_cache_row = self._snapshot_cache_row(position)
            capture_exact = warm_token == capture_token and torch.equal(
                warm_feedback_host,
                capture_feedback_host,
            )
            cache_exact = _tensor_sequences_equal(
                warm_cache_row,
                capture_cache_row,
            )
            self._expected_binding = self._observe_binding()
            return capture_token, capture_seconds, capture_exact, cache_exact
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
            if candidate_trace_id is not None and self._trace_id is None:
                try:
                    ttnn.release_trace(self.mesh, candidate_trace_id)
                except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            self.close()
            if isinstance(error, MtpTraceAdapterError):
                raise
            raise MtpTraceAdapterError("MTP trace capture failed") from error
        finally:
            if warm_outputs is not None:
                self._deallocate_outputs(warm_outputs)

    def _forward_device(self) -> tuple[Any, Any, Any]:
        assert self._token_buffer is not None
        assert self._position_buffer is not None
        assert self._rope_buffer is not None
        assert self._feedback_buffer is not None
        embedding = None
        fused = None
        output = None
        normalized = None
        logits = None
        indices = None
        values = None
        try:
            embedding = self.owner._embed_full_device(self._token_buffer)
            fused = self.owner._fuse_inputs(embedding, self._feedback_buffer)
            ttnn.deallocate(embedding)
            embedding = None
            cos, sin = unpack_rope(self._rope_buffer)
            output = self.owner.mtp_layer.forward(
                fused,
                cos=cos,
                sin=sin,
                mode="decode",
                position_tensor=self._position_buffer,
            )
            ttnn.deallocate(fused)
            fused = None
            normalized = self.owner._final_normalize(output)
            ttnn.deallocate(output)
            output = None
            logits = ttnn.linear(normalized, self.model.lm_head_weight)
            indices, values = self.decode_adapter._reduce_shards(logits)
            ttnn.deallocate(logits)
            logits = None
            result = (normalized, indices, values)
            normalized = None
            indices = None
            values = None
            return result
        finally:
            _deallocate_present((
                embedding,
                fused,
                output,
                normalized,
                logits,
                indices,
                values,
            ))

    def _update_inputs(self, *, token: int, position: int) -> None:
        self._validate_token(token)
        host_inputs = self.model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=None,
        )
        device_inputs = (
            self._token_buffer,
            self._position_buffer,
            self._rope_buffer,
        )
        if any(tensor is None for tensor in device_inputs):
            raise MtpTraceAdapterError("MTP trace inputs are not allocated")
        copy_host_to_device(
            host_inputs[:MUTABLE_DEVICE_INPUT_COUNT],
            device_tensors=device_inputs,
        )

    def _snapshot_cache_row(self, position: int) -> tuple[torch.Tensor, ...]:
        attention = self.owner.mtp_layer.attention
        caches = tuple(attention.k_caches or ()) + tuple(attention.v_caches or ())
        if not caches:
            raise MtpTraceAdapterError("MTP trace cache is not allocated")
        rows = []
        for cache in caches:
            if len(cache.shape) != EXPECTED_DEVICE_INPUT_COUNT:
                raise MtpTraceAdapterError("MTP trace cache rank changed")
            stop = position + 1
            if stop > int(cache.shape[2]):
                raise MtpTraceAdapterError(
                    "MTP trace cache position is outside capacity"
                )
            row = ttnn.slice(
                cache,
                (0, 0, position, 0),
                (
                    int(cache.shape[0]),
                    int(cache.shape[1]),
                    stop,
                    int(cache.shape[3]),
                ),
            )
            try:
                rows.append(self._to_host(row))
            finally:
                ttnn.deallocate(row)
        return tuple(rows)

    def _to_host(self, tensor: Any) -> torch.Tensor:
        return ttnn.to_torch(tensor, mesh_composer=self._mesh_reader).clone()

    def _observe_binding(self) -> MtpTraceBinding:
        required = (
            self._token_buffer,
            self._position_buffer,
            self._rope_buffer,
            self._feedback_buffer,
            self._trace_feedback,
            self._trace_indices,
            self._trace_values,
        )
        if any(tensor is None for tensor in required):
            raise MtpTraceAdapterError("MTP trace binding is incomplete")
        attention = self.owner.mtp_layer.attention
        caches = tuple(attention.k_caches or ()) + tuple(attention.v_caches or ())
        if not caches:
            raise MtpTraceAdapterError("MTP trace binding has no cache")
        assert self._token_buffer is not None
        assert self._position_buffer is not None
        assert self._rope_buffer is not None
        assert self._feedback_buffer is not None
        assert self._trace_feedback is not None
        assert self._trace_indices is not None
        assert self._trace_values is not None
        return MtpTraceBinding(
            token_buffer_identity=_identity(self._token_buffer),
            position_buffer_identity=_identity(self._position_buffer),
            rope_buffer_identity=_identity(self._rope_buffer),
            feedback_buffer_identity=_identity(self._feedback_buffer),
            kv_buffer_identity=_nested_identity(caches),
            feedback_output_identity=_identity(self._trace_feedback),
            index_output_identity=_identity(self._trace_indices),
            value_output_identity=_identity(self._trace_values),
            feedback_output_shape=_shape(self._trace_feedback),
            feedback_output_dtype=str(self._trace_feedback.dtype),
            index_output_shape=_shape(self._trace_indices),
            index_output_dtype=str(self._trace_indices.dtype),
            value_output_shape=_shape(self._trace_values),
            value_output_dtype=str(self._trace_values.dtype),
        )

    def _require_stable_binding(self, stage: str) -> None:
        if self._expected_binding is None:
            raise MtpTraceAdapterError("MTP trace expected binding is missing")
        mismatches = validate_mtp_trace_binding(
            self._expected_binding,
            self._observe_binding(),
        )
        if mismatches:
            raise MtpTraceAdapterError(
                f"MTP trace binding changed {stage}: {','.join(mismatches)}"
            )

    def _validate_token(self, token: int) -> None:
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or not MINIMUM_TOKEN_ID <= token < self.model.args.vocab_size
        ):
            raise MtpTraceAdapterError("MTP trace token is outside the vocabulary")

    @staticmethod
    def _deallocate_outputs(outputs: tuple[Any, Any, Any]) -> None:
        feedback, indices, values = outputs
        ttnn.deallocate(indices)
        ttnn.deallocate(values)
        ttnn.deallocate(feedback)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh, self._trace_id)
            self._trace_id = None
        _deallocate_present((
            self._trace_indices,
            self._trace_values,
            self._trace_feedback,
            self._feedback_buffer,
            self._rope_buffer,
            self._position_buffer,
            self._token_buffer,
        ))
        self._trace_indices = None
        self._trace_values = None
        self._trace_feedback = None
        self._feedback_buffer = None
        self._rope_buffer = None
        self._position_buffer = None
        self._token_buffer = None
        self._expected_binding = None

    def _require_open(self) -> None:
        if self._closed:
            raise MtpTraceAdapterError("MTP trace adapter is closed")


def _deallocate_present(tensors: Any) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor)


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


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


def _tensor_sequences_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    return len(left) == len(right) and all(
        torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left, right, strict=True)
    )
