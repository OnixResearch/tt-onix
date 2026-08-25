"""Concurrent request coordinator for isolated Qwen batch decode."""

# r[impl tenstorrent.qwen38_p150x2.batched_concurrency]

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .batch_decode_adapter import (
    BatchDecodeAdapterError,
    BatchedTracedDecodeAdapter,
)
from .batch_decode_core import ActiveDecodeUser, plan_decode_batch
from .core import CompletionRequest, build_completion_response
from .decode_adapter import DecodeAdapterError
from .decode_core import DecodeProfile, request_deadline_expired
from .readiness_core import StartupWarmupPlan

MAXIMUM_BATCH_WIDTH = 16
BATCH_COLLECTION_SECONDS = 0.01
WORKER_JOIN_TIMEOUT_SECONDS = 5.0
EXPECTED_PROMPT_RANK = 2


@dataclass
class _PendingBatchRequest:
    request: CompletionRequest
    request_id: str
    token_ids: torch.Tensor
    prompt_token_count: int
    started: float
    deadline: float
    completed: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class WarmupWidthObservation:
    width: int
    elapsed_seconds: float
    prefill_seconds: float
    trace_capture_seconds: float
    replay_seconds: float
    reset_seconds: float


class BatchGenerationCoordinator:
    """Collect compatible requests and run one isolated decode step per batch."""

    def __init__(
        self,
        *,
        adapter: BatchedTracedDecodeAdapter,
        tokenizer: Any,
        model: Any,
        profile: DecodeProfile,
        model_alias: str,
        model_identity: str,
        package_identity: str,
        runtime_identity: str,
        maximum_sequence_length: int,
        physical_pcie_device_ids: tuple[int, ...],
        emitter: Callable[..., None],
    ) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.model = model
        self.profile = profile
        self.model_alias = model_alias
        self.model_identity = model_identity
        self.package_identity = package_identity
        self.runtime_identity = runtime_identity
        self.maximum_sequence_length = maximum_sequence_length
        self.physical_pcie_device_ids = physical_pcie_device_ids
        self.emitter = emitter
        self._condition = threading.Condition()
        self._queue: deque[_PendingBatchRequest] = deque()
        self._accepting_requests = False
        self._closing = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="qwen36-batch-decode",
            daemon=True,
        )
        self._worker.start()

    def generate(
        self,
        *,
        request: CompletionRequest,
        request_id: str,
        token_ids: torch.Tensor,
        prompt_token_count: int,
        started: float,
        deadline: float,
    ) -> dict[str, Any]:
        pending = _PendingBatchRequest(
            request=request,
            request_id=request_id,
            token_ids=token_ids,
            prompt_token_count=prompt_token_count,
            started=started,
            deadline=deadline,
        )
        with self._condition:
            if self._closing:
                raise BatchDecodeAdapterError("batch worker is closed")
            if not self._accepting_requests:
                raise BatchDecodeAdapterError("batch worker is not ready")
            self._queue.append(pending)
            self._condition.notify_all()
        pending.completed.wait()
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise BatchDecodeAdapterError("batch worker returned no response")
        return pending.response

    def warm_up(
        self,
        plan: StartupWarmupPlan,
        token_ids: torch.Tensor,
    ) -> tuple[WarmupWidthObservation, ...]:
        """Execute bounded TTNN warm-up before the worker accepts requests."""
        with self._condition:
            if self._closing:
                raise BatchDecodeAdapterError("batch worker is closed")
            if self._accepting_requests:
                raise BatchDecodeAdapterError("batch worker is already ready")
            if self._queue:
                raise BatchDecodeAdapterError("batch worker queued before warm-up")
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.ndim != EXPECTED_PROMPT_RANK
        ):
            raise BatchDecodeAdapterError("warm-up tokens must be rank two")
        if token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
            raise BatchDecodeAdapterError("warm-up tokens must contain one prompt")

        prompt_token_count = int(token_ids.shape[1])
        observations = []
        try:
            for width in plan.widths:
                width_started = time.monotonic()
                prefill_result = self.adapter.prefill_users(
                    tuple(token_ids.clone() for _ in range(width)),
                    generation_tokens=plan.generation_tokens,
                )
                positions = tuple(prompt_token_count for _ in range(width))
                trace_capture_seconds = self.adapter.prepare_width(
                    token_ids=prefill_result.first_token_ids,
                    positions=positions,
                )
                replay_started = time.monotonic()
                replay_tokens = self.adapter.decode_traced(
                    token_ids=prefill_result.first_token_ids,
                    positions=positions,
                )
                replay_seconds = time.monotonic() - replay_started
                if len(replay_tokens) != width:
                    raise BatchDecodeAdapterError(
                        "startup warm-up replay width changed"
                    )
                reset_started = time.monotonic()
                self.adapter.reset_all()
                reset_seconds = time.monotonic() - reset_started
                observation = WarmupWidthObservation(
                    width=width,
                    elapsed_seconds=time.monotonic() - width_started,
                    prefill_seconds=prefill_result.elapsed_seconds,
                    trace_capture_seconds=trace_capture_seconds,
                    replay_seconds=replay_seconds,
                    reset_seconds=reset_seconds,
                )
                observations.append(observation)
                self.emitter(
                    "startup-warmup-width-passed",
                    elapsed_seconds=observation.elapsed_seconds,
                    prefill_seconds=observation.prefill_seconds,
                    replay_seconds=observation.replay_seconds,
                    reset_seconds=observation.reset_seconds,
                    trace_capture_seconds=observation.trace_capture_seconds,
                    width=width,
                )
        except (
            AssertionError,
            BatchDecodeAdapterError,
            DecodeAdapterError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            try:
                self.adapter.reset_all()
            except (
                AssertionError,
                BatchDecodeAdapterError,
                OSError,
                RuntimeError,
                ValueError,
            ) as reset_error:
                raise BatchDecodeAdapterError(
                    "startup warm-up reset failed"
                ) from reset_error
            raise BatchDecodeAdapterError("startup warm-up failed") from error

        with self._condition:
            self._accepting_requests = True
            self._condition.notify_all()
        return tuple(observations)

    def _take_batch(self) -> tuple[_PendingBatchRequest, ...] | None:
        with self._condition:
            while not self._queue and not self._closing:
                self._condition.wait()
            if self._closing and not self._queue:
                return None
            first = self._queue.popleft()
            selected = [first]
            collection_deadline = time.monotonic() + BATCH_COLLECTION_SECONDS
            while len(selected) < MAXIMUM_BATCH_WIDTH:
                compatible = next(
                    (
                        pending
                        for pending in self._queue
                        if pending.request.max_tokens == first.request.max_tokens
                    ),
                    None,
                )
                if compatible is not None:
                    self._queue.remove(compatible)
                    selected.append(compatible)
                    continue
                remaining = collection_deadline - time.monotonic()
                if remaining <= 0 or self._closing:
                    break
                self._condition.wait(timeout=remaining)
            return tuple(selected)

    def _worker_loop(self) -> None:
        while True:
            pending_batch = self._take_batch()
            if pending_batch is None:
                return
            try:
                responses = self._run_group(pending_batch)
                for pending, response in zip(
                    pending_batch,
                    responses,
                    strict=True,
                ):
                    pending.response = response
            except (
                AssertionError,
                BatchDecodeAdapterError,
                DecodeAdapterError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                terminal_error: Exception = error
                try:
                    self.adapter.reset_all()
                except (
                    AssertionError,
                    BatchDecodeAdapterError,
                    OSError,
                    RuntimeError,
                    ValueError,
                ) as reset_error:
                    terminal_error = BatchDecodeAdapterError(
                        "batch reset failed after request error"
                    )
                    terminal_error.__cause__ = reset_error
                self.emitter(
                    "batch-failed",
                    batch_width=len(pending_batch),
                    error_type=type(terminal_error).__name__,
                )
                for pending in pending_batch:
                    pending.error = terminal_error
            finally:
                for pending in pending_batch:
                    pending.completed.set()

    def _run_group(
        self,
        pending_batch: tuple[_PendingBatchRequest, ...],
    ) -> tuple[dict[str, Any], ...]:
        width = len(pending_batch)
        maximum_generation_tokens = pending_batch[0].request.max_tokens
        if any(
            pending.request.max_tokens != maximum_generation_tokens
            for pending in pending_batch
        ):
            raise BatchDecodeAdapterError("batch generation bounds differ")
        if any(
            request_deadline_expired(time.monotonic(), pending.deadline)
            for pending in pending_batch
        ):
            raise BatchDecodeAdapterError("request timeout exceeded before batch start")

        prefill_result = self.adapter.prefill_users(
            tuple(pending.token_ids for pending in pending_batch),
            generation_tokens=maximum_generation_tokens,
        )
        generated = [[token] for token in prefill_result.first_token_ids]
        positions = [pending.prompt_token_count for pending in pending_batch]
        trace_prepare_started = time.monotonic()
        trace_capture_seconds = self.adapter.prepare_width(
            token_ids=prefill_result.first_token_ids,
            positions=tuple(positions),
        )
        trace_prepare_seconds = time.monotonic() - trace_prepare_started
        prefill_finished = time.monotonic()

        decode_durations = []
        while len(generated[0]) < maximum_generation_tokens:
            if any(
                request_deadline_expired(time.monotonic(), pending.deadline)
                for pending in pending_batch
            ):
                raise BatchDecodeAdapterError("request timeout exceeded")
            plan = plan_decode_batch(
                tuple(
                    ActiveDecodeUser(
                        request_id=pending.request_id,
                        token_id=generated[row_index][-1],
                        position=positions[row_index],
                        state_slice=row_index,
                    )
                    for row_index, pending in enumerate(pending_batch)
                ),
                vocabulary_size=self.model.args.vocab_size,
                maximum_sequence_length=self.maximum_sequence_length,
            )
            decode_started = time.monotonic()
            next_tokens = self.adapter.decode_traced(
                token_ids=tuple(row.token_id for row in plan.rows),
                positions=tuple(row.position for row in plan.rows),
            )
            decode_durations.append(time.monotonic() - decode_started)
            for row, next_token in zip(plan.rows, next_tokens, strict=True):
                generated[row.row_index].append(next_token)
                positions[row.row_index] = row.next_position

        cleanup_started = time.monotonic()
        self.adapter.reset_all()
        cleanup_seconds = time.monotonic() - cleanup_started
        steady_durations = (
            decode_durations[1:] if len(decode_durations) > 1 else decode_durations
        )
        if steady_durations:
            average_decode_seconds = sum(steady_durations) / len(steady_durations)
            per_user_decode_rate = 1.0 / average_decode_seconds
            aggregate_decode_rate = width / average_decode_seconds
            tail_token_latency_seconds = max(steady_durations)
        else:
            per_user_decode_rate = 0.0
            aggregate_decode_rate = 0.0
            tail_token_latency_seconds = 0.0

        responses = []
        for state_slice, (pending, user_tokens) in enumerate(
            zip(pending_batch, generated, strict=True)
        ):
            generated_text = self.tokenizer.decode(
                user_tokens,
                skip_special_tokens=True,
            )
            response = build_completion_response(
                request=pending.request,
                request_id=pending.request_id,
                created=int(time.time()),
                generated_text=generated_text,
                prompt_token_count=pending.prompt_token_count,
                generated_token_count=len(user_tokens),
            )
            ttft_seconds = prefill_finished - pending.started
            response["x_onix_decode_evidence"] = {
                "admission_eligible": False,
                "aggregate_decode_tokens_per_second": aggregate_decode_rate,
                "batch_width": width,
                "clean_reset": True,
                "cleanup_seconds": cleanup_seconds,
                "decode_page_table_blocks": self.adapter.blocks_per_user,
                "decode_path": "batched-traced",
                "decode_tokens_per_second": per_user_decode_rate,
                "fallback_reason": None,
                "first_token_readback_seconds": None,
                "generated_token_ids": user_tokens,
                "measurement_mode": "physical-batched-service",
                "model_identity": self.model_identity,
                "package_identity": self.package_identity,
                "physical_pcie_device_ids": self.physical_pcie_device_ids,
                "prefill_device_seconds": prefill_result.elapsed_seconds,
                "prefill_page_table_blocks": prefill_result.page_table_blocks,
                "prefill_profile_receiver_drain_seconds": 0.0,
                "prefill_program_profile": None,
                "profile_id": self.profile.profile_id,
                "request_reset_seconds": 0.0,
                "runtime_identity": self.runtime_identity,
                "simulated": False,
                "state_slice": state_slice,
                "tail_token_latency_seconds": tail_token_latency_seconds,
                "token_latency_seconds": steady_durations,
                "trace_capture_seconds": trace_capture_seconds,
                "trace_prepare_seconds": trace_prepare_seconds,
                "ttft_seconds": ttft_seconds,
            }
            self.emitter(
                "request-passed",
                aggregate_decode_tokens_per_second=aggregate_decode_rate,
                batch_width=width,
                decode_path="batched-traced",
                decode_tokens_per_second=per_user_decode_rate,
                elapsed_seconds=time.monotonic() - pending.started,
                fallback_reason=None,
                generated_token_count=len(user_tokens),
                prompt_token_count=pending.prompt_token_count,
                request_id=pending.request_id,
                state_slice=state_slice,
                tail_token_latency_seconds=tail_token_latency_seconds,
                ttft_seconds=ttft_seconds,
            )
            responses.append(response)
        return tuple(responses)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._worker.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        if self._worker.is_alive():
            raise RuntimeError("batch worker did not stop")
