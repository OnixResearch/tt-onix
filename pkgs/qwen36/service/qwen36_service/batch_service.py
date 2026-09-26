"""Concurrent request coordinator for isolated Qwen batch decode."""

# r[impl tenstorrent.qwen38_p150x2.batched_concurrency]

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .batch_admission_core import (
    PRODUCTION_MEASUREMENT_MODE,
    BatchAdmissionPolicyError,
    BatchAdmissionProfile,
)
from .batch_decode_adapter import (
    BatchDecodeAdapterError,
    BatchedTracedDecodeAdapter,
    StructuredReadRows,
)
from .batch_decode_core import ActiveDecodeUser, plan_decode_batch
from .core import CompletionRequest, build_completion_response, finish_generation
from .decode_adapter import DecodeAdapterError
from .decode_core import DecodeProfile, request_deadline_expired
from .readiness_core import StartupWarmupPlan
from .sampled_decode_core import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    merge_shard_candidates,
    sample_candidates,
)
from .sampled_mode_router_core import (
    ROUTE_SAMPLED,
    classify_temperature,
    deterministic_route_for_mode,
    resolve_batch_route,
)
from .trace_reuse_core import TraceReuseProfile

MAXIMUM_BATCH_WIDTH = 16
BATCH_COLLECTION_SECONDS = 0.01
# Read jobs merged into one worker turn; each job holds one decision's reads.
MAXIMUM_MERGED_READ_JOBS = 16
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


@dataclass(eq=False)
class _PendingRead:
    """One decision's structured reads; the worker batches queued jobs into shared replays."""

    requests: tuple[tuple[torch.Tensor, tuple[int, ...]], ...]
    deadline: float
    completed: threading.Event = field(default_factory=threading.Event)
    results: tuple[StructuredReadRows, ...] | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class WarmupWidthObservation:
    width: int
    elapsed_seconds: float
    prefill_seconds: float
    trace_capture_seconds: float
    trace_reused: bool
    resident_trace_count: int
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
        batch_admission_profile: BatchAdmissionProfile | None = None,
        batch_admission_enabled: bool = False,
        trace_reuse_profile: TraceReuseProfile | None = None,
        trace_reuse_admission_enabled: bool = False,
        sampled_admission_enabled: bool = False,
        stop_token_ids: frozenset[int] = frozenset(),
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
        self.batch_admission_profile = batch_admission_profile
        self.sampled_admission_enabled = sampled_admission_enabled
        # A batch stops once every user has emitted one of these; an empty set generates
        # exactly max_tokens (admission measurements need the fixed length).
        self.stop_token_ids = stop_token_ids
        # Greedy requests stay on the mode the GDN layers started in when that
        # mode is the fused decode step; otherwise on the exact-parity route.
        self.deterministic_route = deterministic_route_for_mode(adapter.fusion_mode)
        self.batch_admission_enabled = batch_admission_enabled
        self.trace_reuse_profile = trace_reuse_profile
        self.trace_reuse_admission_enabled = trace_reuse_admission_enabled
        if batch_admission_enabled and batch_admission_profile is None:
            raise BatchDecodeAdapterError(
                "batch admission mode requires a batch admission profile"
            )
        if trace_reuse_admission_enabled and trace_reuse_profile is None:
            raise BatchDecodeAdapterError(
                "trace-reuse admission mode requires a trace-reuse profile"
            )
        if batch_admission_enabled and trace_reuse_admission_enabled:
            raise BatchDecodeAdapterError(
                "batch and trace-reuse admission modes conflict"
            )
        self._condition = threading.Condition()
        self._queue: deque[_PendingBatchRequest | _PendingRead] = deque()
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

    def read(
        self,
        *,
        requests: tuple[tuple[torch.Tensor, tuple[int, ...]], ...],
        deadline: float,
    ) -> tuple[StructuredReadRows, ...]:
        """Run structured reads on the device worker, between generation batches."""
        if not requests:
            raise BatchDecodeAdapterError("a structured read job needs at least one read")
        pending = _PendingRead(requests=requests, deadline=deadline)
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
        if pending.results is None:
            raise BatchDecodeAdapterError("batch worker returned no reads")
        return pending.results

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
                    trace_reuse_enabled=False,
                )
                positions = tuple(prompt_token_count for _ in range(width))
                trace_preparation = self.adapter.prepare_width(
                    token_ids=prefill_result.first_token_ids,
                    positions=positions,
                )
                # Read the tail that prepare_width captured. A sampled trace
                # contains top-K candidates, not one greedy winner per shard.
                replay = (
                    self.adapter.decode_traced_sampled
                    if self.adapter.fusion_mode == ROUTE_SAMPLED
                    else self.adapter.decode_traced
                )
                replay_started = time.monotonic()
                replay_outputs = replay(
                    token_ids=prefill_result.first_token_ids,
                    positions=positions,
                )
                replay_seconds = time.monotonic() - replay_started
                if len(replay_outputs) != width:
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
                    trace_capture_seconds=trace_preparation.capture_seconds,
                    trace_reused=trace_preparation.reused,
                    resident_trace_count=trace_preparation.resident_trace_count,
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
                    resident_trace_count=observation.resident_trace_count,
                    trace_capture_seconds=observation.trace_capture_seconds,
                    trace_reused=observation.trace_reused,
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
        # Masked prefill traces come last: their capture must follow every program the
        # worker can run. A capture failure leaves eager prefill in service.
        routes = (self.deterministic_route,) + (
            (ROUTE_SAMPLED,) if self.sampled_admission_enabled else ()
        )
        traces_started = time.monotonic()
        try:
            buckets = self.adapter.prepare_masked_prefill_traces(
                int(token_ids[0, 0]), routes
            )
        except (AssertionError, BatchDecodeAdapterError, RuntimeError, ValueError) as error:
            self.adapter.model.release_masked_prefill_traces()
            self.adapter.reset_all()
            self.emitter("masked-prefill-traces-failed", error_type=type(error).__name__)
        else:
            if buckets:
                self.emitter(
                    "masked-prefill-traces-captured",
                    buckets=list(buckets),
                    elapsed_seconds=time.monotonic() - traces_started,
                )

        with self._condition:
            self._accepting_requests = True
            self._condition.notify_all()
        return tuple(observations)

    def _take_batch(
        self,
    ) -> tuple[_PendingBatchRequest, ...] | tuple[_PendingRead, ...] | None:
        with self._condition:
            while not self._queue and not self._closing:
                self._condition.wait()
            if self._closing and not self._queue:
                return None
            first = self._queue.popleft()
            if isinstance(first, _PendingRead):
                # Every queued read job joins, so concurrent decisions share batched replays.
                reads = [first]
                for pending in list(self._queue):
                    if len(reads) == MAXIMUM_MERGED_READ_JOBS:
                        break
                    if isinstance(pending, _PendingRead):
                        self._queue.remove(pending)
                        reads.append(pending)
                return tuple(reads)
            selected = [first]
            collection_deadline = time.monotonic() + BATCH_COLLECTION_SECONDS
            while len(selected) < MAXIMUM_BATCH_WIDTH:
                compatible = next(
                    (
                        pending
                        for pending in self._queue
                        if not isinstance(pending, _PendingRead)
                        and pending.request.max_tokens == first.request.max_tokens
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
            # Decode shards one user per core as a grid rectangle; a width with no
            # rectangle (13 on an 11x10 grid) returns its newest requests to the queue.
            while len(selected) > 1 and not self.adapter.decode_width_supported(len(selected)):
                self._queue.appendleft(selected.pop())
            # An all-sampled batch decodes on the sampled-fusion kernel, which fits fewer
            # users than the grid rectangle (4 on P150x2); the rest wait for the next batch.
            if self.sampled_admission_enabled and all(
                classify_temperature(pending.request.temperature, self.deterministic_route)
                == ROUTE_SAMPLED
                for pending in selected
            ):
                while len(selected) > 1 and not self.adapter.sampled_width_supported(len(selected)):
                    self._queue.appendleft(selected.pop())
            return tuple(selected)

    def _run_reads(self, jobs: tuple[_PendingRead, ...]) -> None:
        # An expired job fails alone; the decisions merged with it still read.
        now = time.monotonic()
        live = []
        for job in jobs:
            if request_deadline_expired(now, job.deadline):
                job.error = BatchDecodeAdapterError(
                    "request timeout exceeded before structured reads"
                )
            else:
                live.append(job)
        if not live:
            return
        flat = tuple(request for job in live for request in job.requests)
        results = self.adapter.read_rows_batch(flat)
        if len(results) != len(flat):
            raise BatchDecodeAdapterError("structured reads returned the wrong count")
        if any(result.masked_traces_released for result in results):
            self.emitter("masked-prefill-traces-released", reason="program-compiled-after-capture")
        start = 0
        for job in live:
            job.results = results[start : start + len(job.requests)]
            start += len(job.requests)
        self.emitter(
            "structured-read-passed",
            device_seconds=sum(result.elapsed_seconds for result in results),
            job_count=len(live),
            read_count=len(results),
            routes=sorted({result.route for result in results}),
        )

    def _fail_pending(self, pending_items: tuple[Any, ...], error: Exception) -> None:
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
            batch_width=len(pending_items),
            error_type=type(terminal_error).__name__,
        )
        for pending in pending_items:
            pending.error = terminal_error

    def _worker_loop(self) -> None:
        while True:
            pending_batch = self._take_batch()
            if pending_batch is None:
                return
            if isinstance(pending_batch[0], _PendingRead):
                try:
                    self._run_reads(pending_batch)
                except (
                    AssertionError,
                    BatchDecodeAdapterError,
                    DecodeAdapterError,
                    OSError,
                    RuntimeError,
                    ValueError,
                ) as error:
                    self._fail_pending(pending_batch, error)
                finally:
                    for job in pending_batch:
                        job.completed.set()
                continue
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
                self._fail_pending(pending_batch, error)
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

        trace_reuse_activation = None
        if self.trace_reuse_profile is not None:
            trace_reuse_activation = self.trace_reuse_profile.activation(
                explicit_measurement=self.trace_reuse_admission_enabled,
                width=width,
            )
        trace_reuse_enabled = (
            trace_reuse_activation.enabled
            if trace_reuse_activation is not None
            else False
        )
        route_decision = resolve_batch_route(
            tuple(pending.request.temperature for pending in pending_batch),
            sampled_admitted=self.sampled_admission_enabled,
            current_mode=self.adapter.fusion_mode,
            deterministic_route=self.deterministic_route,
        )
        if route_decision.switch_required:
            self.adapter.switch_fusion_mode(route_decision.route)
        prefill_started = time.monotonic()
        prefill_result = self.adapter.prefill_users(
            tuple(pending.token_ids for pending in pending_batch),
            generation_tokens=maximum_generation_tokens,
            trace_reuse_enabled=trace_reuse_enabled,
        )
        if prefill_result.masked_traces_released:
            self.emitter("masked-prefill-traces-released", reason="program-compiled-after-capture")
        first_token_ready = prefill_started + prefill_result.first_token_ready_seconds
        generated = [[token] for token in prefill_result.first_token_ids]
        positions = [pending.prompt_token_count for pending in pending_batch]
        trace_prepare_started = time.monotonic()
        trace_preparation = self.adapter.prepare_width(
            token_ids=prefill_result.first_token_ids,
            positions=tuple(positions),
        )
        trace_prepare_seconds = time.monotonic() - trace_prepare_started

        samplers = [
            random.Random(
                pending.request.seed
                if pending.request.seed is not None
                else random.SystemRandom().randrange(2**63)
            )
            for pending in pending_batch
        ]

        decode_durations = []
        stopped = [token in self.stop_token_ids for token in prefill_result.first_token_ids]
        remaining_steps = maximum_generation_tokens - len(generated[0])
        if route_decision.route != ROUTE_SAMPLED and remaining_steps > 0 and not all(stopped):
            # Greedy batches stream: the device feeds each step's tokens into the next step,
            # and the host reads step k while step k + 1 runs.
            stream = self.adapter.decode_greedy_stream(
                token_ids=tuple(tokens[-1] for tokens in generated),
                positions=tuple(positions),
                steps=remaining_steps,
            )
            try:
                step_started = time.monotonic()
                for next_tokens in stream:
                    decode_durations.append(time.monotonic() - step_started)
                    for row_index, next_token in enumerate(next_tokens):
                        generated[row_index].append(next_token)
                        positions[row_index] += 1
                        stopped[row_index] = stopped[row_index] or next_token in self.stop_token_ids
                    if all(stopped):
                        break
                    if any(
                        request_deadline_expired(time.monotonic(), pending.deadline)
                        for pending in pending_batch
                    ):
                        raise BatchDecodeAdapterError("request timeout exceeded")
                    step_started = time.monotonic()
            finally:
                stream.close()
        while len(generated[0]) < maximum_generation_tokens and not all(stopped):
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
            if route_decision.route == ROUTE_SAMPLED:
                candidate_rows = self.adapter.decode_traced_sampled(
                    token_ids=tuple(row.token_id for row in plan.rows),
                    positions=tuple(row.position for row in plan.rows),
                )
                next_tokens = tuple(
                    sample_candidates(
                        values=merged_values,
                        token_ids=merged_tokens,
                        temperature=pending_batch[user_index].request.temperature,
                        top_k=DEFAULT_TOP_K,
                        top_p=DEFAULT_TOP_P,
                        draw=samplers[user_index].random(),
                    )
                    for user_index, (merged_values, merged_tokens) in enumerate(
                        merge_shard_candidates(shards, top_k=DEFAULT_TOP_K)
                        for shards in candidate_rows
                    )
                )
            else:
                next_tokens = self.adapter.decode_traced(
                    token_ids=tuple(row.token_id for row in plan.rows),
                    positions=tuple(row.position for row in plan.rows),
                )
            decode_durations.append(time.monotonic() - decode_started)
            for row, next_token in zip(plan.rows, next_tokens, strict=True):
                generated[row.row_index].append(next_token)
                positions[row.row_index] = row.next_position
                stopped[row.row_index] = stopped[row.row_index] or next_token in self.stop_token_ids

        cleanup_started = time.monotonic()
        self.adapter.reset_all()
        cleanup_seconds = time.monotonic() - cleanup_started
        request_finished = time.monotonic()
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

        measurement_mode = PRODUCTION_MEASUREMENT_MODE
        admission_eligible = False
        evidence_profile_id = self.profile.profile_id
        if self.batch_admission_profile is not None:
            try:
                measurement_mode, admission_eligible = (
                    self.batch_admission_profile.measurement(
                        enabled=self.batch_admission_enabled,
                        width=width,
                    )
                )
            except BatchAdmissionPolicyError as error:
                raise BatchDecodeAdapterError(
                    "batch admission measurement is outside policy"
                ) from error
            if self.batch_admission_enabled:
                evidence_profile_id = self.batch_admission_profile.profile_id
        if (
            trace_reuse_activation is not None
            and not self.batch_admission_enabled
            and trace_reuse_activation.enabled
        ):
            measurement_mode = trace_reuse_activation.measurement_mode
            admission_eligible = trace_reuse_activation.admission_eligible
            evidence_profile_id = self.trace_reuse_profile.profile_id

        responses = []
        for state_slice, (pending, user_tokens) in enumerate(
            zip(pending_batch, generated, strict=True)
        ):
            user_tokens, finish_reason = finish_generation(user_tokens, self.stop_token_ids)
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
                finish_reason=finish_reason,
            )
            queue_seconds = prefill_started - pending.started
            ttft_seconds = first_token_ready - prefill_started
            response["x_onix_decode_evidence"] = {
                "admission_eligible": admission_eligible,
                "aggregate_decode_tokens_per_second": aggregate_decode_rate,
                "batch_width": width,
                "clean_reset": True,
                "cleanup_seconds": cleanup_seconds,
                "decode_page_table_blocks": self.adapter.blocks_per_user,
                "decode_path": "batched-traced",
                "decode_tokens_per_second": per_user_decode_rate,
                "fallback_reason": None,
                "fusion_mode_switched": route_decision.switch_required,
                "sampled_decode": route_decision.route == ROUTE_SAMPLED,
                "serving_route": route_decision.route,
                "first_token_readback_seconds": None,
                "generated_token_ids": user_tokens,
                "measurement_mode": measurement_mode,
                "model_identity": self.model_identity,
                "package_identity": self.package_identity,
                "physical_pcie_device_ids": self.physical_pcie_device_ids,
                "prefill_device_seconds": None,
                "prefill_first_token_seconds": prefill_result.first_token_ready_seconds,
                "prefill_page_table_blocks": prefill_result.page_table_blocks,
                "prefill_state_install_seconds": prefill_result.state_install_seconds,
                "prefill_profile_receiver_drain_seconds": 0.0,
                "prefill_program_profile": None,
                "profile_id": evidence_profile_id,
                "queue_seconds": queue_seconds,
                "request_reset_seconds": 0.0,
                "request_wall_seconds": request_finished - pending.started,
                "resident_trace_count": trace_preparation.resident_trace_count,
                "runtime_identity": self.runtime_identity,
                "simulated": False,
                "state_slice": state_slice,
                "tail_token_latency_seconds": tail_token_latency_seconds,
                "token_latency_seconds": steady_durations,
                "trace_binding_stable": trace_preparation.binding_stable,
                "trace_capture_count": trace_preparation.capture_count,
                "trace_capture_seconds": trace_preparation.capture_seconds,
                "trace_prepare_seconds": trace_prepare_seconds,
                "trace_reuse_enabled": trace_reuse_enabled,
                "trace_reused": trace_preparation.reused,
                "trace_transition_reason": trace_preparation.transition_reason,
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
