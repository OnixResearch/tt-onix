"""Continuous-slot coordinator: one device worker refilling B4 decode slots.

r[impl tenstorrent.qwen38_p150x2.continuous_slot_effects]

A construction-time alternative to BatchGenerationCoordinator, selected by the
continuous-slot policy. The pure core (continuous_slot_core) decides every admission,
refill, retirement and window; this shell only executes those effects, all on its one
worker thread, and reports what the device did back to the core.

Failure handling, in order of severity:
- A request refused by the core (queue full, deadline, cancellation) fails alone.
- A refill that compiled a program or dropped the masked prefill traces releases the
  decode trace instead of replaying it; the group admits nothing more and drains, and
  its live users continue on a recaptured trace (the group coordinator's behaviour).
- Any other device error fails the whole group, which is quarantined until every row
  is reset; the next group starts clean.
- Trace-binding drift, a failed reset, or one device effect outliving the policy's
  stall bound quarantines the engine: every request fails and the device is not used
  again by this process.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from . import continuous_slot_core as slots
from .batch_decode_adapter import (
    BatchDecodeAdapterError,
    SlotRefillResult,
    StructuredReadRows,
    TraceBindingDriftError,
)
from .batch_service import (
    MAXIMUM_MERGED_READ_JOBS,
    BatchGenerationCoordinator,
    _PendingBatchRequest,
    _PendingRead,
)
from .core import CompletionRequest, build_completion_response, finish_generation
from .decode_adapter import DecodeAdapterError
from .prefix_cache_core import PrefixCachePolicy, cached_blocks, prefix_digests
from .sampled_decode_core import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    merge_shard_candidates,
    sample_candidates,
)
from .sampled_mode_router_core import ROUTE_SAMPLED, classify_temperature

CONTINUOUS_DECODE_PATH = "continuous-slot"
CONTINUOUS_MEASUREMENT_MODE = "continuous-slot-candidate"
# How often a waiting caller checks the worker's current device effect for a stall.
MAXIMUM_STALL_POLL_SECONDS = 1.0
STALL_POLLS_PER_BOUND = 4
INACTIVE_TOKEN_ID = 0
INACTIVE_POSITION = 0
DEVICE_ERRORS = (
    AssertionError,
    BatchDecodeAdapterError,
    DecodeAdapterError,
    OSError,
    RuntimeError,
    ValueError,
)


class EngineQuarantinedError(BatchDecodeAdapterError):
    """The engine was quarantined; this process runs no further device work."""


class SlotRequestRefused(BatchDecodeAdapterError):
    """The scheduler refused or ended a request without a response."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"request refused by the slot scheduler: {reason}")
        self.reason = reason


@dataclass(eq=False)
class _SlotRun:
    """Host-side record of one admitted request; the core owns its scheduling facts."""

    pending: _PendingBatchRequest
    route: str
    sampler: random.Random
    slot: int
    generation: int
    admitted: float
    # Slot-trace captures before this request's refill, and KV blocks held by every
    # occupied slot once it was admitted.
    captures_at_admission: int = 0
    kv_blocks_in_use: int = 0
    prefix_hit_tokens: int = 0
    prefix_saved_tokens: int = 0
    tokens: list[int] = field(default_factory=list)
    refill: SlotRefillResult | None = None
    first_token_ready: float | None = None
    step_seconds: list[float] = field(default_factory=list)


class ContinuousSlotCoordinator(BatchGenerationCoordinator):
    """Serve generation requests from continuously refilled decode slots."""

    def __init__(
        self,
        *,
        slot_policy: slots.ContinuousSlotPolicy,
        prefix_policy: PrefixCachePolicy | None = None,
        prefix_identity: str = "",
        blake3_digest: Callable[[bytes], bytes] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("batch_admission_enabled") or kwargs.get("trace_reuse_admission_enabled"):
            raise BatchDecodeAdapterError(
                "continuous slots cannot run a fixed-group admission measurement"
            )
        adapter = kwargs["adapter"]
        if adapter.maximum_width < slot_policy.slot_capacity:
            raise BatchDecodeAdapterError("the adapter holds fewer rows than the slot policy")
        pool = adapter.kv_pool
        if prefix_policy is not None:
            if adapter.prefix_states != prefix_policy.state_slots:
                raise BatchDecodeAdapterError("the adapter's prefix states differ from the prefix policy")
            if not prefix_identity or blake3_digest is None:
                raise BatchDecodeAdapterError("the prefix cache needs a serving identity and BLAKE3")
        self.slot_policy = slot_policy
        self.prefix_policy = prefix_policy
        self._prefix_identity = prefix_identity
        self._blake3_digest = blake3_digest
        self._kv_range = slots.KvBlockRange(
            first_block=pool.reserved_prefix_blocks,
            end_block=pool.scratch_block,
            row_blocks=pool.row_blocks,
            block_tokens=pool.kv_block_size,
        )
        self._schedule = slots.initial_schedule(slot_policy)
        self._waiting: dict[str, _PendingBatchRequest] = {}
        self._runs: dict[str, _SlotRun] = {}
        self._quarantine_reason: str | None = None
        self._effect_started: float | None = None
        # The worker starts in the base constructor, so every field above exists first.
        super().__init__(**kwargs)

    @property
    def quarantine_reason(self) -> str | None:
        with self._condition:
            return self._quarantine_reason

    def _require_serving(self) -> None:
        if self._quarantine_reason is not None:
            raise EngineQuarantinedError(f"engine quarantined: {self._quarantine_reason}")
        if self._closing:
            raise BatchDecodeAdapterError("batch worker is closed")
        if not self._accepting_requests:
            raise BatchDecodeAdapterError("batch worker is not ready")

    def _route_for(self, request: CompletionRequest) -> str:
        route = classify_temperature(request.temperature, self.deterministic_route)
        if route == ROUTE_SAMPLED and not self.sampled_admission_enabled:
            return self.deterministic_route
        return route

    def generate(
        self,
        *,
        request: CompletionRequest,
        request_id: str,
        token_ids: torch.Tensor,
        prompt_token_count: int,
        started: float,
        deadline: float,
        token_sink: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        blocks = self.adapter.request_blocks(
            prompt_tokens=prompt_token_count, generation_tokens=request.max_tokens
        )
        digests: tuple[str, ...] = ()
        if self.prefix_policy is not None:
            digests = prefix_digests(
                [int(token) for token in token_ids[0].tolist()],
                identity=self._prefix_identity,
                tenant=self.prefix_policy.tenant,
                blake3_digest=self._blake3_digest,
            )
        pending = _PendingBatchRequest(
            request=request,
            request_id=request_id,
            token_ids=token_ids,
            prompt_token_count=prompt_token_count,
            started=started,
            deadline=deadline,
            token_sink=token_sink,
        )
        slot_request = slots.SlotRequest(
            request_id=request_id,
            prompt_tokens=prompt_token_count,
            max_tokens=request.max_tokens,
            route=self._route_for(request),
            blocks=blocks,
            arrival_seconds=started,
            deadline_seconds=deadline,
            prefix_digests=digests,
        )
        with self._condition:
            self._require_serving()
            schedule, refused = slots.submit(
                self._schedule, self.slot_policy, self._kv_range, slot_request
            )
            if refused is not None:
                raise SlotRequestRefused(refused.reason)
            self._schedule = schedule
            self._waiting[request_id] = pending
            self._condition.notify_all()
        self._await(pending)
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
        """Run structured reads once the slot group drains; refills wait for them."""
        if not requests:
            raise BatchDecodeAdapterError("a structured read job needs at least one read")
        pending = _PendingRead(requests=requests, deadline=deadline)
        with self._condition:
            self._require_serving()
            self._queue.append(pending)
            self._condition.notify_all()
        self._await(pending)
        if pending.error is not None:
            raise pending.error
        if pending.results is None:
            raise BatchDecodeAdapterError("batch worker returned no reads")
        return pending.results

    def cancel(self, request_id: str) -> bool:
        """Withdraw a request whose client left; True when the scheduler knew it."""
        with self._condition:
            schedule, refused = slots.cancel(self._schedule, request_id)
            known = refused is not None or schedule != self._schedule
            self._schedule = schedule
            if refused is not None:
                pending = self._waiting.pop(request_id, None)
                if pending is not None:
                    self._finish(pending, error=SlotRequestRefused(refused.reason))
            self._condition.notify_all()
        return known

    def _await(self, pending: _PendingBatchRequest | _PendingRead) -> None:
        """Wait for the worker; quarantine the engine if one device effect stalls."""
        stall_bound = self.slot_policy.worker_stall_seconds
        poll = min(MAXIMUM_STALL_POLL_SECONDS, stall_bound / STALL_POLLS_PER_BOUND)
        while not pending.completed.wait(timeout=poll):
            with self._condition:
                started = self._effect_started
                if started is not None and time.monotonic() - started > stall_bound:
                    self._quarantine_locked("worker-stalled")

    @staticmethod
    def _finish(
        pending: _PendingBatchRequest | _PendingRead,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if pending.completed.is_set():
            return
        pending.response = response
        pending.error = error
        pending.completed.set()

    def _quarantine_locked(self, reason: str) -> None:
        """Fail every request and stop device use. Caller holds the condition."""
        if self._quarantine_reason is not None:
            return
        self._quarantine_reason = reason
        error = EngineQuarantinedError(f"engine quarantined: {reason}")
        schedule, _failed = slots.quarantine_group(self._schedule)
        self._schedule, _refused = slots.drain_queue(schedule, "quarantined")
        for pending in list(self._waiting.values()):
            self._finish(pending, error=error)
        for run in self._runs.values():
            self._finish(run.pending, error=error)
        for job in self._queue:
            self._finish(job, error=error)
        self._waiting.clear()
        self._runs.clear()
        self._queue.clear()
        self._condition.notify_all()
        self.emitter("engine-quarantined", reason=reason)

    @contextmanager
    def _device_effect(self) -> Iterator[None]:
        """Bracket one device call so a stalled one can be detected and fenced off."""
        with self._condition:
            if self._quarantine_reason is not None:
                raise EngineQuarantinedError(self._quarantine_reason)
            self._effect_started = time.monotonic()
        try:
            yield
        finally:
            with self._condition:
                self._effect_started = None
                if self._quarantine_reason is not None:
                    raise EngineQuarantinedError(self._quarantine_reason)

    def _occupied(self) -> bool:
        return any(slot.state in slots.OCCUPYING_STATES for slot in self._schedule.slots)

    def _has_generation_work(self) -> bool:
        return bool(self._schedule.queue) or self._occupied()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._closing
                    and self._quarantine_reason is None
                    and not self._queue
                    and not self._has_generation_work()
                ):
                    self._condition.wait()
                if self._quarantine_reason is not None:
                    return
                if self._closing:
                    self._close_locked()
                    return
                reads_waiting = bool(self._queue)
                self._schedule, effects = slots.plan_boundary(
                    self._schedule,
                    self.slot_policy,
                    self._kv_range,
                    now=time.monotonic(),
                    reads_waiting=reads_waiting,
                    prefix_policy=self.prefix_policy,
                    prefix_available=self.prefix_policy is not None
                    and self.adapter.prefix_cache_available(),
                )
                reads: tuple[_PendingRead, ...] = ()
                if reads_waiting and not self._occupied():
                    reads = tuple(
                        self._queue.popleft()
                        for _ in range(min(len(self._queue), MAXIMUM_MERGED_READ_JOBS))
                    )
                    # A read writes KV through the read row, over blocks cached prefixes hold.
                    self._schedule = slots.invalidate_prefix_cache(self._schedule)
            try:
                self._apply_boundary(effects)
                if reads:
                    self._serve_reads(reads)
                self._run_window()
            except EngineQuarantinedError:
                return
            except TraceBindingDriftError:
                with self._condition:
                    self._quarantine_locked("trace-binding-drift")
                return
            except DEVICE_ERRORS as error:
                if not self._fail_group(error):
                    return

    def _close_locked(self) -> None:
        error = BatchDecodeAdapterError("batch worker is closed")
        schedule, _failed = slots.quarantine_group(self._schedule)
        self._schedule, _refused = slots.drain_queue(schedule, "closed")
        for pending in list(self._waiting.values()):
            self._finish(pending, error=error)
        for run in self._runs.values():
            self._finish(run.pending, error=error)
        for job in self._queue:
            self._finish(job, error=error)
        self._waiting.clear()
        self._runs.clear()
        self._queue.clear()

    def _serve_reads(self, reads: tuple[_PendingRead, ...]) -> None:
        try:
            with self._device_effect():
                self._run_reads(reads)
        except EngineQuarantinedError:
            raise
        except TraceBindingDriftError:
            raise
        except DEVICE_ERRORS as error:
            for job in reads:
                self._finish(job, error=error)
            self.emitter("structured-read-failed", error_type=type(error).__name__)
        finally:
            for job in reads:
                job.completed.set()

    def _apply_boundary(self, effects: tuple[slots.BoundaryEffect, ...]) -> None:
        for effect in effects:
            if isinstance(effect, slots.RejectRequest):
                with self._condition:
                    pending = self._waiting.pop(effect.request_id, None)
                if pending is not None:
                    self._finish(pending, error=SlotRequestRefused(effect.reason))
            elif isinstance(effect, slots.RetireSlot):
                with self._device_effect():
                    self.adapter.retire_slot(slot=effect.slot)
                self._retire(effect)
            elif isinstance(effect, slots.StartGroup):
                with self._device_effect():
                    self.adapter.begin_slots(width=self.slot_policy.slot_capacity, route=effect.route)
                self.emitter("continuous-slot-group-started", route=effect.route)
            elif isinstance(effect, slots.PrefillSlot):
                self._refill(effect)

    def _refill(self, effect: slots.PrefillSlot) -> None:
        with self._condition:
            pending = self._waiting.pop(effect.request_id)
            request = pending.request
            run = _SlotRun(
                pending=pending,
                route=self._schedule.route,
                sampler=random.Random(
                    request.seed
                    if request.seed is not None
                    else random.SystemRandom().randrange(2**63)
                ),
                slot=effect.slot,
                generation=effect.generation,
                admitted=time.monotonic(),
                captures_at_admission=self.adapter.slot_trace_captures,
                kv_blocks_in_use=len(
                    cached_blocks(self._schedule.prefix_entries).union(
                        block
                        for slot in self._schedule.slots
                        if slot.state in slots.OCCUPYING_STATES and slot.request is not None
                        for block in range(slot.block_start, slot.block_start + slot.block_count)
                    )
                ),
            )
            self._runs[effect.request_id] = run
        with self._device_effect():
            result = self.adapter.refill_slot(
                slot=effect.slot,
                token_ids=pending.token_ids,
                block_start=effect.block_start,
                block_count=effect.block_count,
                prefix_ranges=effect.prefix_ranges,
                prefix_tokens=effect.prefix_tokens,
                resume_state=effect.resume_state,
                save_state=effect.save_state,
                save_at=effect.save_at,
            )
        run.refill = result
        run.first_token_ready = run.admitted + result.first_token_ready_seconds
        with self._condition:
            self._schedule = slots.observe_prefill(
                self._schedule,
                slot=effect.slot,
                generation=effect.generation,
                token_id=result.first_token_id,
                stopped=result.first_token_id in self.stop_token_ids,
            )
            if result.decode_trace_released:
                self._schedule = slots.mark_draining(self._schedule)
            if result.masked_traces_released:
                # Prefix state can no longer be restored or saved in this process.
                self._schedule = slots.invalidate_prefix_cache(self._schedule)
        run.prefix_hit_tokens = effect.prefix_tokens
        run.prefix_saved_tokens = effect.save_at
        run.tokens.append(result.first_token_id)
        self._publish(pending, result.first_token_id)
        if result.masked_traces_released:
            self.emitter("masked-prefill-traces-released", reason="program-compiled-after-capture")
        if result.decode_trace_released:
            self.emitter("continuous-slot-group-draining", reason="unsafe-refill-transition")

    def _run_window(self) -> None:
        with self._condition:
            window = slots.plan_window(self._schedule, self.slot_policy)
            route = self._schedule.route
            runs = {row.slot: self._runs[row.request_id] for row in window.rows} if window else {}
        if window is None:
            return
        width = self.slot_policy.slot_capacity
        token_ids = [INACTIVE_TOKEN_ID] * width
        positions = [INACTIVE_POSITION] * width
        for row in window.rows:
            token_ids[row.slot] = row.token_id
            positions[row.slot] = row.position
        if route == ROUTE_SAMPLED:
            for _step in range(window.steps):
                step_started = time.monotonic()
                with self._device_effect():
                    candidate_rows = self.adapter.decode_slots_sampled(
                        token_ids=tuple(token_ids), positions=tuple(positions)
                    )
                next_tokens = [INACTIVE_TOKEN_ID] * width
                for slot, run in runs.items():
                    values, candidates = merge_shard_candidates(
                        candidate_rows[slot], top_k=DEFAULT_TOP_K
                    )
                    next_tokens[slot] = sample_candidates(
                        values=values,
                        token_ids=candidates,
                        temperature=run.pending.request.temperature,
                        top_k=DEFAULT_TOP_K,
                        top_p=DEFAULT_TOP_P,
                        draw=run.sampler.random(),
                    )
                if not self._observe_step(window, runs, next_tokens, time.monotonic() - step_started):
                    return
                for row in window.rows:
                    token_ids[row.slot] = next_tokens[row.slot]
                    positions[row.slot] += 1
            return
        step_started = time.monotonic()
        with self._device_effect():
            stream = self.adapter.decode_slots_greedy(
                token_ids=tuple(token_ids), positions=tuple(positions), steps=window.steps
            )
        try:
            while True:
                with self._device_effect():
                    next_tokens = next(stream, None)
                if next_tokens is None:
                    return
                if not self._observe_step(window, runs, next_tokens, time.monotonic() - step_started):
                    return
                step_started = time.monotonic()
        finally:
            stream.close()

    def _observe_step(
        self,
        window: slots.DecodeWindow,
        runs: dict[int, _SlotRun],
        next_tokens: tuple[int, ...] | list[int],
        step_seconds: float,
    ) -> bool:
        """Apply one step's tokens to the rows still running; False when none remain."""
        running = False
        published = []
        with self._condition:
            for row in window.rows:
                if not slots.slot_running(self._schedule, row.slot, row.generation):
                    continue
                token = int(next_tokens[row.slot])
                self._schedule = slots.observe_step(
                    self._schedule,
                    slot=row.slot,
                    generation=row.generation,
                    token_id=token,
                    stopped=token in self.stop_token_ids,
                )
                run = runs[row.slot]
                run.tokens.append(token)
                run.step_seconds.append(step_seconds)
                published.append((run.pending, token))
                running = running or slots.slot_running(self._schedule, row.slot, row.generation)
        for pending, token in published:
            self._publish(pending, token)
        return running

    def _retire(self, effect: slots.RetireSlot) -> None:
        with self._condition:
            run = self._runs.pop(effect.request_id, None)
        if run is None:
            return
        if effect.finish_reason in (slots.FINISH_STOP, slots.FINISH_LENGTH):
            self._finish(run.pending, response=self._response(run))
        elif effect.finish_reason == slots.FINISH_TIMEOUT:
            self._finish(run.pending, error=BatchDecodeAdapterError("request timeout exceeded"))
        else:
            self._finish(run.pending, error=SlotRequestRefused(effect.finish_reason))

    def _fail_group(self, error: Exception) -> bool:
        """Fail and reset the group after a device error; False when the engine is quarantined."""
        with self._condition:
            self._schedule, failed = slots.quarantine_group(self._schedule)
            runs = [self._runs.pop(item.request_id) for item in failed if item.request_id in self._runs]
        for run in runs:
            self._finish(run.pending, error=error)
        self.emitter(
            "continuous-slot-group-failed", error_type=type(error).__name__, failed=len(runs)
        )
        try:
            with self._device_effect():
                self.adapter.abandon_slots()
        except EngineQuarantinedError:
            return False
        except DEVICE_ERRORS:
            with self._condition:
                self._quarantine_locked("slot-reset-failed")
            return False
        with self._condition:
            self._schedule = slots.release_quarantine(self._schedule)
        return True

    def _response(self, run: _SlotRun) -> dict[str, Any]:
        pending = run.pending
        user_tokens, finish_reason = finish_generation(run.tokens, self.stop_token_ids)
        response = build_completion_response(
            request=pending.request,
            request_id=pending.request_id,
            created=int(time.time()),
            generated_text=self.tokenizer.decode(user_tokens, skip_special_tokens=True),
            prompt_token_count=pending.prompt_token_count,
            generated_token_count=len(user_tokens),
            finish_reason=finish_reason,
        )
        finished = time.monotonic()
        steps = run.step_seconds
        decode_rate = len(steps) / sum(steps) if steps and sum(steps) > 0 else 0.0
        refill = run.refill
        response["x_onix_decode_evidence"] = {
            "admission_eligible": False,
            "decode_path": CONTINUOUS_DECODE_PATH,
            "decode_tokens_per_second": decode_rate,
            "fallback_reason": None,
            "generated_token_ids": user_tokens,
            "measurement_mode": CONTINUOUS_MEASUREMENT_MODE,
            "model_identity": self.model_identity,
            "package_identity": self.package_identity,
            "physical_pcie_device_ids": self.physical_pcie_device_ids,
            "prefill_decode_trace_released": refill.decode_trace_released,
            "prefill_masked_traces": refill.masked_traces_used,
            "prefill_seconds": refill.elapsed_seconds,
            "profile_id": self.slot_policy.profile_id,
            "queue_seconds": run.admitted - pending.started,
            "request_wall_seconds": finished - pending.started,
            "runtime_identity": self.runtime_identity,
            "sampled_decode": run.route == ROUTE_SAMPLED,
            "serving_route": run.route,
            "simulated": False,
            "slot": run.slot,
            "slot_generation": run.generation,
            "kv_blocks_in_use": run.kv_blocks_in_use,
            "prefix_cache_profile_id": self.prefix_policy.profile_id if self.prefix_policy else None,
            "prefix_hit_tokens": run.prefix_hit_tokens,
            "prefix_saved_tokens": run.prefix_saved_tokens,
            "tail_token_latency_seconds": max(steps) if steps else 0.0,
            "trace_captures": self.adapter.slot_trace_captures - run.captures_at_admission,
            "ttft_seconds": refill.first_token_ready_seconds,
        }
        self.emitter(
            "request-passed",
            decode_path=CONTINUOUS_DECODE_PATH,
            decode_tokens_per_second=decode_rate,
            elapsed_seconds=finished - pending.started,
            generated_token_count=len(user_tokens),
            prompt_token_count=pending.prompt_token_count,
            queue_seconds=run.admitted - pending.started,
            request_id=pending.request_id,
            slot=run.slot,
            ttft_seconds=refill.first_token_ready_seconds,
        )
        return response
