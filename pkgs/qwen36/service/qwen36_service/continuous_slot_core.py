"""Pure continuous-slot scheduler for the Qwen3.8 P150x2 batch worker.

r[impl tenstorrent.qwen38_p150x2.continuous_slot_policy]

The group coordinator runs one group of equal-bound requests to the end before the next
starts. This core instead keeps a fixed set of decode slots (the rows of one resident
B4 trace) and refills a slot as soon as its request finishes, so a short request does
not wait behind a long one and a finished row stops occupying the batch.

The core owns every scheduling fact: slot state and generation, request identity, next
position, deadline, KV block run, queue order and refill decisions. It reads explicit
inputs only (time, request bounds, route, the KV block range, and completion
observations) and returns a new schedule plus typed effects. The shell executes the
effects on the one device worker and reports what happened; it holds no scheduling
state of its own.

Slot lifecycle, each admission under a fresh generation number:

    free -> prefilling -> active -> complete -> free
    prefilling -> complete, when the first token already ends the request
    any occupied state -> quarantined -> free (after the shell reset the device)

An event names the slot and generation it was issued for. An event for an older
generation, or for a slot in another state, is rejected and leaves the schedule as it
was, so a late completion can never touch the request that now owns the slot.

Fairness: the queue is first in, first out. A request later in the queue may take a
slot ahead of the head only while the head does not fit (another route runs, or no KV
run is free), and only `maximum_head_bypasses` times; after that nothing is admitted
until the head fits, which it does once the group drains. Budget limits delay the head
to the next boundary but never let another request pass it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .decode_core import request_deadline_expired

PROFILE_SCHEMA_VERSION = 1
SLOT_CAPACITY = 4
MAXIMUM_QUEUE_CAPACITY = 256
MAXIMUM_DECODE_WINDOW_STEPS = 256
MAXIMUM_REFILL_PROMPT_TOKEN_BUDGET = 65536
MAXIMUM_HEAD_BYPASSES = 16
MAXIMUM_POLICY_SECONDS = 86400.0

SLOT_FREE = "free"
SLOT_PREFILLING = "prefilling"
SLOT_ACTIVE = "active"
SLOT_COMPLETE = "complete"
SLOT_QUARANTINED = "quarantined"
# States that own a KV run and a row of device state.
OCCUPYING_STATES = frozenset({SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE, SLOT_QUARANTINED})
# States a request is still running in.
RUNNING_STATES = frozenset({SLOT_PREFILLING, SLOT_ACTIVE})

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_CANCELLED = "cancelled"
FINISH_TIMEOUT = "timeout"
FINISH_FAILED = "failed"

REJECT_QUEUE_FULL = "queue-full"
REJECT_EXCEEDS_KV_ROW = "exceeds-kv-row"
REJECT_DUPLICATE = "duplicate-request"
REJECT_QUEUE_TIMEOUT = "queue-timeout"
REJECT_DEADLINE = "deadline-expired"
REJECT_CANCELLED = "cancelled"

POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "slot_capacity",
        "queue_capacity",
        "decode_window_steps",
        "refills_per_boundary",
        "refill_prompt_token_budget",
        "maximum_head_bypasses",
        "maximum_queue_seconds",
        "worker_stall_seconds",
    }
)


class ContinuousSlotPolicyError(ValueError):
    """The policy, a request, or the KV range is outside the closed scheduler contract."""


class SlotEventRejected(ValueError):
    """An observation does not match the slot's current generation and state."""


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContinuousSlotPolicyError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _require_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuousSlotPolicyError(f"{name} must be a number")
    seconds = float(value)
    if not 0.0 < seconds <= MAXIMUM_POLICY_SECONDS:
        raise ContinuousSlotPolicyError(f"{name} must be in (0, {MAXIMUM_POLICY_SECONDS}]")
    return seconds


@dataclass(frozen=True)
class ContinuousSlotPolicy:
    """Closed scheduler bounds projected from continuous-slot-policy.ncl."""

    profile_id: str
    slot_capacity: int
    queue_capacity: int
    decode_window_steps: int
    refills_per_boundary: int
    refill_prompt_token_budget: int
    maximum_head_bypasses: int
    maximum_queue_seconds: float
    worker_stall_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ContinuousSlotPolicyError("profile_id must be a non-empty string")
        if self.slot_capacity != SLOT_CAPACITY:
            raise ContinuousSlotPolicyError(f"slot_capacity must be {SLOT_CAPACITY}")
        _require_int(self.queue_capacity, "queue_capacity", 1, MAXIMUM_QUEUE_CAPACITY)
        _require_int(
            self.decode_window_steps, "decode_window_steps", 1, MAXIMUM_DECODE_WINDOW_STEPS
        )
        _require_int(self.refills_per_boundary, "refills_per_boundary", 1, self.slot_capacity)
        _require_int(
            self.refill_prompt_token_budget,
            "refill_prompt_token_budget",
            1,
            MAXIMUM_REFILL_PROMPT_TOKEN_BUDGET,
        )
        _require_int(self.maximum_head_bypasses, "maximum_head_bypasses", 0, MAXIMUM_HEAD_BYPASSES)
        _require_seconds(self.maximum_queue_seconds, "maximum_queue_seconds")
        _require_seconds(self.worker_stall_seconds, "worker_stall_seconds")

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> ContinuousSlotPolicy:
        if not isinstance(root, Mapping):
            raise ContinuousSlotPolicyError("policy must be a mapping")
        if set(root) != POLICY_FIELDS:
            raise ContinuousSlotPolicyError("policy fields differ from the closed schema")
        if _require_int(root["schema_version"], "schema_version", 1, 1) != PROFILE_SCHEMA_VERSION:
            raise ContinuousSlotPolicyError("schema_version is unsupported")
        return cls(
            profile_id=root["profile_id"],
            slot_capacity=_require_int(root["slot_capacity"], "slot_capacity", 1, SLOT_CAPACITY),
            queue_capacity=root["queue_capacity"],
            decode_window_steps=root["decode_window_steps"],
            refills_per_boundary=root["refills_per_boundary"],
            refill_prompt_token_budget=root["refill_prompt_token_budget"],
            maximum_head_bypasses=root["maximum_head_bypasses"],
            maximum_queue_seconds=_require_seconds(
                root["maximum_queue_seconds"], "maximum_queue_seconds"
            ),
            worker_stall_seconds=_require_seconds(
                root["worker_stall_seconds"], "worker_stall_seconds"
            ),
        )


@dataclass(frozen=True)
class KvBlockRange:
    """The KV pool blocks generation rows may take: [first_block, end_block).

    `row_blocks` is one page-table row's length, the most one request can take.
    """

    first_block: int
    end_block: int
    row_blocks: int

    def __post_init__(self) -> None:
        _require_int(self.first_block, "first_block", 0, 2**31 - 1)
        _require_int(self.row_blocks, "row_blocks", 1, 2**31 - 1)
        _require_int(self.end_block, "end_block", self.first_block + self.row_blocks, 2**31 - 1)


@dataclass(frozen=True)
class SlotRequest:
    """One generation request's scheduling facts; route is the resolved serving route."""

    request_id: str
    prompt_tokens: int
    max_tokens: int
    route: str
    blocks: int
    arrival_seconds: float
    deadline_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ContinuousSlotPolicyError("request_id must be a non-empty string")
        if not isinstance(self.route, str) or not self.route:
            raise ContinuousSlotPolicyError("route must be a non-empty string")
        _require_int(self.prompt_tokens, "prompt_tokens", 1, 2**31 - 1)
        _require_int(self.max_tokens, "max_tokens", 1, 2**31 - 1)
        _require_int(self.blocks, "blocks", 1, 2**31 - 1)
        # Both are validated as finite by request_deadline_expired.
        request_deadline_expired(self.arrival_seconds, self.deadline_seconds)


@dataclass(frozen=True)
class QueuedRequest:
    request: SlotRequest
    # Times a later request took a slot while this one was the blocked queue head.
    bypassed: int = 0


@dataclass(frozen=True)
class Slot:
    index: int
    state: str = SLOT_FREE
    generation: int = 0
    request: SlotRequest | None = None
    block_start: int = 0
    # The token the next decode step consumes, and the position it is written at.
    token_id: int = 0
    position: int = 0
    generated: int = 0
    finish_reason: str | None = None


@dataclass(frozen=True)
class SlotSchedule:
    slots: tuple[Slot, ...]
    queue: tuple[QueuedRequest, ...]
    # The serving route every occupied slot runs; None when no slot is occupied.
    route: str | None
    # No refill until the group is empty (an unsafe refill transition was observed).
    draining: bool
    next_generation: int


@dataclass(frozen=True)
class RejectRequest:
    request_id: str
    reason: str


@dataclass(frozen=True)
class StartGroup:
    """Every slot is free: begin a new group on this route before its first prefill."""

    route: str


@dataclass(frozen=True)
class PrefillSlot:
    slot: int
    generation: int
    request_id: str
    block_start: int
    block_count: int


@dataclass(frozen=True)
class RetireSlot:
    slot: int
    generation: int
    request_id: str
    finish_reason: str


@dataclass(frozen=True)
class DecodeRow:
    slot: int
    generation: int
    request_id: str
    token_id: int
    position: int


@dataclass(frozen=True)
class DecodeWindow:
    steps: int
    rows: tuple[DecodeRow, ...]


BoundaryEffect = RejectRequest | StartGroup | PrefillSlot | RetireSlot


def initial_schedule(policy: ContinuousSlotPolicy) -> SlotSchedule:
    return SlotSchedule(
        slots=tuple(Slot(index=index) for index in range(policy.slot_capacity)),
        queue=(),
        route=None,
        draining=False,
        next_generation=1,
    )


def _known_request_ids(schedule: SlotSchedule) -> set[str]:
    ids = {queued.request.request_id for queued in schedule.queue}
    ids.update(slot.request.request_id for slot in schedule.slots if slot.request is not None)
    return ids


def submit(
    schedule: SlotSchedule,
    policy: ContinuousSlotPolicy,
    kv_range: KvBlockRange,
    request: SlotRequest,
) -> tuple[SlotSchedule, RejectRequest | None]:
    """Queue one request, or refuse it without changing the schedule."""
    if request.blocks > kv_range.row_blocks:
        return schedule, RejectRequest(request.request_id, REJECT_EXCEEDS_KV_ROW)
    if request.request_id in _known_request_ids(schedule):
        return schedule, RejectRequest(request.request_id, REJECT_DUPLICATE)
    if len(schedule.queue) >= policy.queue_capacity:
        return schedule, RejectRequest(request.request_id, REJECT_QUEUE_FULL)
    return replace(schedule, queue=schedule.queue + (QueuedRequest(request),)), None


def cancel(schedule: SlotSchedule, request_id: str) -> tuple[SlotSchedule, RejectRequest | None]:
    """Withdraw a request: a queued one is refused now, a running one completes as cancelled.

    A cancelled slot keeps its KV run until the next boundary retires it. An unknown or
    already finished request leaves the schedule unchanged.
    """
    for position, queued in enumerate(schedule.queue):
        if queued.request.request_id == request_id:
            queue = schedule.queue[:position] + schedule.queue[position + 1 :]
            return replace(schedule, queue=queue), RejectRequest(request_id, REJECT_CANCELLED)
    for slot in schedule.slots:
        if (
            slot.request is not None
            and slot.request.request_id == request_id
            and slot.state in RUNNING_STATES
        ):
            finished = replace(slot, state=SLOT_COMPLETE, finish_reason=FINISH_CANCELLED)
            return _with_slot(schedule, finished), None
    return schedule, None


def _with_slot(schedule: SlotSchedule, slot: Slot) -> SlotSchedule:
    slots = tuple(slot if existing.index == slot.index else existing for existing in schedule.slots)
    return replace(schedule, slots=slots)


def _free_run(slots: tuple[Slot, ...], kv_range: KvBlockRange, blocks: int) -> int | None:
    """First block of the lowest free contiguous run of `blocks`, or None."""
    runs = sorted(
        (slot.block_start, slot.block_start + slot.request.blocks)
        for slot in slots
        if slot.state in OCCUPYING_STATES and slot.request is not None
    )
    start = kv_range.first_block
    for run_start, run_end in runs:
        if run_start - start >= blocks:
            return start
        start = max(start, run_end)
    return start if kv_range.end_block - start >= blocks else None


def plan_boundary(
    schedule: SlotSchedule,
    policy: ContinuousSlotPolicy,
    kv_range: KvBlockRange,
    *,
    now: float,
    reads_waiting: bool = False,
) -> tuple[SlotSchedule, tuple[BoundaryEffect, ...]]:
    """Decide everything that happens between two decode windows.

    Effects are ordered for execution: retirements, refusals, then one optional group
    start and the prefills. A waiting structured read admits nothing, so the group
    drains and the read runs on an idle device.
    """
    effects: list[BoundaryEffect] = []
    slots = list(schedule.slots)

    # Retire finished and timed-out requests; their slots and KV runs become free.
    for position, slot in enumerate(slots):
        request = slot.request
        if request is None:
            continue
        reason = None
        if slot.state == SLOT_COMPLETE:
            reason = slot.finish_reason
        elif slot.state == SLOT_ACTIVE and request_deadline_expired(now, request.deadline_seconds):
            reason = FINISH_TIMEOUT
        if reason is None:
            continue
        effects.append(RetireSlot(slot.index, slot.generation, request.request_id, reason))
        slots[position] = Slot(index=slot.index, generation=slot.generation)

    # Refuse queued requests that can no longer be served in time.
    queue = []
    for queued in schedule.queue:
        request = queued.request
        if request_deadline_expired(now, request.deadline_seconds):
            effects.append(RejectRequest(request.request_id, REJECT_DEADLINE))
        elif now - request.arrival_seconds >= policy.maximum_queue_seconds:
            effects.append(RejectRequest(request.request_id, REJECT_QUEUE_TIMEOUT))
        else:
            queue.append(queued)

    occupied = any(slot.state in OCCUPYING_STATES for slot in slots)
    route = schedule.route if occupied else None
    draining = schedule.draining if occupied else False
    next_generation = schedule.next_generation

    if not reads_waiting and not draining and not any(
        slot.state == SLOT_QUARANTINED for slot in slots
    ):
        group_start = not occupied
        # Prefill interference limits apply only beside active users.
        admission_limit = policy.slot_capacity if group_start else policy.refills_per_boundary
        admitted = 0
        admitted_tokens = 0
        while admitted < admission_limit and queue:
            free = [slot for slot in slots if slot.state == SLOT_FREE]
            if not free:
                break
            chosen = None
            for queue_index, queued in enumerate(queue):
                request = queued.request
                over_budget = (
                    not group_start
                    and admitted > 0
                    and admitted_tokens + request.prompt_tokens > policy.refill_prompt_token_budget
                )
                if over_budget:
                    # Waiting for the next boundary is not a reason to pass anyone.
                    break
                if route is None or request.route == route:
                    block_start = _free_run(tuple(slots), kv_range, request.blocks)
                    if block_start is not None:
                        chosen = (queue_index, block_start)
                        break
                if queue_index == 0 and queued.bypassed >= policy.maximum_head_bypasses:
                    break
            if chosen is None:
                break
            queue_index, block_start = chosen
            request = queue.pop(queue_index).request
            if queue_index > 0:
                head = queue[0]
                queue[0] = replace(head, bypassed=head.bypassed + 1)
            if route is None:
                route = request.route
                effects.append(StartGroup(route))
            slot = free[0]
            generation = next_generation
            next_generation += 1
            slots[slot.index] = Slot(
                index=slot.index,
                state=SLOT_PREFILLING,
                generation=generation,
                request=request,
                block_start=block_start,
                position=request.prompt_tokens,
            )
            effects.append(
                PrefillSlot(slot.index, generation, request.request_id, block_start, request.blocks)
            )
            admitted += 1
            admitted_tokens += request.prompt_tokens

    return (
        SlotSchedule(
            slots=tuple(slots),
            queue=tuple(queue),
            route=route,
            draining=draining,
            next_generation=next_generation,
        ),
        tuple(effects),
    )


def _require_slot(schedule: SlotSchedule, slot_index: int, generation: int, state: str) -> Slot:
    if isinstance(slot_index, bool) or not isinstance(slot_index, int):
        raise SlotEventRejected("slot index must be an integer")
    if not 0 <= slot_index < len(schedule.slots):
        raise SlotEventRejected("slot index is outside the schedule")
    slot = schedule.slots[slot_index]
    if slot.generation != generation:
        raise SlotEventRejected(
            f"slot {slot_index} is at generation {slot.generation}, not {generation}"
        )
    if slot.state != state:
        raise SlotEventRejected(f"slot {slot_index} is {slot.state}, not {state}")
    return slot


def _advance(slot: Slot, token_id: int, stopped: bool) -> Slot:
    generated = slot.generated + 1
    if stopped:
        return replace(
            slot, state=SLOT_COMPLETE, generated=generated, token_id=token_id, finish_reason=FINISH_STOP
        )
    if generated >= slot.request.max_tokens:
        return replace(
            slot, state=SLOT_COMPLETE, generated=generated, token_id=token_id, finish_reason=FINISH_LENGTH
        )
    return replace(slot, state=SLOT_ACTIVE, generated=generated, token_id=token_id)


def observe_prefill(
    schedule: SlotSchedule, *, slot: int, generation: int, token_id: int, stopped: bool
) -> SlotSchedule:
    """Record a prefill's first token; the slot decodes from its prompt length."""
    current = _require_slot(schedule, slot, generation, SLOT_PREFILLING)
    return _with_slot(schedule, _advance(current, token_id, stopped))


def observe_step(
    schedule: SlotSchedule, *, slot: int, generation: int, token_id: int, stopped: bool
) -> SlotSchedule:
    """Record one decoded token for an active slot."""
    current = _require_slot(schedule, slot, generation, SLOT_ACTIVE)
    advanced = _advance(current, token_id, stopped)
    return _with_slot(schedule, replace(advanced, position=current.position + 1))


def slot_running(schedule: SlotSchedule, slot: int, generation: int) -> bool:
    """Whether a decode row still belongs to the request it was issued for."""
    current = schedule.slots[slot]
    return current.generation == generation and current.state == SLOT_ACTIVE


def plan_window(schedule: SlotSchedule, policy: ContinuousSlotPolicy) -> DecodeWindow | None:
    """The next decode window: every active slot, ending no later than the first bound.

    The window never runs a slot past its own max_tokens, so the earliest length-bounded
    completion is refilled at the next boundary; an end-of-sequence token frees its slot
    at most `decode_window_steps` steps early.
    """
    rows = tuple(
        DecodeRow(slot.index, slot.generation, slot.request.request_id, slot.token_id, slot.position)
        for slot in schedule.slots
        if slot.state == SLOT_ACTIVE
    )
    if not rows:
        return None
    remaining = min(
        slot.request.max_tokens - slot.generated
        for slot in schedule.slots
        if slot.state == SLOT_ACTIVE
    )
    return DecodeWindow(steps=min(policy.decode_window_steps, remaining), rows=rows)


def mark_draining(schedule: SlotSchedule) -> SlotSchedule:
    """Admit nothing more into this group; it runs until empty, then a new group starts."""
    if not any(slot.state in OCCUPYING_STATES for slot in schedule.slots):
        return schedule
    return replace(schedule, draining=True)


def quarantine_group(schedule: SlotSchedule) -> tuple[SlotSchedule, tuple[RetireSlot, ...]]:
    """Take every occupied slot out of service after a device failure.

    The returned retirements name the requests to fail. The slots stay quarantined, and
    nothing is admitted, until release_quarantine records that the device was reset.
    """
    failed = []
    slots = []
    for slot in schedule.slots:
        if slot.state in (SLOT_PREFILLING, SLOT_ACTIVE, SLOT_COMPLETE):
            failed.append(
                RetireSlot(slot.index, slot.generation, slot.request.request_id, FINISH_FAILED)
            )
            slots.append(replace(slot, state=SLOT_QUARANTINED))
        else:
            slots.append(slot)
    return replace(schedule, slots=tuple(slots), draining=True), tuple(failed)


def release_quarantine(schedule: SlotSchedule) -> SlotSchedule:
    """Free quarantined slots once the shell has reset every row of device state."""
    slots = tuple(
        Slot(index=slot.index, generation=slot.generation)
        if slot.state == SLOT_QUARANTINED
        else slot
        for slot in schedule.slots
    )
    released = replace(schedule, slots=slots)
    if any(slot.state in OCCUPYING_STATES for slot in slots):
        return released
    return replace(released, route=None, draining=False)


def drain_queue(schedule: SlotSchedule, reason: str) -> tuple[SlotSchedule, tuple[RejectRequest, ...]]:
    """Refuse every queued request, for shutdown or engine quarantine."""
    rejected = tuple(RejectRequest(queued.request.request_id, reason) for queued in schedule.queue)
    return replace(schedule, queue=()), rejected
