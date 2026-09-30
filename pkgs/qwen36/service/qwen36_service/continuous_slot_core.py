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

With a prefix-cache policy, admission also consults the cached hybrid prefixes
(prefix_cache_core). A request whose prompt extends a ready entry reads the entry's KV
blocks and resumes from its saved carried state, so it needs KV blocks only past the
prefix. Ordinarily it donates the longest chunk boundary that fits the cache's KV
budget; the opt-in root policy instead donates the chunk-aligned system/tool header
once, then later requests donate their conversation prefixes. The root retains its
own state buffer when a descendant is saved. Entries in use are pinned; unpinned
conversation entries are evicted before the one soft-retained root. A live request
can still evict the root for KV, and a device failure drops every entry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .decode_core import request_deadline_expired
from .prefix_cache_core import (
    CHUNK_TOKENS,
    DIGEST_HEX_LENGTH,
    PrefixCachePolicy,
    PrefixEntry,
    cached_blocks,
    eviction_order,
    expand_blocks,
    free_state_index,
    lookup,
)

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
    # Tokens one KV block holds.
    block_tokens: int

    def __post_init__(self) -> None:
        _require_int(self.first_block, "first_block", 0, 2**31 - 1)
        _require_int(self.row_blocks, "row_blocks", 1, 2**31 - 1)
        _require_int(self.end_block, "end_block", self.first_block + self.row_blocks, 2**31 - 1)
        _require_int(self.block_tokens, "block_tokens", 1, CHUNK_TOKENS)
        if CHUNK_TOKENS % self.block_tokens:
            raise ContinuousSlotPolicyError("a KV block must divide the prefill chunk")


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
    # Chained prefix digests (prefix_cache_core.prefix_digests); empty opts out of the cache.
    prefix_digests: tuple[str, ...] = ()
    # Chunk-aligned boundary of the rendered system/tool header, verified against token ids.
    shared_root_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ContinuousSlotPolicyError("request_id must be a non-empty string")
        if not isinstance(self.route, str) or not self.route:
            raise ContinuousSlotPolicyError("route must be a non-empty string")
        _require_int(self.prompt_tokens, "prompt_tokens", 1, 2**31 - 1)
        _require_int(self.max_tokens, "max_tokens", 1, 2**31 - 1)
        _require_int(self.blocks, "blocks", 1, 2**31 - 1)
        if len(self.prefix_digests) > (self.prompt_tokens - 1) // CHUNK_TOKENS or any(
            not isinstance(digest, str) or len(digest) != DIGEST_HEX_LENGTH for digest in self.prefix_digests
        ):
            raise ContinuousSlotPolicyError("prefix digests must cover chunks strictly inside the prompt")
        _require_int(self.shared_root_tokens, "shared_root_tokens", 0, self.prompt_tokens - 1)
        if self.shared_root_tokens % CHUNK_TOKENS or (
            self.shared_root_tokens and len(self.prefix_digests) < self.shared_root_tokens // CHUNK_TOKENS
        ):
            raise ContinuousSlotPolicyError("shared root must have a digest strictly inside the prompt")
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
    # The request's own KV run; a prefix hit adds the entry's blocks before it.
    block_start: int = 0
    block_count: int = 0
    prefix_entry: int | None = None
    prefix_tokens: int = 0
    prefix_ranges: tuple[tuple[int, int], ...] = ()
    donation_entry: int | None = None
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
    prefix_entries: tuple[PrefixEntry, ...] = ()
    next_entry_id: int = 1


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
    # A prefix hit: the entry's KV block ranges come first in the page-table row, the
    # saved state `resume_state` is restored, and prefill resumes at `prefix_tokens`.
    prefix_ranges: tuple[tuple[int, int], ...] = ()
    prefix_tokens: int = 0
    resume_state: int | None = None
    hit_entry: int | None = None
    # A donation: after the segment ending at `save_at`, save the carried state into
    # `save_state` for entry `donation_entry`.
    save_state: int | None = None
    save_at: int = 0
    donation_entry: int | None = None


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


def _block_intervals(blocks: frozenset[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for block in sorted(blocks):
        if intervals and intervals[-1][1] == block:
            intervals[-1] = (intervals[-1][0], block + 1)
        else:
            intervals.append((block, block + 1))
    return intervals


def _free_run(
    slots: tuple[Slot, ...],
    kv_range: KvBlockRange,
    blocks: int,
    entries: tuple[PrefixEntry, ...] | list[PrefixEntry] = (),
) -> int | None:
    """First block of the lowest free contiguous run of `blocks`, or None.

    Live requests' own runs and every cached entry's blocks are occupied."""
    runs = sorted(
        [
            (slot.block_start, slot.block_start + slot.block_count)
            for slot in slots
            if slot.state in OCCUPYING_STATES and slot.request is not None
        ]
        + _block_intervals(cached_blocks(entries))
    )
    start = kv_range.first_block
    for run_start, run_end in runs:
        if run_start - start >= blocks:
            return start
        start = max(start, run_end)
    return start if kv_range.end_block - start >= blocks else None


@dataclass(frozen=True)
class _Placement:
    block_start: int
    block_count: int
    hit: PrefixEntry | None
    entries: tuple[PrefixEntry, ...]


def _place(
    slots: list[Slot],
    entries: tuple[PrefixEntry, ...],
    kv_range: KvBlockRange,
    request: SlotRequest,
    caching: bool,
) -> _Placement | None:
    """Where a request's own KV run fits, evicting unused cache entries if that makes room."""
    hit = lookup(entries, request.prefix_digests) if caching else None
    prefix_blocks = hit.prefix_tokens // kv_range.block_tokens if hit is not None else 0
    own_blocks = request.blocks - prefix_blocks
    keep = frozenset({hit.entry_id}) if hit is not None else frozenset()
    current = list(entries)
    while True:
        block_start = _free_run(tuple(slots), kv_range, own_blocks, current)
        if block_start is not None:
            return _Placement(block_start, own_blocks, hit, tuple(current))
        victims = eviction_order(current, keep)
        if not victims:
            return None
        current.remove(victims[0])


def _plan_donation(
    entries: tuple[PrefixEntry, ...],
    request: SlotRequest,
    placement: _Placement,
    prefix_policy: PrefixCachePolicy,
    kv_range: KvBlockRange,
    entry_id: int,
    now: float,
) -> tuple[PrefixEntry, tuple[PrefixEntry, ...]] | None:
    """A new entry for the root or longest budget-fitting conversation prefix.

    A single system/tool root is soft-retained; a leaf never overwrites its saved state.
    An ordinary hit with one reader may still be superseded by its longer leaf."""
    save_at = (request.prompt_tokens - 1) // CHUNK_TOKENS * CHUNK_TOKENS
    root = False
    if (
        prefix_policy.shared_root_minimum_tokens
        and request.shared_root_tokens >= prefix_policy.shared_root_minimum_tokens
        and request.shared_root_tokens <= save_at
        and request.shared_root_tokens > (placement.hit.prefix_tokens if placement.hit is not None else 0)
        and not any(entry.root for entry in entries)
    ):
        save_at = request.shared_root_tokens
        root = True
    if not root:
        # A long turn can still donate a reusable prefix when its last chunk
        # exceeds the KV budget; keep the exact root boundary when present.
        maximum_chunks = prefix_policy.maximum_cached_blocks * kv_range.block_tokens // CHUNK_TOKENS
        save_at = min(save_at, maximum_chunks * CHUNK_TOKENS)
    prefix_tokens = placement.hit.prefix_tokens if placement.hit is not None else 0
    if save_at < prefix_policy.minimum_prefix_tokens or save_at <= prefix_tokens:
        return None
    digest = request.prefix_digests[save_at // CHUNK_TOKENS - 1] if request.prefix_digests else None
    if digest is None or len(request.prefix_digests) < save_at // CHUNK_TOKENS:
        return None
    if any(entry.digest == digest for entry in entries):
        return None
    donated_blocks = (save_at - prefix_tokens) // kv_range.block_tokens
    if donated_blocks > placement.block_count:
        return None
    ranges = (placement.hit.block_ranges if placement.hit is not None else ()) + (
        (placement.block_start, donated_blocks),
    )
    # The hit as recorded now, with this request's pin already counted.
    hit = (
        next(entry for entry in entries if entry.entry_id == placement.hit.entry_id)
        if placement.hit is not None
        else None
    )
    supersede = hit is not None and hit.pins == 1 and not hit.root
    keep = frozenset({hit.entry_id}) if hit is not None else frozenset()
    trial = [entry for entry in entries if not (supersede and entry.entry_id == hit.entry_id)]
    new_blocks = frozenset(expand_blocks(ranges))
    while len(cached_blocks(trial) | new_blocks) > prefix_policy.maximum_cached_blocks:
        victims = eviction_order(trial, keep)
        if not victims:
            return None
        trial.remove(victims[0])
    state_index = hit.state_index if supersede else free_state_index(trial, prefix_policy.state_slots)
    while state_index is None:
        victims = eviction_order(trial, keep)
        if not victims:
            return None
        trial.remove(victims[0])
        state_index = free_state_index(trial, prefix_policy.state_slots)
    entry = PrefixEntry(
        entry_id=entry_id,
        digest=digest,
        prefix_tokens=save_at,
        block_ranges=ranges,
        state_index=state_index,
        ready=False,
        pins=1 if supersede else 0,
        last_used=now,
        root=root,
    )
    entry.validate(kv_range.block_tokens, prefix_policy.state_slots)
    return entry, tuple(trial) + (entry,)


def _repin(entries: tuple[PrefixEntry, ...], entry_id: int | None, delta: int, now: float | None = None) -> tuple[PrefixEntry, ...]:
    if entry_id is None:
        return entries
    updated = []
    for entry in entries:
        if entry.entry_id == entry_id:
            entry = replace(
                entry,
                pins=entry.pins + delta,
                last_used=entry.last_used if now is None else now,
            )
        updated.append(entry)
    return tuple(updated)


def plan_boundary(
    schedule: SlotSchedule,
    policy: ContinuousSlotPolicy,
    kv_range: KvBlockRange,
    *,
    now: float,
    reads_waiting: bool = False,
    prefix_policy: PrefixCachePolicy | None = None,
    prefix_available: bool = False,
) -> tuple[SlotSchedule, tuple[BoundaryEffect, ...]]:
    """Decide everything that happens between two decode windows.

    Effects are ordered for execution: retirements, refusals, then one optional group
    start and the prefills. A waiting structured read admits nothing, so the group
    drains and the read runs on an idle device. Prefix hits and donations are planned
    only with a prefix policy and while the device can restore and save prefix state.
    """
    effects: list[BoundaryEffect] = []
    slots = list(schedule.slots)
    entries = schedule.prefix_entries
    next_entry_id = schedule.next_entry_id
    caching = prefix_policy is not None and prefix_available

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
        entries = _repin(entries, slot.prefix_entry, -1)
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
                    placement = _place(slots, entries, kv_range, request, caching)
                    if placement is not None:
                        chosen = (queue_index, placement)
                        break
                if queue_index == 0 and queued.bypassed >= policy.maximum_head_bypasses:
                    break
            if chosen is None:
                break
            queue_index, placement = chosen
            entries = placement.entries
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
            hit = placement.hit
            entries = _repin(entries, hit.entry_id if hit is not None else None, 1, now)
            donation = None
            if caching:
                donation = _plan_donation(
                    entries, request, placement, prefix_policy, kv_range, next_entry_id, now
                )
            reads_entry = hit.entry_id if hit is not None else None
            if donation is not None:
                donated, entries = donation
                next_entry_id += 1
                if donated.pins:
                    # The new entry superseded the one this request reads, and holds its pin.
                    reads_entry = donated.entry_id
            slots[slot.index] = Slot(
                index=slot.index,
                state=SLOT_PREFILLING,
                generation=generation,
                request=request,
                block_start=placement.block_start,
                block_count=placement.block_count,
                prefix_entry=reads_entry,
                prefix_tokens=hit.prefix_tokens if hit is not None else 0,
                prefix_ranges=hit.block_ranges if hit is not None else (),
                donation_entry=donated.entry_id if donation is not None else None,
                position=request.prompt_tokens,
            )
            effects.append(
                PrefillSlot(
                    slot.index,
                    generation,
                    request.request_id,
                    placement.block_start,
                    placement.block_count,
                    prefix_ranges=hit.block_ranges if hit is not None else (),
                    prefix_tokens=hit.prefix_tokens if hit is not None else 0,
                    resume_state=hit.state_index if hit is not None else None,
                    hit_entry=hit.entry_id if hit is not None else None,
                    save_state=donated.state_index if donation is not None else None,
                    save_at=donated.prefix_tokens if donation is not None else 0,
                    donation_entry=donated.entry_id if donation is not None else None,
                )
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
            prefix_entries=entries,
            next_entry_id=next_entry_id,
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
    """Record a prefill's first token; the slot decodes from its prompt length.

    A donation planned for this prefill becomes a ready entry: its state was saved and its
    KV blocks are written."""
    current = _require_slot(schedule, slot, generation, SLOT_PREFILLING)
    advanced = _with_slot(schedule, _advance(current, token_id, stopped))
    if current.donation_entry is None:
        return advanced
    entries = tuple(
        replace(entry, ready=True) if entry.entry_id == current.donation_entry else entry
        for entry in advanced.prefix_entries
    )
    return replace(advanced, prefix_entries=entries)


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
    # The device state behind every cached prefix is unknown after a failure.
    return replace(schedule, slots=tuple(slots), draining=True, prefix_entries=()), tuple(failed)


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


def invalidate_prefix_cache(schedule: SlotSchedule) -> SlotSchedule:
    """Drop every entry no live request reads: a structured read rewrote KV blocks, or the
    device can no longer restore prefix state."""
    return replace(schedule, prefix_entries=tuple(entry for entry in schedule.prefix_entries if entry.pins))


def drain_queue(schedule: SlotSchedule, reason: str) -> tuple[SlotSchedule, tuple[RejectRequest, ...]]:
    """Refuse every queued request, for shutdown or engine quarantine."""
    rejected = tuple(RejectRequest(queued.request.request_id, reason) for queued in schedule.queue)
    return replace(schedule, queue=()), rejected
