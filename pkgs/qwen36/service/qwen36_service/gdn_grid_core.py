"""Pure contiguous-chunk work assignment for the GDN grid guards.

r[impl tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]

Two GDN program factories cap a launch at one program instance per compute
core. The fused decode factory refuses any head count above the core count.
The chunk-prefill phased factory carries the same shape of guard in
``distribute_scan``.

This module owns the assignment arithmetic and its range validation. It stays
device-free and owns no I/O: the shell reads the grid, the method returns a
plan, and the shell passes the plan's runtime arguments to the kernels.

The plan is a uniform contiguous chunk. Every active core takes ``ceil(n / k)``
instances, so a chunk start is exactly ``core_index * chunk_length`` and one
forward pass builds the whole plan. Chunks partition ``[0, n)`` with no gap and
no overlap. Only the final chunk may be short, because ``n`` need not divide
``ceil(n / k)`` evenly.

When ``n <= k`` the chunk length is one and every used core takes exactly one
instance. That case reproduces the pre-change one-instance-per-core mapping, so
every receipt produced under the old mapping stays valid.
"""

from __future__ import annotations

from dataclasses import dataclass

# r[impl tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.invalid]

MINIMUM_CORE_COUNT = 1
MINIMUM_INSTANCE_COUNT = 1


class GridAssignmentError(ValueError):
    """A launch plan cannot be expressed as contiguous chunks."""


@dataclass(frozen=True)
class Chunk:
    """One core's contiguous instance range ``[start, start + count)``."""

    core_index: int
    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class ChunkPlan:
    """A complete assignment of ``instances`` over ``cores``."""

    instances: int
    cores: int
    chunk_length: int
    chunks: tuple[Chunk, ...]

    @property
    def active_cores(self) -> int:
        return len(self.chunks)


def _require_positive(value: object, name: str) -> int:
    """Return ``value`` as an int, or raise with a named diagnostic."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise GridAssignmentError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < 1:
        raise GridAssignmentError(f"{name} must be positive, got {value}")
    return value


def plan_chunks(instances: object, cores: object) -> ChunkPlan:
    """Assign ``instances`` contiguous work-items over at most ``cores`` cores.

    Raises ``GridAssignmentError`` for a zero or negative core count, a zero or
    negative instance count, a non-integer input, or a planned chunk length
    below one.
    """
    n = _require_positive(instances, "instances")
    k = _require_positive(cores, "cores")

    chunk_length = -(-n // k)  # ceiling division, one forward pass
    if chunk_length < 1:
        raise GridAssignmentError(
            f"chunk length must be at least one, got {chunk_length}"
        )

    active = chunk_length and -(-n // chunk_length)
    chunks = tuple(
        Chunk(
            core_index=i,
            start=i * chunk_length,
            count=min(chunk_length, n - i * chunk_length),
        )
        for i in range(active)
    )
    result = ChunkPlan(instances=n, cores=k, chunk_length=chunk_length, chunks=chunks)
    validate_plan(result)
    return result


def validate_plan(plan: ChunkPlan) -> None:
    """Raise unless ``plan`` partitions ``[0, instances)`` exactly.

    Rejects a gap, an overlap, a short interior chunk, a chunk outside the
    instance range, a duplicated core index, a chunk start that is not
    ``core_index * chunk_length``, and a core index at or above the core count.
    """
    if not plan.chunks:
        raise GridAssignmentError("plan has no chunks")
    if plan.chunk_length < 1:
        raise GridAssignmentError(
            f"chunk length must be at least one, got {plan.chunk_length}"
        )
    if len(plan.chunks) > plan.cores:
        raise GridAssignmentError(
            f"plan uses {len(plan.chunks)} cores above the available {plan.cores}"
        )

    expected_start = 0
    seen: set[int] = set()
    final = len(plan.chunks) - 1
    for position, chunk in enumerate(plan.chunks):
        if chunk.core_index in seen:
            raise GridAssignmentError(f"duplicate core index {chunk.core_index}")
        seen.add(chunk.core_index)
        if chunk.core_index < 0 or chunk.core_index >= plan.cores:
            raise GridAssignmentError(
                f"core index {chunk.core_index} outside the available {plan.cores} cores"
            )
        if chunk.core_index != position:
            raise GridAssignmentError(
                f"chunk {position} has core index {chunk.core_index}; cores must be contiguous"
            )
        if chunk.start != chunk.core_index * plan.chunk_length:
            raise GridAssignmentError(
                f"chunk {position} start {chunk.start} is not core_index * chunk_length"
            )
        if chunk.start != expected_start:
            raise GridAssignmentError(
                f"chunk {position} starts at {chunk.start}, expected {expected_start}"
            )
        if chunk.count < 1:
            raise GridAssignmentError(f"chunk {position} has count {chunk.count}")
        if chunk.count > plan.chunk_length:
            raise GridAssignmentError(
                f"chunk {position} has count {chunk.count} above {plan.chunk_length}"
            )
        if chunk.count < plan.chunk_length and position != final:
            raise GridAssignmentError(
                f"chunk {position} is short at {chunk.count}; only the final chunk may be short"
            )
        expected_start = chunk.stop

    if expected_start != plan.instances:
        raise GridAssignmentError(
            f"plan covers {expected_start} instances, expected {plan.instances}"
        )
