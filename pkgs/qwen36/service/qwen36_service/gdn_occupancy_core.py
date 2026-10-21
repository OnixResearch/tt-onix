"""Pure decode-occupancy planning for the GDN fused kernel.

r[impl tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]

The fused GDN decode factory maps one program instance per ``(batch, value
head)``. At TP=2 a device owns 24 value heads, so a launch carries ``24 * B``
instances against 110 measured compute cores. Two regimes follow:

- **idle grid** (B=1: 24 instances, 21.8 percent): splitting each head's 128x128
  state across cores raises occupancy;
- **looping grid** (B>=8: 192 or 384 instances): each core already runs two or
  four instances sequentially, and splitting the state cannot reduce per-core
  work, because instances grow by the same factor that per-instance work falls.

Splitting is therefore not free: the per-instance cost has a part that does not
divide with the value width (the ``k`` transpose is ``K x K``, plus reader, CB, and
launch overhead). This core models that explicitly, so a split is chosen only
when its estimated per-core cost actually falls, and never for the looping
regime.

Everything here is pure and device-free: the caller supplies the geometry, the
core count, the split candidates, and the measured fixed-cost share; the planner
returns the chosen split with its schedule and an explicit reason.
"""

from __future__ import annotations

from dataclasses import dataclass

# r[impl tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]

MINIMUM_HEADS = 1
MINIMUM_BATCH = 1
MINIMUM_VALUE_TILES = 1
MINIMUM_COMPUTE_CORES = 1
MINIMUM_SPLIT_FACTOR = 1

REASON_NO_SPLIT_IMPROVES_COST = "no-split-improves-cost"
REASON_SPLIT_SELECTED = "split-selected"
REASON_SPLIT_REFUSED_LOOPING_GRID = "split-refused-looping-grid"


class GdnOccupancyPolicyError(ValueError):
    """An occupancy plan cannot describe a launch."""


@dataclass(frozen=True)
class OccupancyPlan:
    """The chosen state split, its schedule, and the cost it was chosen for."""

    heads: int
    batch: int
    value_tiles: int
    cores: int
    split: int
    instances: int
    instances_per_core: int
    active_cores: int
    occupancy: float
    fixed_cost_share: float
    estimated_relative_cost: float
    per_core_instances: tuple[tuple[int, ...], ...]
    reason: str


def instance_coordinates(
    flat: object, *, heads: object, split: object
) -> tuple[int, int, int]:
    """Map a flat instance index to ``(batch, head, block)``, batch-major.

    The launch enumerates ``batch * heads * split`` instances in that order, so a
    contiguous chunk of flat indices is a contiguous slice of the launch.
    """
    index = _require_non_negative(flat, "flat")
    head_count = _require_positive(heads, "heads")
    split_count = _require_positive(split, "split")
    per_batch = head_count * split_count
    batch_index, remainder = divmod(index, per_batch)
    head, block = divmod(remainder, split_count)
    return batch_index, head, block


def _require_non_negative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GdnOccupancyPolicyError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < 0:
        raise GdnOccupancyPolicyError(f"{name} cannot be negative, got {value}")
    return value


def _require_positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GdnOccupancyPolicyError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < MINIMUM_HEADS:
        raise GdnOccupancyPolicyError(f"{name} must be positive, got {value}")
    return value


def candidate_splits(value_tiles: object) -> tuple[int, ...]:
    """Split factors that divide the value width exactly, ascending."""
    tiles = _require_positive(value_tiles, "value_tiles")
    return tuple(
        factor
        for factor in range(MINIMUM_SPLIT_FACTOR, tiles + 1)
        if tiles % factor == 0
    )


def _estimate_relative_cost(
    *,
    instances_per_core: int,
    split: int,
    fixed_cost_share: float,
) -> float:
    """Per-core cost relative to the unsplit, one-instance-per-core launch.

    ``fixed_cost_share`` is the fraction of a single instance's cost that does not
    divide with the value width. The variable part falls as ``1 / split``.
    """
    per_instance = fixed_cost_share + (1.0 - fixed_cost_share) / split
    return instances_per_core * per_instance


def plan_occupancy(
    *,
    heads: object,
    batch: object,
    value_tiles: object,
    cores: object,
    fixed_cost_share: object,
    minimum_occupancy: object = 1.0,
) -> OccupancyPlan:
    """Choose the state split that minimizes estimated per-core decode cost.

    ``fixed_cost_share`` is required rather than defaulted: it is a measurement,
    and inventing it here would write a performance claim into a policy. The
    caller passes what the probe measured.

    Returns the unsplit plan whenever no candidate split lowers the estimated
    cost, including every looping-grid case, and names why in ``reason``.
    """
    head_count = _require_positive(heads, "heads")
    batch_count = _require_positive(batch, "batch")
    tiles = _require_positive(value_tiles, "value_tiles")
    core_count = _require_positive(cores, "cores")
    if not isinstance(fixed_cost_share, (int, float)) or isinstance(
        fixed_cost_share, bool
    ):
        raise GdnOccupancyPolicyError(
            f"fixed_cost_share must be a number, got {type(fixed_cost_share).__name__}"
        )
    share = float(fixed_cost_share)
    if not 0.0 <= share < 1.0:
        raise GdnOccupancyPolicyError(
            f"fixed_cost_share must be in [0, 1), got {share}"
        )
    if not isinstance(minimum_occupancy, (int, float)) or isinstance(
        minimum_occupancy, bool
    ):
        raise GdnOccupancyPolicyError("minimum_occupancy must be a number")
    floor = float(minimum_occupancy)
    if not 0.0 < floor <= 1.0:
        raise GdnOccupancyPolicyError(
            f"minimum_occupancy must be in (0, 1], got {floor}"
        )

    head_instances = head_count * batch_count
    looping = head_instances > core_count

    def build(split: int, reason: str) -> OccupancyPlan:
        instances = head_instances * split
        instances_per_core = -(-instances // core_count)
        active_cores = -(-instances // instances_per_core)
        # Uniform contiguous chunks over the launch's own instance order, so the
        # schedule and the launch agree without a second mapping.
        per_core: list[tuple[int, ...]] = []
        for core_index in range(active_cores):
            start = core_index * instances_per_core
            count = min(instances_per_core, instances - start)
            per_core.append(tuple(range(start, start + count)))
        return OccupancyPlan(
            heads=head_count,
            batch=batch_count,
            value_tiles=tiles,
            cores=core_count,
            split=split,
            instances=instances,
            instances_per_core=instances_per_core,
            active_cores=active_cores,
            occupancy=active_cores / core_count,
            fixed_cost_share=share,
            estimated_relative_cost=_estimate_relative_cost(
                instances_per_core=instances_per_core,
                split=split,
                fixed_cost_share=share,
            ),
            per_core_instances=tuple(per_core),
            reason=reason,
        )

    unsplit = build(MINIMUM_SPLIT_FACTOR, REASON_NO_SPLIT_IMPROVES_COST)

    if looping:
        # A split multiplies instances without reducing any core's work.
        unsplit_split = build(MINIMUM_SPLIT_FACTOR, REASON_SPLIT_REFUSED_LOOPING_GRID)
        return unsplit_split

    best = unsplit
    best_split = MINIMUM_SPLIT_FACTOR
    for split in candidate_splits(tiles):
        if split == MINIMUM_SPLIT_FACTOR:
            continue
        candidate = build(split, REASON_SPLIT_SELECTED)
        if candidate.occupancy < floor:
            continue
        if candidate.estimated_relative_cost < best.estimated_relative_cost:
            best = candidate
            best_split = split
    if best_split == MINIMUM_SPLIT_FACTOR:
        return unsplit
    return best


def validate_plan(plan: OccupancyPlan) -> None:
    """Raise unless ``plan`` is internally consistent and covers every instance.

    Rejects a split that does not divide the value width, an instance count that
    disagrees with the geometry, per-core chunks that are not uniform contiguous
    blocks of at most the chunk length, a schedule that misses or repeats an
    instance, an occupancy that disagrees with the schedule, and a plan whose
    recorded cost does not match its own numbers.
    """
    if plan.heads < MINIMUM_HEADS or plan.batch < MINIMUM_BATCH:
        raise GdnOccupancyPolicyError("plan geometry is not positive")
    if plan.value_tiles < MINIMUM_VALUE_TILES or plan.cores < MINIMUM_COMPUTE_CORES:
        raise GdnOccupancyPolicyError("plan bounds are not positive")
    if plan.split < MINIMUM_SPLIT_FACTOR or plan.value_tiles % plan.split != 0:
        raise GdnOccupancyPolicyError(
            f"split {plan.split} does not divide {plan.value_tiles} value tiles"
        )
    expected_instances = plan.heads * plan.batch * plan.split
    if plan.instances != expected_instances:
        raise GdnOccupancyPolicyError(
            f"plan carries {plan.instances} instances, expected {expected_instances}"
        )
    expected_per_core = -(-plan.instances // plan.cores)
    if plan.instances_per_core != expected_per_core:
        raise GdnOccupancyPolicyError(
            f"chunk length {plan.instances_per_core} disagrees with the geometry"
        )
    if plan.active_cores != len(plan.per_core_instances):
        raise GdnOccupancyPolicyError("active core count disagrees with the schedule")
    if plan.active_cores > plan.cores:
        raise GdnOccupancyPolicyError("schedule uses more cores than are available")
    seen: set[int] = set()
    covered = 0
    for core_index, chunk in enumerate(plan.per_core_instances):
        if not chunk:
            raise GdnOccupancyPolicyError(f"core {core_index} has no instances")
        if len(chunk) > plan.instances_per_core:
            raise GdnOccupancyPolicyError(
                f"core {core_index} has {len(chunk)} instances above the chunk length"
            )
        expected_start = core_index * plan.instances_per_core
        for offset, flat in enumerate(chunk):
            if flat != expected_start + offset:
                raise GdnOccupancyPolicyError(
                    f"core {core_index} chunk is not contiguous at offset {offset}"
                )
            batch_index, head, block = instance_coordinates(
                flat, heads=plan.heads, split=plan.split
            )
            if batch_index >= plan.batch:
                raise GdnOccupancyPolicyError(
                    f"instance {flat} is outside the batch bound"
                )
            if head >= plan.heads:
                raise GdnOccupancyPolicyError(
                    f"instance {flat} is outside the head bound"
                )
            if block >= plan.split:
                raise GdnOccupancyPolicyError(
                    f"instance {flat} is outside the block bound"
                )
            if flat in seen:
                raise GdnOccupancyPolicyError(f"instance {flat} appears twice")
            seen.add(flat)
            covered += 1
    if covered != plan.instances:
        raise GdnOccupancyPolicyError(
            f"schedule covers {covered} instances, expected {plan.instances}"
        )
    expected_occupancy = plan.active_cores / plan.cores
    if abs(plan.occupancy - expected_occupancy) > 0:
        raise GdnOccupancyPolicyError("occupancy disagrees with the schedule")
    expected_cost = _estimate_relative_cost(
        instances_per_core=plan.instances_per_core,
        split=plan.split,
        fixed_cost_share=plan.fixed_cost_share,
    )
    if abs(plan.estimated_relative_cost - expected_cost) > 0:
        raise GdnOccupancyPolicyError(
            "recorded cost disagrees with the plan's own numbers"
        )
