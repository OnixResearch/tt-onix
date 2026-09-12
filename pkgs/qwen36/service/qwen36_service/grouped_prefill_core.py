"""Pure grouped-prefill eligibility for the Qwen3.8 P150x2 service.

r[impl tenstorrent.qwen38_p150x2.grouped_prefill_eligibility]

The adopted model already contains a grouped prefill path
(``Qwen36Model.prefill_paged_grouped``). Its own source comment derives its
group cap from TP4 geometry: the fused scan maps one ``(head, v-block)`` row
per core, so ``BH = B * Nv`` must fit the compute grid, and with ``Nv_tp = 12``
that yields ``B <= 8``. At TP2 the local value-head count doubles to 24, so the
same derivation gives a different bound. Copying the literal 8 into TP2 would
over-subscribe the grid.

This module owns that derivation. It stays device-free and pure: the caller
supplies the runtime capability, the measured compute grid, the local value-head
count, the bucket shape, and the scratch budget; the planner returns either a
grouped plan with explicit user order and state destinations, or a per-user
decision with a named reason. Nothing here allocates, and no fallback decision
happens after allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

# r[impl tenstorrent.qwen38_p150x2.grouped_prefill_eligibility.invalid]

MINIMUM_USER_COUNT = 1
MINIMUM_VALUE_HEADS = 1
MINIMUM_COMPUTE_CORES = 1
MINIMUM_GDN_CHUNK = 1
MINIMUM_INSTANCES_PER_CORE = 1
MINIMUM_SCRATCH_BYTES = 1

# A group of one user is the per-user route, so it is never a grouped plan.
MINIMUM_GROUP_SIZE = 2

# The model's grouped path processes one GDN bucket per forward and routes
# longer prompts elsewhere, so a bucket above this multiple is defined as a
# per-user candidate rather than a group.
MAXIMUM_BUCKET_CHUNKS = 2

REASON_ELIGIBLE = "grouped-eligible"
REASON_EMPTY_GROUP = "empty-group"
REASON_SINGLE_USER = "single-user"
REASON_BUCKET_ABOVE_SINGLE_PASS = "bucket-above-single-pass"
REASON_HEAD_CAPACITY_EXCEEDED = "head-capacity-exceeded"
REASON_SCRATCH_BUDGET_EXCEEDED = "scratch-budget-exceeded"


class GroupedPrefillPolicyError(ValueError):
    """The planner was given a shape that cannot describe a launch."""


@dataclass(frozen=True)
class RuntimeCapability:
    """What the linked runtime can actually do with scan program instances.

    ``chunked_scan_instances`` is true only for a runtime whose GDN scan
    factory chunks instances across cores. When false, the scan maps one
    instance per core and the grid bounds the head rows directly.
    """

    chunked_scan_instances: bool
    compute_cores: int
    instances_per_core: int

    def validate(self) -> None:
        if self.chunked_scan_instances:
            if self.instances_per_core < MINIMUM_INSTANCES_PER_CORE:
                raise GroupedPrefillPolicyError(
                    "a chunked scan needs at least one instance per core"
                )
        elif self.instances_per_core != MINIMUM_INSTANCES_PER_CORE:
            raise GroupedPrefillPolicyError(
                "a one-instance-per-core scan cannot claim a wider chunk"
            )
        if self.compute_cores < MINIMUM_COMPUTE_CORES:
            raise GroupedPrefillPolicyError("capability needs at least one compute core")

    @property
    def instance_capacity(self) -> int:
        """Head-instance capacity of one launch on this runtime."""
        return self.compute_cores * self.instances_per_core


@dataclass(frozen=True)
class GroupPlan:
    """Either a grouped prefill plan or a per-user decision with its reason."""

    grouped: bool
    reason: str
    bucket: int
    local_value_heads: int
    head_bound: int
    memory_bound: int
    groups: tuple[tuple[int, ...], ...]
    destinations: tuple[int, ...]


def _require_positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroupedPrefillPolicyError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < 1:
        raise GroupedPrefillPolicyError(f"{name} must be positive, got {value}")
    return value


def _require_non_negative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroupedPrefillPolicyError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if value < 0:
        raise GroupedPrefillPolicyError(f"{name} cannot be negative, got {value}")
    return value


def _per_user_plan(
    reason: str,
    *,
    user_count: int,
    bucket: int,
    local_value_heads: int,
    head_bound: int,
    memory_bound: int,
) -> GroupPlan:
    """A fallback that keeps every user on the staged per-user route."""
    return GroupPlan(
        grouped=False,
        reason=reason,
        bucket=bucket,
        local_value_heads=local_value_heads,
        head_bound=head_bound,
        memory_bound=memory_bound,
        groups=tuple((user,) for user in range(user_count)),
        destinations=tuple(range(user_count)),
    )


def plan_grouped_prefill(
    *,
    user_count: object,
    user_lengths: tuple,
    slots: tuple,
    bucket: object,
    local_value_heads: object,
    gdn_bucket_chunk: object,
    capability: RuntimeCapability,
    capacity_units: object,
    units_per_user: object,
) -> GroupPlan:
    """Decide whether these users may share one grouped prefill.

    ``gdn_bucket_chunk`` is the model's GDN prefill bucket chunk
    (``args.gdn_chunk_size``, 128), not the fused scan's internal chunk (32).
    The grouped path fits one bucket per forward, so a bucket above
    ``MAXIMUM_BUCKET_CHUNKS`` of that chunk is a per-user candidate.

    ``capacity_units`` and ``units_per_user`` share one caller-chosen unit: the
    adapter budgets rows of state, the fixtures budget bytes.

    ``slots`` names each user's persistent decode slot in request order, so the
    plan's destinations stay explicit: group row ``i`` installs into
    ``slots[user]`` and never into an assumed contiguous row.

    Returns a grouped plan only when every bound passes. Any failure returns the
    per-user plan with the bound that rejected it, before any allocation.
    """
    users = _require_non_negative(user_count, "user_count")
    heads = _require_positive(local_value_heads, "local_value_heads")
    chunk = _require_positive(gdn_bucket_chunk, "gdn_bucket_chunk")
    bucket_value = _require_positive(bucket, "bucket")
    budget = _require_positive(capacity_units, "capacity_units")
    per_user = _require_positive(units_per_user, "units_per_user")
    capability.validate()

    if len(user_lengths) != users or len(slots) != users:
        raise GroupedPrefillPolicyError(
            "user_lengths and slots must each carry one entry per user"
        )
    if len(set(slots)) != users:
        raise GroupedPrefillPolicyError("state destinations must be distinct")
    for length in user_lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise GroupedPrefillPolicyError(f"user length is not a positive int: {length!r}")

    # Derived bounds. Each is a count of users, never a copied literal.
    head_bound = capability.instance_capacity // heads
    memory_bound = budget // per_user if per_user <= budget else 0

    empty = _per_user_plan(
        REASON_EMPTY_GROUP,
        user_count=users,
        bucket=bucket_value,
        local_value_heads=heads,
        head_bound=head_bound,
        memory_bound=memory_bound,
    )
    if users == 0:
        return empty
    if max(user_lengths) > bucket_value:
        raise GroupedPrefillPolicyError("bucket is smaller than the longest user")
    if users == MINIMUM_USER_COUNT:
        return _per_user_plan(
            REASON_SINGLE_USER,
            user_count=users,
            bucket=bucket_value,
            local_value_heads=heads,
            head_bound=head_bound,
            memory_bound=memory_bound,
        )
    if bucket_value > MAXIMUM_BUCKET_CHUNKS * chunk:
        return _per_user_plan(
            REASON_BUCKET_ABOVE_SINGLE_PASS,
            user_count=users,
            bucket=bucket_value,
            local_value_heads=heads,
            head_bound=head_bound,
            memory_bound=memory_bound,
        )

    group_size = min(users, head_bound, memory_bound)
    if group_size < MINIMUM_GROUP_SIZE:
        reason = (
            REASON_SCRATCH_BUDGET_EXCEEDED
            if memory_bound < MINIMUM_GROUP_SIZE
            else REASON_HEAD_CAPACITY_EXCEEDED
        )
        return _per_user_plan(
            reason,
            user_count=users,
            bucket=bucket_value,
            local_value_heads=heads,
            head_bound=head_bound,
            memory_bound=memory_bound,
        )

    groups = tuple(
        tuple(range(start, min(start + group_size, users)))
        for start in range(0, users, group_size)
    )
    return GroupPlan(
        grouped=True,
        reason=REASON_ELIGIBLE,
        bucket=bucket_value,
        local_value_heads=heads,
        head_bound=head_bound,
        memory_bound=memory_bound,
        groups=groups,
        destinations=tuple(slots),
    )
