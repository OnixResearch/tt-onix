"""Pure matched evaluation of the hybrid prefix cache.

r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

Four roots per workload run the same agent-shaped conversations, turn by turn, on one
package, runtime, and device pair, each in a fresh process, in the policy's ABBA order:
cold, cached, cached, cold. The cached roots differ only by the prefix-cache policy.
Every turn must decode exactly the same tokens in all four roots, and both cached roots
must resume the same prefix. Each timing gate compares, turn by turn, the mean of the two
cached roots with the mean of the two cold roots, which cancels host load that drifts
linearly across the four roots. Turns that resumed from a cached prefix must be faster by
the predeclared ratios; turns that could not (a first turn, or a turn whose entry was
evicted) may cost at most the predeclared overhead of saving state. Startup memory may
drop by at most the declared budget. The ratios of each adjacent cold and cached pair are
diagnostics only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

POLICY_FIELDS = frozenset(
    {"schema_version", "profile_id", "prefix_cache_profile_id", "root_order", "workloads", "gates"}
)
GATE_FIELDS = frozenset(
    {
        "require_matched_identity",
        "require_exact_parity",
        "require_resumed_hits",
        "maximum_hit_median_ttft_ratio",
        "maximum_hit_median_wall_ratio",
        "maximum_miss_median_wall_ratio",
        "maximum_miss_wall_ratio",
        "maximum_startup_dram_drop_bytes",
    }
)
ROOT_ORDER = ("cold", "cached", "cached", "cold")
PAIR_NAMES = ("cold_first/cached_first", "cached_second/cold_second")
EXPECTED_DEVICE_IDS = (0, 1)


class PrefixAdmissionPolicyError(ValueError):
    """The prefix-cache admission policy is outside its closed schema."""


class PrefixAdmissionEvidenceError(ValueError):
    """A root is incomplete or does not describe the declared workload."""


def _number(value: Any, name: str, error: type[ValueError]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise error(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class PrefixAdmissionPolicy:
    profile_id: str
    prefix_cache_profile_id: str
    root_order: tuple[str, ...]
    workloads: tuple[tuple[int, int], ...]
    maximum_hit_median_ttft_ratio: float
    maximum_hit_median_wall_ratio: float
    maximum_miss_median_wall_ratio: float
    maximum_miss_wall_ratio: float
    maximum_startup_dram_drop_bytes: int

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> PrefixAdmissionPolicy:
        error = PrefixAdmissionPolicyError
        if not isinstance(root, Mapping) or set(root) != POLICY_FIELDS:
            raise error("policy fields differ from the closed schema")
        if root["schema_version"] != 2:
            raise error("schema_version is unsupported")
        if not isinstance(root["root_order"], Sequence) or tuple(root["root_order"]) != ROOT_ORDER:
            raise error("root_order must be cold, cached, cached, cold")
        gates = root["gates"]
        if not isinstance(gates, Mapping) or set(gates) != GATE_FIELDS:
            raise error("gate fields differ from the closed schema")
        for flag in ("require_matched_identity", "require_exact_parity", "require_resumed_hits"):
            if gates[flag] is not True:
                raise error(f"gates.{flag} must be true")
        workloads = tuple(
            (int(item["conversations"]), int(item["turns"])) for item in root["workloads"]
        )
        if not workloads or any(conversations < 1 or turns < 1 for conversations, turns in workloads):
            raise error("each workload needs at least one conversation and one turn")
        ratios = {
            name: _number(gates[name], f"gates.{name}", error)
            for name in (
                "maximum_hit_median_ttft_ratio",
                "maximum_hit_median_wall_ratio",
                "maximum_miss_median_wall_ratio",
                "maximum_miss_wall_ratio",
            )
        }
        if any(value <= 0 for value in ratios.values()):
            raise error("ratio gates must be positive")
        drop = gates["maximum_startup_dram_drop_bytes"]
        if isinstance(drop, bool) or not isinstance(drop, int) or drop < 0:
            raise error("the DRAM budget must be a non-negative byte count")
        return cls(
            profile_id=str(root["profile_id"]),
            prefix_cache_profile_id=str(root["prefix_cache_profile_id"]),
            root_order=ROOT_ORDER,
            workloads=workloads,
            maximum_startup_dram_drop_bytes=drop,
            **ratios,
        )


@dataclass(frozen=True)
class Turn:
    conversation: int
    turn: int
    token_ids: tuple[int, ...]
    ttft_seconds: float
    wall_seconds: float
    prefix_hit_tokens: int
    prefix_saved_tokens: int
    # Decode traces the server captured while serving the turn; None when not reported.
    trace_captures: int | None


def _count(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrefixAdmissionEvidenceError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Root:
    kind: str
    identity: Mapping[str, Any]
    simulated: bool
    # (workload_id, content BLAKE3, conversations, turns)
    workload: tuple[str, str, int, int]
    started: datetime
    finished: datetime
    startup_dram_free_bytes: tuple[int, ...]
    turns: tuple[Turn, ...]


def _utc(value: Any, name: str) -> datetime:
    error = PrefixAdmissionEvidenceError
    if not isinstance(value, str):
        raise error(f"{name} must be an ISO 8601 time")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise error(f"{name} must be an ISO 8601 time") from exc
    if moment.tzinfo is None:
        raise error(f"{name} must carry a UTC offset")
    return moment


def root_from_mapping(mapping: Mapping[str, Any], *, expected_kind: str) -> Root:
    error = PrefixAdmissionEvidenceError
    if not isinstance(mapping, Mapping) or mapping.get("schema_version") != 2:
        raise error("root schema_version is unsupported")
    if mapping.get("root_kind") != expected_kind:
        raise error(f"root kind is {mapping.get('root_kind')!r}, expected {expected_kind!r}")
    if not isinstance(mapping.get("simulated"), bool):
        raise error("root must state whether it is simulated")
    identity = mapping.get("identity")
    workload = mapping.get("workload")
    measured_at = mapping.get("measured_at")
    if not isinstance(identity, Mapping) or not isinstance(workload, Mapping) or not isinstance(measured_at, Mapping):
        raise error("root needs identity, workload, and measured_at")
    for name in ("package", "execution_identity", "runtime_identity"):
        if not isinstance(identity.get(name), str) or not identity[name]:
            raise error(f"identity.{name} is missing")
    if not isinstance(workload.get("blake3"), str) or not workload["blake3"]:
        raise error("workload.blake3 is missing")
    started = _utc(measured_at.get("started_utc"), "measured_at.started_utc")
    finished = _utc(measured_at.get("finished_utc"), "measured_at.finished_utc")
    if finished < started:
        raise error("the root finished before it started")
    requests = mapping.get("requests")
    if not isinstance(requests, Sequence) or not requests:
        raise error("root has no measured turns")
    turns = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise error("a turn must be a mapping")
        tokens = request.get("token_ids")
        if not isinstance(tokens, Sequence) or not tokens:
            raise error(f"turn {request.get('conversation')}/{request.get('turn')} decoded nothing")
        ttft = _number(request.get("ttft_seconds"), "ttft", error)
        wall = _number(request.get("wall_seconds"), "wall", error)
        if not 0 < ttft <= wall:
            raise error("a turn's first token must arrive after it starts and before it ends")
        turns.append(
            Turn(
                conversation=int(request["conversation"]),
                turn=int(request["turn"]),
                token_ids=tuple(int(token) for token in tokens),
                ttft_seconds=ttft,
                wall_seconds=wall,
                prefix_hit_tokens=_count(request.get("prefix_hit_tokens"), "prefix_hit_tokens"),
                prefix_saved_tokens=_count(request.get("prefix_saved_tokens"), "prefix_saved_tokens"),
                trace_captures=_count(request.get("trace_captures"), "trace_captures", optional=True),
            )
        )
    conversations, turn_count = int(workload["conversations"]), int(workload["turns"])
    expected = {(c, t) for c in range(conversations) for t in range(turn_count)}
    if {(turn.conversation, turn.turn) for turn in turns} != expected or len(turns) != len(expected):
        raise error("the root does not measure every declared turn exactly once")
    dram = mapping.get("startup", {}).get("device_dram_free_bytes")
    if not isinstance(dram, Sequence) or not dram:
        raise error("the root lacks the startup DRAM view")
    return Root(
        kind=expected_kind,
        identity=identity,
        simulated=mapping["simulated"],
        workload=(str(workload["workload_id"]), workload["blake3"], conversations, turn_count),
        started=started,
        finished=finished,
        startup_dram_free_bytes=tuple(int(value) for value in dram),
        turns=tuple(turns),
    )


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PrefixAdmissionEvidenceError("a median needs values")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


@dataclass
class _Ratios:
    """Cached-to-cold ratios of resumed and missed turns."""

    hit_ttft: list[float] = field(default_factory=list)
    hit_wall: list[float] = field(default_factory=list)
    miss_wall: list[float] = field(default_factory=list)

    def add(self, hit: bool, *, ttft: float, wall: float) -> None:
        if hit:
            self.hit_ttft.append(ttft)
            self.hit_wall.append(wall)
        else:
            self.miss_wall.append(wall)

    def summary(self) -> dict[str, float | None]:
        return {
            "hit_median_ttft_ratio": median(self.hit_ttft) if self.hit_ttft else None,
            "hit_median_wall_ratio": median(self.hit_wall) if self.hit_wall else None,
            "miss_median_wall_ratio": median(self.miss_wall) if self.miss_wall else None,
            "miss_wall_ratio": max(self.miss_wall) if self.miss_wall else None,
        }


@dataclass(frozen=True)
class Gate:
    name: str
    measured: float | bool
    bound: float | bool
    passed: bool


@dataclass(frozen=True)
class PrefixAdmission:
    admitted: bool
    gates: tuple[Gate, ...]
    reasons: tuple[str, ...]
    hit_turns: int
    miss_turns: int
    # Diagnostics only: the same ratios for each adjacent pair, named in PAIR_NAMES.
    pairs: tuple[Mapping[str, float | None], ...]


def _group_reasons(policy: PrefixAdmissionPolicy, group: Sequence[Root], shape: tuple[int, int]) -> list[str]:
    reasons = []
    first = group[0]
    for root in group:
        if root.simulated:
            reasons.append("simulated-root")
        for name in ("package", "execution_identity", "runtime_identity"):
            if root.identity[name] != first.identity[name]:
                reasons.append(f"identity-differs:{name}")
        if tuple(root.identity.get("physical_pcie_device_ids") or ()) != EXPECTED_DEVICE_IDS:
            reasons.append("unexpected-devices")
        profile = root.identity.get("prefix_cache_profile_id")
        if root.kind == "cached" and profile != policy.prefix_cache_profile_id:
            reasons.append("cache-profile-differs")
        if root.kind == "cold" and profile is not None:
            reasons.append("cold-root-used-a-cache")
    if any(earlier.finished > later.started for earlier, later in zip(group, group[1:])):
        reasons.append("roots-out-of-order")
    if any(root.workload != first.workload for root in group) or first.workload[2:] != shape:
        reasons.append("workload-differs")
    return reasons


def evaluate(policy: PrefixAdmissionPolicy, groups: Sequence[Sequence[Root]]) -> PrefixAdmission:
    """Compare each workload's four ABBA roots turn by turn."""
    if len(groups) != len(policy.workloads):
        raise PrefixAdmissionEvidenceError("every declared workload needs four roots")
    reasons: list[str] = []
    pooled = _Ratios()
    pairs = (_Ratios(), _Ratios())
    parity = True
    resumed = True
    dram_drops: list[int] = []
    for group, shape in zip(groups, policy.workloads, strict=True):
        if tuple(root.kind for root in group) != policy.root_order:
            raise PrefixAdmissionEvidenceError("each workload's roots must be cold, cached, cached, cold")
        group_reasons = _group_reasons(policy, group, shape)
        reasons.extend(group_reasons)
        if len({len(root.startup_dram_free_bytes) for root in group}) != 1:
            raise PrefixAdmissionEvidenceError("the roots report different startup DRAM views")
        dram_drops.extend(
            max(cold_first, cold_second) - min(cached_first, cached_second)
            for cold_first, cached_first, cached_second, cold_second in zip(
                *(root.startup_dram_free_bytes for root in group)
            )
        )
        if "workload-differs" in group_reasons:
            continue
        by_turn = [{(turn.conversation, turn.turn): turn for turn in root.turns} for root in group]
        for key in sorted(by_turn[0]):
            cold_first, cached_first, cached_second, cold_second = (turns[key] for turns in by_turn)
            parity = parity and (
                cold_first.token_ids == cached_first.token_ids == cached_second.token_ids == cold_second.token_ids
            )
            if cold_first.prefix_hit_tokens or cold_second.prefix_hit_tokens:
                reasons.append("cold-root-resumed")
            if cached_first.prefix_hit_tokens != cached_second.prefix_hit_tokens:
                reasons.append("cached-roots-disagree")
            hit = cached_first.prefix_hit_tokens > 0 and cached_second.prefix_hit_tokens > 0
            pooled.add(
                hit,
                ttft=(cached_first.ttft_seconds + cached_second.ttft_seconds)
                / (cold_first.ttft_seconds + cold_second.ttft_seconds),
                wall=(cached_first.wall_seconds + cached_second.wall_seconds)
                / (cold_first.wall_seconds + cold_second.wall_seconds),
            )
            for ratios, cold, cached in ((pairs[0], cold_first, cached_first), (pairs[1], cold_second, cached_second)):
                ratios.add(hit, ttft=cached.ttft_seconds / cold.ttft_seconds, wall=cached.wall_seconds / cold.wall_seconds)
        # The first declared workload has at most as many conversations as state
        # buffers, so every later turn must resume in both cached roots.
        if shape == policy.workloads[0]:
            resumed = resumed and all(
                turn.prefix_hit_tokens > 0 for root in group if root.kind == "cached" for turn in root.turns if turn.turn > 0
            )
    if not pooled.hit_ttft:
        raise PrefixAdmissionEvidenceError("no turn resumed from a cached prefix")
    measured = pooled.summary()
    identity_reasons = sorted(set(reasons))
    dram_drop = max(dram_drops)

    def bounded(name: str, bound: float) -> Gate:
        value = measured[name]
        return Gate(name, 0.0 if value is None else value, bound, value is None or value <= bound)

    gates = [
        Gate("matched_identity", not identity_reasons, True, not identity_reasons),
        Gate("exact_parity", parity, True, parity),
        Gate("resumed_hits", resumed, True, resumed),
        bounded("hit_median_ttft_ratio", policy.maximum_hit_median_ttft_ratio),
        bounded("hit_median_wall_ratio", policy.maximum_hit_median_wall_ratio),
        bounded("miss_median_wall_ratio", policy.maximum_miss_median_wall_ratio),
        bounded("miss_wall_ratio", policy.maximum_miss_wall_ratio),
        Gate(
            "startup_dram_drop_bytes",
            float(dram_drop),
            float(policy.maximum_startup_dram_drop_bytes),
            dram_drop <= policy.maximum_startup_dram_drop_bytes,
        ),
    ]
    return PrefixAdmission(
        admitted=all(gate.passed for gate in gates),
        gates=tuple(gates),
        reasons=tuple(identity_reasons) + tuple(f"gate-failed:{gate.name}" for gate in gates if not gate.passed),
        hit_turns=len(pooled.hit_ttft),
        miss_turns=len(pooled.miss_wall),
        pairs=tuple(ratios.summary() for ratios in pairs),
    )
