"""Pure matched evaluation of the hybrid prefix cache.

r[impl tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

A cold root and a cached root run the same agent-shaped conversations, turn by turn, on
one package, runtime, and device pair. The cached root differs only by the prefix-cache
policy. Every turn must decode exactly the tokens it decodes cold. Turns that resumed
from a cached prefix must be faster by the predeclared ratios; turns that could not (a
first turn, or a turn whose entry was evicted) may cost at most the predeclared overhead
of saving state. Startup memory may drop by at most the declared budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

POLICY_FIELDS = frozenset({"schema_version", "profile_id", "prefix_cache_profile_id", "workloads", "gates"})
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
        if root["schema_version"] != 1:
            raise error("schema_version is unsupported")
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


@dataclass(frozen=True)
class Root:
    kind: str
    identity: Mapping[str, Any]
    simulated: bool
    workload: tuple[str, int, int]
    startup_dram_free_bytes: tuple[int, ...]
    turns: tuple[Turn, ...]


def root_from_mapping(mapping: Mapping[str, Any], *, expected_kind: str) -> Root:
    error = PrefixAdmissionEvidenceError
    if not isinstance(mapping, Mapping) or mapping.get("schema_version") != 1:
        raise error("root schema_version is unsupported")
    if mapping.get("root_kind") != expected_kind:
        raise error(f"root kind is {mapping.get('root_kind')!r}, expected {expected_kind!r}")
    if not isinstance(mapping.get("simulated"), bool):
        raise error("root must state whether it is simulated")
    identity = mapping.get("identity")
    workload = mapping.get("workload")
    if not isinstance(identity, Mapping) or not isinstance(workload, Mapping):
        raise error("root needs identity and workload")
    for field in ("package", "execution_identity", "runtime_identity"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise error(f"identity.{field} is missing")
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
        turns.append(
            Turn(
                conversation=int(request["conversation"]),
                turn=int(request["turn"]),
                token_ids=tuple(int(token) for token in tokens),
                ttft_seconds=_number(request.get("ttft_seconds"), "ttft", error),
                wall_seconds=_number(request.get("wall_seconds"), "wall", error),
                prefix_hit_tokens=int(request.get("prefix_hit_tokens") or 0),
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
        workload=(str(workload["workload_id"]), conversations, turn_count),
        startup_dram_free_bytes=tuple(int(value) for value in dram),
        turns=tuple(turns),
    )


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PrefixAdmissionEvidenceError("a median needs values")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


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


def evaluate(
    policy: PrefixAdmissionPolicy,
    pairs: Sequence[tuple[Root, Root]],
) -> PrefixAdmission:
    """Compare each (cold, cached) workload pair turn by turn."""
    if len(pairs) != len(policy.workloads):
        raise PrefixAdmissionEvidenceError("every declared workload needs a cold and a cached root")
    identity_reasons = []
    hit_ttft, hit_wall, miss_wall = [], [], []
    parity = True
    resumed = True
    for (cold, cached), (conversations, turns) in zip(pairs, policy.workloads, strict=True):
        if cold.kind != "cold" or cached.kind != "cached":
            raise PrefixAdmissionEvidenceError("each pair must be (cold, cached)")
        if cold.simulated or cached.simulated:
            identity_reasons.append("simulated-root")
        for field in ("package", "execution_identity", "runtime_identity"):
            if cold.identity[field] != cached.identity[field]:
                identity_reasons.append(f"identity-differs:{field}")
        if tuple(cached.identity.get("physical_pcie_device_ids") or ()) != EXPECTED_DEVICE_IDS:
            identity_reasons.append("unexpected-devices")
        if cached.identity.get("prefix_cache_profile_id") != policy.prefix_cache_profile_id:
            identity_reasons.append("cache-profile-differs")
        if cold.identity.get("prefix_cache_profile_id") is not None:
            identity_reasons.append("cold-root-used-a-cache")
        if cold.workload != cached.workload or cold.workload[1:] != (conversations, turns):
            identity_reasons.append("workload-differs")
            continue
        cold_turns = {(turn.conversation, turn.turn): turn for turn in cold.turns}
        for turn in cached.turns:
            reference = cold_turns[(turn.conversation, turn.turn)]
            parity = parity and turn.token_ids == reference.token_ids
            if reference.prefix_hit_tokens:
                identity_reasons.append("cold-root-resumed")
            if turn.prefix_hit_tokens:
                hit_ttft.append(turn.ttft_seconds / reference.ttft_seconds)
                hit_wall.append(turn.wall_seconds / reference.wall_seconds)
            else:
                miss_wall.append(turn.wall_seconds / reference.wall_seconds)
        # The first declared workload has at most as many conversations as state
        # buffers, so every later turn must resume.
        if (conversations, turns) == policy.workloads[0]:
            resumed = resumed and all(turn.prefix_hit_tokens > 0 for turn in cached.turns if turn.turn > 0)
    if not hit_ttft:
        raise PrefixAdmissionEvidenceError("no turn resumed from a cached prefix")
    dram_drop = max(
        cold_value - cached_value
        for cold, cached in pairs
        for cold_value, cached_value in zip(cold.startup_dram_free_bytes, cached.startup_dram_free_bytes, strict=True)
    )
    gates = [
        Gate("matched_identity", not identity_reasons, True, not identity_reasons),
        Gate("exact_parity", parity, True, parity),
        Gate("resumed_hits", resumed, True, resumed),
        Gate("hit_median_ttft_ratio", median(hit_ttft), policy.maximum_hit_median_ttft_ratio, median(hit_ttft) <= policy.maximum_hit_median_ttft_ratio),
        Gate("hit_median_wall_ratio", median(hit_wall), policy.maximum_hit_median_wall_ratio, median(hit_wall) <= policy.maximum_hit_median_wall_ratio),
        Gate("miss_median_wall_ratio", median(miss_wall) if miss_wall else 0.0, policy.maximum_miss_median_wall_ratio,
             not miss_wall or median(miss_wall) <= policy.maximum_miss_median_wall_ratio),
        Gate("miss_wall_ratio", max(miss_wall) if miss_wall else 0.0, policy.maximum_miss_wall_ratio,
             not miss_wall or max(miss_wall) <= policy.maximum_miss_wall_ratio),
        Gate("startup_dram_drop_bytes", float(dram_drop), float(policy.maximum_startup_dram_drop_bytes),
             dram_drop <= policy.maximum_startup_dram_drop_bytes),
    ]
    reasons = tuple(sorted(set(identity_reasons))) + tuple(f"gate-failed:{gate.name}" for gate in gates if not gate.passed)
    return PrefixAdmission(
        admitted=all(gate.passed for gate in gates),
        gates=tuple(gates),
        reasons=reasons,
        hit_turns=len(hit_ttft),
        miss_turns=len(miss_wall),
    )
