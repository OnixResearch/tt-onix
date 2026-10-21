"""Pure matched evaluation of the resident slot decode trace.

r[impl tenstorrent.qwen38_p150x2.resident_slot_trace]

Four roots per workload serve the same agent-shaped conversations with the continuous-slot
scheduler and the prefix cache, each from a fresh process, in the policy's ABBA order:
baseline (the deployed package, which captures a decode trace for every slot group),
candidate (which keeps one stable trace across groups), candidate, baseline. Every turn
must decode the same tokens, and resume and save the same prefix, in all four roots.
Candidate roots may capture a trace on few turns only. Each timing gate compares, turn by
turn, the mean of the two candidate roots with the mean of the two baseline roots, which
cancels host load that drifts linearly across the roots. The ratios of each adjacent pair
are diagnostics only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .prefix_cache_admission_core import EXPECTED_DEVICE_IDS, Gate, Root, median

POLICY_FIELDS = frozenset({
    "schema_version",
    "profile_id",
    "prefix_cache_profile_id",
    "root_order",
    "workloads",
    "gates",
})
WORKLOAD_FIELDS = frozenset({"workload_id", "conversations", "turns"})
GATE_FIELDS = frozenset({
    "require_matched_identity",
    "require_exact_parity",
    "require_matching_prefix_cache",
    "maximum_candidate_capture_share",
    "maximum_median_wall_ratio",
    "maximum_wall_ratio",
    "maximum_median_ttft_ratio",
    "maximum_startup_dram_drop_bytes",
})
RATIO_GATES = (
    "maximum_median_wall_ratio",
    "maximum_wall_ratio",
    "maximum_median_ttft_ratio",
)
ROOT_ORDER = ("baseline", "candidate", "candidate", "baseline")
PAIR_NAMES = ("baseline_first/candidate_first", "candidate_second/baseline_second")


class ResidentTracePolicyError(ValueError):
    """The resident-trace admission policy is outside its closed schema."""


class ResidentTraceEvidenceError(ValueError):
    """The roots are incomplete or do not describe the declared comparison."""


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ResidentTracePolicyError(f"{name} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class ResidentTracePolicy:
    profile_id: str
    prefix_cache_profile_id: str
    # (workload_id, conversations, turns)
    workloads: tuple[tuple[str, int, int], ...]
    maximum_candidate_capture_share: float
    maximum_median_wall_ratio: float
    maximum_wall_ratio: float
    maximum_median_ttft_ratio: float
    maximum_startup_dram_drop_bytes: int

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> ResidentTracePolicy:
        error = ResidentTracePolicyError
        if not isinstance(root, Mapping) or set(root) != POLICY_FIELDS:
            raise error("policy fields differ from the closed schema")
        if root["schema_version"] != 1:
            raise error("schema_version is unsupported")
        if (
            not isinstance(root["root_order"], Sequence)
            or tuple(root["root_order"]) != ROOT_ORDER
        ):
            raise error("root_order must be baseline, candidate, candidate, baseline")
        gates = root["gates"]
        if not isinstance(gates, Mapping) or set(gates) != GATE_FIELDS:
            raise error("gate fields differ from the closed schema")
        for flag in (
            "require_matched_identity",
            "require_exact_parity",
            "require_matching_prefix_cache",
        ):
            if gates[flag] is not True:
                raise error(f"gates.{flag} must be true")
        workloads = []
        for item in root["workloads"]:
            if not isinstance(item, Mapping) or set(item) != WORKLOAD_FIELDS:
                raise error("workload fields differ from the closed schema")
            if not isinstance(item["workload_id"], str) or not item["workload_id"]:
                raise error("a workload needs an id")
            counts = (item["conversations"], item["turns"])
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count < 1
                for count in counts
            ):
                raise error(
                    "each workload needs at least one conversation and one turn"
                )
            workloads.append((item["workload_id"], *counts))
        if not workloads:
            raise error("the policy declares no workload")
        share = gates["maximum_candidate_capture_share"]
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not 0 <= share < 1
        ):
            raise error("the candidate capture share must lie in [0, 1)")
        drop = gates["maximum_startup_dram_drop_bytes"]
        if isinstance(drop, bool) or not isinstance(drop, int) or drop < 0:
            raise error("the DRAM budget must be a non-negative byte count")
        return cls(
            profile_id=str(root["profile_id"]),
            prefix_cache_profile_id=str(root["prefix_cache_profile_id"]),
            workloads=tuple(workloads),
            maximum_candidate_capture_share=float(share),
            maximum_startup_dram_drop_bytes=drop,
            **{name: _positive(gates[name], f"gates.{name}") for name in RATIO_GATES},
        )


@dataclass(frozen=True)
class ResidentTraceAdmission:
    admitted: bool
    gates: tuple[Gate, ...]
    reasons: tuple[str, ...]
    compared_turns: int
    baseline_package: str
    candidate_package: str
    # Diagnostics only: the same timing ratios for each adjacent pair, named in PAIR_NAMES.
    pairs: tuple[Mapping[str, float], ...]


def _timing(ttft: Sequence[float], wall: Sequence[float]) -> dict[str, float]:
    return {
        "median_wall_ratio": median(wall),
        "wall_ratio": max(wall),
        "median_ttft_ratio": median(ttft),
    }


def evaluate(
    policy: ResidentTracePolicy, groups: Sequence[Sequence[Root]]
) -> ResidentTraceAdmission:
    """Compare each workload's four ABBA roots turn by turn."""
    error = ResidentTraceEvidenceError
    if len(groups) != len(policy.workloads):
        raise error("every declared workload needs four roots")
    reasons: list[str] = []
    packages: dict[str, set[str]] = {"baseline": set(), "candidate": set()}
    ttft: list[float] = []
    wall: list[float] = []
    pair_ttft: tuple[list[float], list[float]] = ([], [])
    pair_wall: tuple[list[float], list[float]] = ([], [])
    parity = True
    cache_match = True
    capture_shares: list[float] = []
    dram_drops: list[int] = []
    for group, (workload_id, conversations, turns) in zip(
        groups, policy.workloads, strict=True
    ):
        if len(group) != len(ROOT_ORDER):
            raise error(
                "each workload needs four roots: baseline, candidate, candidate, baseline"
            )
        first = group[0]
        for role, root in zip(ROOT_ORDER, group, strict=True):
            packages[role].add(root.identity["package"])
            if root.simulated:
                reasons.append("simulated-root")
            if root.identity["runtime_identity"] != first.identity["runtime_identity"]:
                reasons.append("identity-differs:runtime_identity")
            if (
                tuple(root.identity.get("physical_pcie_device_ids") or ())
                != EXPECTED_DEVICE_IDS
            ):
                reasons.append("unexpected-devices")
            if (
                root.identity.get("prefix_cache_profile_id")
                != policy.prefix_cache_profile_id
            ):
                reasons.append("cache-profile-differs")
        if any(
            earlier.finished > later.started for earlier, later in zip(group, group[1:])
        ):
            reasons.append("roots-out-of-order")
        if len({len(root.startup_dram_free_bytes) for root in group}) != 1:
            raise error("the roots report different startup DRAM views")
        dram_drops.extend(
            max(baseline_first, baseline_second)
            - min(candidate_first, candidate_second)
            for baseline_first, candidate_first, candidate_second, baseline_second in zip(
                *(root.startup_dram_free_bytes for root in group)
            )
        )
        if any(root.workload != first.workload for root in group) or (
            first.workload[0],
            *first.workload[2:],
        ) != (workload_id, conversations, turns):
            reasons.append("workload-differs")
            continue
        for candidate in (group[1], group[2]):
            captures = [turn.trace_captures for turn in candidate.turns]
            if any(count is None for count in captures):
                raise error("a candidate turn does not report its trace captures")
            capture_shares.append(sum(1 for count in captures if count) / len(captures))
        by_turn = [
            {(turn.conversation, turn.turn): turn for turn in root.turns}
            for root in group
        ]
        for key in sorted(by_turn[0]):
            quad = tuple(turns_by_key[key] for turns_by_key in by_turn)
            baseline_first, candidate_first, candidate_second, baseline_second = quad
            parity = parity and len({turn.token_ids for turn in quad}) == 1
            cache_match = (
                cache_match
                and len({
                    (turn.prefix_hit_tokens, turn.prefix_saved_tokens) for turn in quad
                })
                == 1
            )
            ttft.append(
                (candidate_first.ttft_seconds + candidate_second.ttft_seconds)
                / (baseline_first.ttft_seconds + baseline_second.ttft_seconds)
            )
            wall.append(
                (candidate_first.wall_seconds + candidate_second.wall_seconds)
                / (baseline_first.wall_seconds + baseline_second.wall_seconds)
            )
            for index, (baseline, candidate) in enumerate((
                (baseline_first, candidate_first),
                (baseline_second, candidate_second),
            )):
                pair_ttft[index].append(candidate.ttft_seconds / baseline.ttft_seconds)
                pair_wall[index].append(candidate.wall_seconds / baseline.wall_seconds)
    if not wall:
        raise error("no turn could be compared")
    baseline_packages, candidate_packages = packages["baseline"], packages["candidate"]
    if (
        len(baseline_packages) != 1
        or len(candidate_packages) != 1
        or baseline_packages == candidate_packages
    ):
        reasons.append("package-roles-differ")
    identity_reasons = sorted(set(reasons))
    measured = _timing(ttft, wall)
    capture_share = max(capture_shares)
    dram_drop = max(dram_drops)
    gates = [
        Gate("matched_identity", not identity_reasons, True, not identity_reasons),
        Gate("exact_parity", parity, True, parity),
        Gate("matching_prefix_cache", cache_match, True, cache_match),
        Gate(
            "candidate_capture_share",
            capture_share,
            policy.maximum_candidate_capture_share,
            capture_share <= policy.maximum_candidate_capture_share,
        ),
        *(
            Gate(name, measured[name], bound, measured[name] <= bound)
            for name, bound in (
                ("median_wall_ratio", policy.maximum_median_wall_ratio),
                ("wall_ratio", policy.maximum_wall_ratio),
                ("median_ttft_ratio", policy.maximum_median_ttft_ratio),
            )
        ),
        Gate(
            "startup_dram_drop_bytes",
            float(dram_drop),
            float(policy.maximum_startup_dram_drop_bytes),
            dram_drop <= policy.maximum_startup_dram_drop_bytes,
        ),
    ]
    return ResidentTraceAdmission(
        admitted=all(gate.passed for gate in gates),
        gates=tuple(gates),
        reasons=tuple(identity_reasons)
        + tuple(f"gate-failed:{gate.name}" for gate in gates if not gate.passed),
        compared_turns=len(wall),
        baseline_package=min(baseline_packages),
        candidate_package=min(candidate_packages),
        pairs=tuple(_timing(pair_ttft[index], pair_wall[index]) for index in range(2)),
    )
