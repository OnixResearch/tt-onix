"""Pure queue-inclusive admission evaluator for the continuous-slot scheduler.

r[impl tenstorrent.qwen38_p150x2.continuous_slot_admission]

Two roots measure the same staggered, mixed-length workload: the group coordinator
(baseline) and the continuous-slot coordinator (candidate), on one package, runtime,
and device pair. Each root holds solo references (every request run alone on that
root's path) and several measured passes of the concurrent workload, all timed at the
client from each request's send to the end of its stream.

The evaluator applies the predeclared gates of continuous-slot-admission-policy.ncl to
every matched pair of passes and admits only if every gate holds on every pair.
Capture counts are reported and never gated: fewer captures alone cannot admit.
A structurally incomplete root raises AdmissionEvidenceError with the specific defect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

POLICY_SCHEMA_VERSION = 1
ROOT_SCHEMA_VERSION = 1
BASELINE_KIND = "group"
CANDIDATE_KIND = "continuous"
COMPLETED_FINISH_REASONS = frozenset({"stop", "length"})
EXPECTED_DEVICE_IDS = (0, 1)
PERCENTILE_95 = 0.95

POLICY_FIELDS = frozenset({"schema_version", "profile_id", "scheduler_profile_id", "workload", "gates"})
WORKLOAD_FIELDS = frozenset({"workload_id", "repetitions", "short_class", "long_class"})
GATE_FIELDS = frozenset(
    {
        "require_matched_identity",
        "require_all_completed",
        "require_exact_candidate_parity",
        "maximum_short_median_wall_ratio",
        "maximum_short_p95_wall_ratio",
        "maximum_long_p95_wall_ratio",
        "maximum_slowdown_ratio",
        "maximum_inter_token_gap_seconds",
        "maximum_median_inter_token_gap_ratio",
        "minimum_delivered_throughput_ratio",
        "minimum_startup_dram_free_ratio",
    }
)
IDENTITY_FIELDS = frozenset(
    {
        "package",
        "execution_identity",
        "runtime_identity",
        "physical_pcie_device_ids",
        "scheduler_profile_id",
        "scheduler_policy_blake3",
    }
)


class AdmissionPolicyError(ValueError):
    """The admission policy is outside its closed schema."""


class AdmissionEvidenceError(ValueError):
    """A root is structurally incomplete or does not describe the declared workload."""


def _require_mapping(value: Any, name: str, error: type[ValueError]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be a mapping")
    return value


def _require_fields(value: Mapping[str, Any], fields: frozenset[str], name: str, error: type[ValueError]) -> None:
    if set(value) != fields:
        missing = sorted(fields - set(value))
        extra = sorted(set(value) - fields)
        raise error(f"{name} fields differ from the closed schema: missing {missing}, extra {extra}")


def _require_number(value: Any, name: str, error: type[ValueError], *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise error(f"{name} must be a finite number of at least {minimum}")
    return number


def _require_int(value: Any, name: str, error: type[ValueError], *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error(f"{name} must be an integer of at least {minimum}")
    return value


def _require_string(value: Any, name: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class AdmissionPolicy:
    profile_id: str
    scheduler_profile_id: str
    workload_id: str
    repetitions: int
    short_class: str
    long_class: str
    maximum_short_median_wall_ratio: float
    maximum_short_p95_wall_ratio: float
    maximum_long_p95_wall_ratio: float
    maximum_slowdown_ratio: float
    maximum_inter_token_gap_seconds: float
    maximum_median_inter_token_gap_ratio: float
    minimum_delivered_throughput_ratio: float
    minimum_startup_dram_free_ratio: float

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> AdmissionPolicy:
        error = AdmissionPolicyError
        root = _require_mapping(root, "policy", error)
        _require_fields(root, POLICY_FIELDS, "policy", error)
        if _require_int(root["schema_version"], "schema_version", error) != POLICY_SCHEMA_VERSION:
            raise error("schema_version is unsupported")
        workload = _require_mapping(root["workload"], "workload", error)
        gates = _require_mapping(root["gates"], "gates", error)
        _require_fields(workload, WORKLOAD_FIELDS, "workload", error)
        _require_fields(gates, GATE_FIELDS, "gates", error)
        for flag in ("require_matched_identity", "require_all_completed", "require_exact_candidate_parity"):
            if gates[flag] is not True:
                raise error(f"gates.{flag} must be true")
        short_class = _require_string(workload["short_class"], "short_class", error)
        long_class = _require_string(workload["long_class"], "long_class", error)
        if short_class == long_class:
            raise error("the short and long classes must differ")

        def positive(field: str) -> float:
            value = _require_number(gates[field], f"gates.{field}", error)
            if value <= 0:
                raise error(f"gates.{field} must be positive")
            return value

        return cls(
            profile_id=_require_string(root["profile_id"], "profile_id", error),
            scheduler_profile_id=_require_string(
                root["scheduler_profile_id"], "scheduler_profile_id", error
            ),
            workload_id=_require_string(workload["workload_id"], "workload_id", error),
            repetitions=_require_int(workload["repetitions"], "repetitions", error, minimum=1),
            short_class=short_class,
            long_class=long_class,
            maximum_short_median_wall_ratio=positive("maximum_short_median_wall_ratio"),
            maximum_short_p95_wall_ratio=positive("maximum_short_p95_wall_ratio"),
            maximum_long_p95_wall_ratio=positive("maximum_long_p95_wall_ratio"),
            maximum_slowdown_ratio=positive("maximum_slowdown_ratio"),
            maximum_inter_token_gap_seconds=positive("maximum_inter_token_gap_seconds"),
            maximum_median_inter_token_gap_ratio=positive("maximum_median_inter_token_gap_ratio"),
            minimum_delivered_throughput_ratio=positive("minimum_delivered_throughput_ratio"),
            minimum_startup_dram_free_ratio=positive("minimum_startup_dram_free_ratio"),
        )


@dataclass(frozen=True)
class RequestObservation:
    """One request of one concurrent pass; seconds are relative to the pass start."""

    name: str
    request_class: str
    arrival_seconds: float
    first_event_seconds: float | None
    done_seconds: float | None
    content_event_seconds: tuple[float, ...]
    token_ids: tuple[int, ...]
    finish_reason: str | None
    trace_captures: int

    @property
    def completed(self) -> bool:
        return (
            self.finish_reason in COMPLETED_FINISH_REASONS
            and self.done_seconds is not None
            and self.first_event_seconds is not None
            and bool(self.token_ids)
        )

    @property
    def wall_seconds(self) -> float:
        if self.done_seconds is None:
            raise AdmissionEvidenceError(f"{self.name} has no completion time")
        return self.done_seconds - self.arrival_seconds


@dataclass(frozen=True)
class SoloObservation:
    name: str
    token_ids: tuple[int, ...]
    wall_seconds: float


@dataclass(frozen=True)
class MeasurementRoot:
    kind: str
    simulated: bool
    identity: Mapping[str, Any]
    workload_id: str
    workload_blake3: str
    classes: Mapping[str, str]
    startup_dram_free_bytes: tuple[int, ...]
    solo: Mapping[str, SoloObservation]
    passes: tuple[tuple[RequestObservation, ...], ...]


def _times(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise AdmissionEvidenceError(f"{name} must be a list of times")
    times = tuple(_require_number(value, name, AdmissionEvidenceError) for value in values)
    if any(later < earlier for earlier, later in zip(times, times[1:], strict=False)):
        raise AdmissionEvidenceError(f"{name} must not go backwards")
    return times


def _tokens(values: Any, name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise AdmissionEvidenceError(f"{name} must be a list of token ids")
    return tuple(_require_int(value, name, AdmissionEvidenceError) for value in values)


def root_from_mapping(mapping: Mapping[str, Any], *, expected_kind: str) -> MeasurementRoot:
    """Validate one measurement root; every defect names the missing or broken part."""
    error = AdmissionEvidenceError
    root = _require_mapping(mapping, "root", error)
    if root.get("schema_version") != ROOT_SCHEMA_VERSION:
        raise error("root schema_version is unsupported")
    kind = root.get("root_kind")
    if kind != expected_kind:
        raise error(f"root kind is {kind!r}, expected {expected_kind!r}")
    simulated = root.get("simulated")
    if not isinstance(simulated, bool):
        raise error("root must state whether it is simulated")
    identity = _require_mapping(root.get("identity"), "identity", error)
    _require_fields(identity, IDENTITY_FIELDS, "identity", error)
    for field in ("package", "execution_identity", "runtime_identity"):
        _require_string(identity[field], f"identity.{field}", error)
    workload = _require_mapping(root.get("workload"), "workload", error)
    workload_id = _require_string(workload.get("workload_id"), "workload.workload_id", error)
    workload_blake3 = _require_string(workload.get("blake3"), "workload.blake3", error)
    requests = workload.get("requests")
    if not isinstance(requests, Sequence) or not requests:
        raise error("workload.requests must list the declared requests")
    classes: dict[str, str] = {}
    for request in requests:
        request = _require_mapping(request, "workload request", error)
        name = _require_string(request.get("name"), "workload request name", error)
        if name in classes:
            raise error(f"workload request {name} is declared twice")
        classes[name] = _require_string(request.get("class"), f"{name} class", error)
    startup = _require_mapping(root.get("startup"), "startup", error)
    # The allocator's DRAM view after startup, as the service's device-dram stage reports it.
    dram = startup.get("device_dram_free_bytes")
    if not isinstance(dram, Sequence) or isinstance(dram, (str, bytes)) or not dram:
        raise error("startup.device_dram_free_bytes must hold the startup DRAM view")
    startup_dram = tuple(_require_int(value, "device DRAM free bytes", error, minimum=1) for value in dram)

    solo_entries = root.get("solo")
    if not isinstance(solo_entries, Sequence):
        raise error("solo must list one reference per request")
    solo: dict[str, SoloObservation] = {}
    for entry in solo_entries:
        entry = _require_mapping(entry, "solo entry", error)
        name = _require_string(entry.get("name"), "solo name", error)
        tokens = _tokens(entry.get("token_ids"), f"solo {name} token_ids")
        if not tokens:
            raise error(f"solo {name} generated no tokens")
        wall = _require_number(entry.get("wall_seconds"), f"solo {name} wall", error)
        if wall <= 0:
            raise error(f"solo {name} wall must be positive")
        solo[name] = SoloObservation(name=name, token_ids=tokens, wall_seconds=wall)
    if set(solo) != set(classes):
        raise error(f"solo references cover {sorted(solo)}, the workload declares {sorted(classes)}")

    pass_entries = root.get("repetitions")
    if not isinstance(pass_entries, Sequence) or not pass_entries:
        raise error("repetitions must list the measured passes")
    passes = []
    for index, entry in enumerate(pass_entries):
        entry = _require_mapping(entry, f"repetition {index}", error)
        observations = []
        for request in entry.get("requests") or ():
            request = _require_mapping(request, f"repetition {index} request", error)
            name = _require_string(request.get("name"), "request name", error)
            if name not in classes:
                raise error(f"repetition {index} measured undeclared request {name}")
            if request.get("class") != classes[name]:
                raise error(f"repetition {index} request {name} changed class")
            first = request.get("first_event_seconds")
            done = request.get("done_seconds")
            finish = request.get("finish_reason")
            observations.append(
                RequestObservation(
                    name=name,
                    request_class=classes[name],
                    arrival_seconds=_require_number(request.get("arrival_seconds"), f"{name} arrival", error),
                    first_event_seconds=None if first is None else _require_number(first, f"{name} first event", error),
                    done_seconds=None if done is None else _require_number(done, f"{name} done", error),
                    content_event_seconds=_times(request.get("content_event_seconds"), f"{name} content events"),
                    token_ids=_tokens(request.get("token_ids", ()), f"{name} token_ids"),
                    finish_reason=None if finish is None else _require_string(finish, f"{name} finish", error),
                    trace_captures=_require_int(request.get("trace_captures", 0), f"{name} captures", error),
                )
            )
        if sorted(observation.name for observation in observations) != sorted(classes):
            raise error(f"repetition {index} does not measure every declared request exactly once")
        passes.append(tuple(observations))
    return MeasurementRoot(
        kind=kind,
        simulated=simulated,
        identity=identity,
        workload_id=workload_id,
        workload_blake3=workload_blake3,
        classes=classes,
        startup_dram_free_bytes=startup_dram,
        solo=solo,
        passes=tuple(passes),
    )


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AdmissionEvidenceError("a median needs at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        raise AdmissionEvidenceError("a percentile needs at least one value")
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class PassMetrics:
    short_median_wall: float
    short_p95_wall: float
    long_p95_wall: float
    maximum_slowdown: float
    maximum_inter_token_gap: float
    median_inter_token_gap: float
    delivered_throughput: float
    completed: bool
    trace_captures: int
    parity_mismatches: tuple[str, ...]


def pass_metrics(root: MeasurementRoot, observations: tuple[RequestObservation, ...], policy: AdmissionPolicy) -> PassMetrics:
    completed = all(observation.completed for observation in observations)
    finished = [observation for observation in observations if observation.completed]

    def walls(request_class: str) -> list[float]:
        values = [o.wall_seconds for o in finished if o.request_class == request_class]
        if not values:
            raise AdmissionEvidenceError(f"no completed {request_class} request to measure")
        return values

    gaps = [
        later - earlier
        for observation in finished
        for earlier, later in zip(
            observation.content_event_seconds, observation.content_event_seconds[1:], strict=False
        )
    ]
    if not gaps:
        raise AdmissionEvidenceError("no request streamed two content events")
    if finished:
        makespan = max(o.done_seconds for o in finished) - min(o.arrival_seconds for o in observations)
        delivered = sum(len(o.token_ids) for o in finished) / makespan if makespan > 0 else 0.0
    else:
        delivered = 0.0
    return PassMetrics(
        short_median_wall=median(walls(policy.short_class)),
        short_p95_wall=percentile(walls(policy.short_class), PERCENTILE_95),
        long_p95_wall=percentile(walls(policy.long_class), PERCENTILE_95),
        maximum_slowdown=max(o.wall_seconds / root.solo[o.name].wall_seconds for o in finished),
        maximum_inter_token_gap=max(gaps),
        median_inter_token_gap=median(gaps),
        delivered_throughput=delivered,
        completed=completed,
        trace_captures=sum(o.trace_captures for o in observations),
        parity_mismatches=tuple(
            sorted(o.name for o in observations if o.token_ids != root.solo[o.name].token_ids)
        ),
    )


@dataclass(frozen=True)
class GateResult:
    name: str
    measured: float | bool
    bound: float | bool
    passed: bool


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    gates: tuple[GateResult, ...]
    reasons: tuple[str, ...]
    baseline: tuple[PassMetrics, ...]
    candidate: tuple[PassMetrics, ...]
    cross_root_parity_mismatches: tuple[str, ...]


def _identity_mismatches(policy: AdmissionPolicy, baseline: MeasurementRoot, candidate: MeasurementRoot) -> list[str]:
    reasons = []
    if baseline.simulated or candidate.simulated:
        reasons.append("simulated-root")
    for field in ("package", "execution_identity", "runtime_identity"):
        if baseline.identity[field] != candidate.identity[field]:
            reasons.append(f"identity-differs:{field}")
    for root in (baseline, candidate):
        if tuple(root.identity["physical_pcie_device_ids"] or ()) != EXPECTED_DEVICE_IDS:
            reasons.append(f"unexpected-devices:{root.kind}")
        if root.workload_id != policy.workload_id:
            reasons.append(f"workload-id-differs:{root.kind}")
        if len(root.passes) != policy.repetitions:
            reasons.append(f"repetition-count:{root.kind}")
    if baseline.workload_blake3 != candidate.workload_blake3 or baseline.classes != candidate.classes:
        reasons.append("workload-differs")
    if candidate.identity["scheduler_profile_id"] != policy.scheduler_profile_id:
        reasons.append("candidate-scheduler-profile-differs")
    if not candidate.identity["scheduler_policy_blake3"]:
        reasons.append("candidate-scheduler-policy-unbound")
    if baseline.identity["scheduler_profile_id"] is not None:
        reasons.append("baseline-is-not-the-group-coordinator")
    return reasons


def evaluate(policy: AdmissionPolicy, baseline: MeasurementRoot, candidate: MeasurementRoot) -> AdmissionResult:
    """Apply every gate to every matched pair of passes; admit only if all hold."""
    if baseline.kind != BASELINE_KIND or candidate.kind != CANDIDATE_KIND:
        raise AdmissionEvidenceError("the baseline must be the group root and the candidate the continuous root")
    identity_reasons = _identity_mismatches(policy, baseline, candidate)
    pairs = min(len(baseline.passes), len(candidate.passes))
    baseline_metrics = tuple(pass_metrics(baseline, observations, policy) for observations in baseline.passes)
    candidate_metrics = tuple(pass_metrics(candidate, observations, policy) for observations in candidate.passes)
    pair_metrics = list(zip(baseline_metrics[:pairs], candidate_metrics[:pairs], strict=True))

    def ratios(field: str) -> list[float]:
        values = []
        for b, c in pair_metrics:
            base = getattr(b, field)
            if base <= 0:
                raise AdmissionEvidenceError(f"the baseline {field} is not positive")
            values.append(getattr(c, field) / base)
        return values

    def worst_ratio(field: str) -> float:
        return max(ratios(field))

    def worst_inverse_ratio(field: str) -> float:
        return min(ratios(field))

    completed = all(m.completed for m in baseline_metrics + candidate_metrics)
    candidate_parity = not any(m.parity_mismatches for m in candidate_metrics)
    gates = [
        GateResult("matched_identity", not identity_reasons, True, not identity_reasons),
        GateResult("all_completed", completed, True, completed),
        GateResult("exact_candidate_parity", candidate_parity, True, candidate_parity),
    ]
    for name, field, bound in (
        ("short_median_wall_ratio", "short_median_wall", policy.maximum_short_median_wall_ratio),
        ("short_p95_wall_ratio", "short_p95_wall", policy.maximum_short_p95_wall_ratio),
        ("long_p95_wall_ratio", "long_p95_wall", policy.maximum_long_p95_wall_ratio),
        ("slowdown_ratio", "maximum_slowdown", policy.maximum_slowdown_ratio),
        ("median_inter_token_gap_ratio", "median_inter_token_gap", policy.maximum_median_inter_token_gap_ratio),
    ):
        measured = worst_ratio(field)
        gates.append(GateResult(name, measured, bound, measured <= bound))
    worst_gap = max(m.maximum_inter_token_gap for m in candidate_metrics)
    gates.append(
        GateResult(
            "inter_token_gap_seconds",
            worst_gap,
            policy.maximum_inter_token_gap_seconds,
            worst_gap <= policy.maximum_inter_token_gap_seconds,
        )
    )
    throughput = worst_inverse_ratio("delivered_throughput")
    gates.append(
        GateResult(
            "delivered_throughput_ratio",
            throughput,
            policy.minimum_delivered_throughput_ratio,
            throughput >= policy.minimum_delivered_throughput_ratio,
        )
    )
    if len(baseline.startup_dram_free_bytes) != len(candidate.startup_dram_free_bytes):
        raise AdmissionEvidenceError("the roots report different DRAM views")
    dram = min(
        c / b for b, c in zip(baseline.startup_dram_free_bytes, candidate.startup_dram_free_bytes, strict=True)
    )
    gates.append(
        GateResult(
            "startup_dram_free_ratio",
            dram,
            policy.minimum_startup_dram_free_ratio,
            dram >= policy.minimum_startup_dram_free_ratio,
        )
    )
    reasons = tuple(identity_reasons) + tuple(f"gate-failed:{gate.name}" for gate in gates if not gate.passed)
    cross = tuple(
        sorted(
            name
            for name in candidate.solo
            if name in baseline.solo and candidate.solo[name].token_ids != baseline.solo[name].token_ids
        )
    )
    return AdmissionResult(
        admitted=all(gate.passed for gate in gates),
        gates=tuple(gates),
        reasons=reasons,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        cross_root_parity_mismatches=cross,
    )
