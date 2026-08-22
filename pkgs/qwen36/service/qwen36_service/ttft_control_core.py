"""Pure policy and evidence evaluation for the Qwen same-process TTFT control."""

# r[impl tenstorrent.qwen38_p150x2.traced_ttft_control]

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REQUEST_CACHE_UNTRACED = "request-cache-untraced"
CAPACITY_CACHE_UNTRACED = "capacity-cache-untraced"
CAPACITY_CACHE_TRACED = "capacity-cache-traced"
EXPECTED_ROUTE_SCHEDULE = (
    REQUEST_CACHE_UNTRACED,
    CAPACITY_CACHE_UNTRACED,
    CAPACITY_CACHE_TRACED,
    CAPACITY_CACHE_TRACED,
    CAPACITY_CACHE_UNTRACED,
    REQUEST_CACHE_UNTRACED,
)
CACHE_CAPACITY_MECHANISM = "cache-capacity"
PARKED_TRACE_MECHANISM = "parked-decode-trace"
NO_MATERIAL_REGRESSION = "no-material-same-process-regression"
UNEXPLAINED_MATERIAL_REGRESSION = "unexplained-material-regression"
SELECTED_MECHANISM = "selected-mechanism"
PHASE_PREFILL = "prefill-device"
PHASE_READBACK = "first-token-readback"
PHASE_TTFT = "ttft"
PHASES = (PHASE_PREFILL, PHASE_READBACK, PHASE_TTFT)
PROFILE_SCHEMA_VERSION = 1
REQUIRED_ROOT_COUNT = len(EXPECTED_ROUTE_SCHEDULE)
REVISION_LENGTH = 40
BLAKE3_LENGTH = 64
MESH_DIMENSIONS = 2
ROOTS_PER_ROUTE = 2
MINIMUM_RETAINED_SAMPLES = 5
MINIMUM_RATIO = 1.0


class TtftControlPolicyError(ValueError):
    """The TTFT control policy or evidence is invalid."""


@dataclass(frozen=True)
class TtftControlProfile:
    profile_id: str
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    mesh_shape: tuple[int, int]
    expected_cpu_affinity: tuple[int, ...]
    trace_region_size_bytes: int
    maximum_sequence_length: int
    kv_block_size: int
    request_cache_blocks: int
    trace_capacity_blocks: int
    page_table_blocks: int
    prompt: str
    generated_tokens: int
    route_schedule: tuple[str, ...]
    warmup_samples: int
    retained_samples: int
    maximum_within_root_spread_ratio: float
    maximum_duplicate_root_ratio: float
    material_regression_ratio: float
    phase_sum_tolerance_seconds: float

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> TtftControlProfile:
        root = _require_mapping(source, "profile")
        _require_exact_keys(
            root,
            {
                "schema_version",
                "profile_id",
                "diagnostic_only",
                "identities",
                "topology",
                "workload",
                "roots",
            },
            "profile",
        )
        schema_version = _require_positive_int(
            root.get("schema_version"), "schema_version"
        )
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise TtftControlPolicyError("TTFT control schema version is unsupported")
        if _require_bool(root.get("diagnostic_only"), "diagnostic_only") is not True:
            raise TtftControlPolicyError("TTFT control must remain diagnostic-only")

        identities = _require_mapping(root.get("identities"), "identities")
        topology = _require_mapping(root.get("topology"), "topology")
        workload = _require_mapping(root.get("workload"), "workload")
        roots = _require_mapping(root.get("roots"), "roots")
        _require_exact_keys(
            identities,
            {
                "authority_blake3",
                "checkpoint_revision",
                "model_source_revision",
                "tokenizer_revision",
            },
            "identities",
        )
        _require_exact_keys(
            topology,
            {
                "expected_pcie_device_ids",
                "mesh_rows",
                "mesh_columns",
                "minimum_cpu",
                "maximum_cpu",
                "trace_region_size_bytes",
                "maximum_sequence_length",
                "kv_block_size",
                "request_cache_blocks",
                "trace_capacity_blocks",
                "page_table_blocks",
            },
            "topology",
        )
        _require_exact_keys(workload, {"prompt", "generated_tokens"}, "workload")
        _require_exact_keys(
            roots,
            {
                "route_schedule",
                "warmup_samples",
                "retained_samples",
                "maximum_within_root_spread_ratio",
                "maximum_duplicate_root_ratio",
                "material_regression_ratio",
                "phase_sum_tolerance_seconds",
            },
            "roots",
        )

        minimum_cpu = _require_nonnegative_int(
            topology.get("minimum_cpu"), "topology.minimum_cpu"
        )
        maximum_cpu = _require_nonnegative_int(
            topology.get("maximum_cpu"), "topology.maximum_cpu"
        )
        if minimum_cpu > maximum_cpu:
            raise TtftControlPolicyError("CPU affinity bounds are reversed")

        route_schedule = _require_string_tuple(
            roots.get("route_schedule"), "roots.route_schedule"
        )
        if route_schedule != EXPECTED_ROUTE_SCHEDULE:
            raise TtftControlPolicyError("TTFT control route schedule changed")

        request_cache_blocks = _require_positive_int(
            topology.get("request_cache_blocks"),
            "topology.request_cache_blocks",
        )
        trace_capacity_blocks = _require_positive_int(
            topology.get("trace_capacity_blocks"),
            "topology.trace_capacity_blocks",
        )
        page_table_blocks = _require_positive_int(
            topology.get("page_table_blocks"), "topology.page_table_blocks"
        )
        if page_table_blocks != request_cache_blocks:
            raise TtftControlPolicyError(
                "request cache and bounded page table widths differ"
            )
        if trace_capacity_blocks <= request_cache_blocks:
            raise TtftControlPolicyError(
                "trace cache capacity must exceed request cache capacity"
            )

        maximum_within_root_spread_ratio = _require_positive_float(
            roots.get("maximum_within_root_spread_ratio"),
            "roots.maximum_within_root_spread_ratio",
        )
        maximum_duplicate_root_ratio = _require_positive_float(
            roots.get("maximum_duplicate_root_ratio"),
            "roots.maximum_duplicate_root_ratio",
        )
        material_regression_ratio = _require_positive_float(
            roots.get("material_regression_ratio"),
            "roots.material_regression_ratio",
        )
        for label, value in (
            (
                "roots.maximum_within_root_spread_ratio",
                maximum_within_root_spread_ratio,
            ),
            ("roots.maximum_duplicate_root_ratio", maximum_duplicate_root_ratio),
            ("roots.material_regression_ratio", material_regression_ratio),
        ):
            if value < MINIMUM_RATIO:
                raise TtftControlPolicyError(f"{label} is less than one")

        expected_pcie_device_ids = _require_int_tuple(
            topology.get("expected_pcie_device_ids"),
            "topology.expected_pcie_device_ids",
        )
        if not expected_pcie_device_ids or len(set(expected_pcie_device_ids)) != len(
            expected_pcie_device_ids
        ):
            raise TtftControlPolicyError(
                "physical device IDs must be non-empty and unique"
            )
        mesh_shape = (
            _require_positive_int(topology.get("mesh_rows"), "topology.mesh_rows"),
            _require_positive_int(
                topology.get("mesh_columns"), "topology.mesh_columns"
            ),
        )
        if len(mesh_shape) != MESH_DIMENSIONS:
            raise TtftControlPolicyError("TTFT control mesh rank changed")
        if math.prod(mesh_shape) != len(expected_pcie_device_ids):
            raise TtftControlPolicyError("mesh size and physical device count differ")

        retained_samples = _require_positive_int(
            roots.get("retained_samples"), "roots.retained_samples"
        )
        if retained_samples < MINIMUM_RETAINED_SAMPLES:
            raise TtftControlPolicyError(
                f"retained samples must be at least {MINIMUM_RETAINED_SAMPLES}"
            )
        maximum_sequence_length = _require_positive_int(
            topology.get("maximum_sequence_length"),
            "topology.maximum_sequence_length",
        )
        kv_block_size = _require_positive_int(
            topology.get("kv_block_size"), "topology.kv_block_size"
        )
        if trace_capacity_blocks * kv_block_size < maximum_sequence_length:
            raise TtftControlPolicyError("trace cache cannot hold the sequence limit")

        return cls(
            profile_id=_require_nonempty_string(root.get("profile_id"), "profile_id"),
            authority_blake3=_require_hex_string(
                identities.get("authority_blake3"),
                "identities.authority_blake3",
                BLAKE3_LENGTH,
            ),
            checkpoint_revision=_require_hex_string(
                identities.get("checkpoint_revision"),
                "identities.checkpoint_revision",
                REVISION_LENGTH,
            ),
            model_source_revision=_require_hex_string(
                identities.get("model_source_revision"),
                "identities.model_source_revision",
                REVISION_LENGTH,
            ),
            tokenizer_revision=_require_hex_string(
                identities.get("tokenizer_revision"),
                "identities.tokenizer_revision",
                REVISION_LENGTH,
            ),
            expected_pcie_device_ids=expected_pcie_device_ids,
            mesh_shape=mesh_shape,
            expected_cpu_affinity=tuple(range(minimum_cpu, maximum_cpu + 1)),
            trace_region_size_bytes=_require_positive_int(
                topology.get("trace_region_size_bytes"),
                "topology.trace_region_size_bytes",
            ),
            maximum_sequence_length=maximum_sequence_length,
            kv_block_size=kv_block_size,
            request_cache_blocks=request_cache_blocks,
            trace_capacity_blocks=trace_capacity_blocks,
            page_table_blocks=page_table_blocks,
            prompt=_require_nonempty_string(workload.get("prompt"), "workload.prompt"),
            generated_tokens=_require_positive_int(
                workload.get("generated_tokens"), "workload.generated_tokens"
            ),
            route_schedule=route_schedule,
            warmup_samples=_require_positive_int(
                roots.get("warmup_samples"), "roots.warmup_samples"
            ),
            retained_samples=retained_samples,
            maximum_within_root_spread_ratio=maximum_within_root_spread_ratio,
            maximum_duplicate_root_ratio=maximum_duplicate_root_ratio,
            material_regression_ratio=material_regression_ratio,
            phase_sum_tolerance_seconds=_require_positive_float(
                roots.get("phase_sum_tolerance_seconds"),
                "roots.phase_sum_tolerance_seconds",
            ),
        )


@dataclass(frozen=True)
class TtftControlIdentity:
    runtime_identity: str
    package_identity: str
    model_identity: str
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str


@dataclass(frozen=True)
class TtftControlSample:
    root_index: int
    sample_index: int
    route: str
    cache_blocks: int
    page_table_blocks: int
    trace_parked: bool
    first_token_id: int
    prefill_device_seconds: float
    first_token_readback_seconds: float
    ttft_seconds: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "root_index": self.root_index,
            "sample_index": self.sample_index,
            "route": self.route,
            "cache_blocks": self.cache_blocks,
            "page_table_blocks": self.page_table_blocks,
            "trace_parked": self.trace_parked,
            "first_token_id": self.first_token_id,
            "prefill_device_seconds": self.prefill_device_seconds,
            "first_token_readback_seconds": self.first_token_readback_seconds,
            "ttft_seconds": self.ttft_seconds,
        }


@dataclass(frozen=True)
class TtftControlEvidence:
    identity: TtftControlIdentity
    physical_pcie_device_ids: tuple[int, ...]
    mesh_shape: tuple[int, int]
    host_cpu_affinity: tuple[int, ...]
    samples: tuple[TtftControlSample, ...]
    cleanup_complete: bool


@dataclass(frozen=True)
class TtftControlRootSummary:
    root_index: int
    route: str
    sample_count: int
    first_token_id: int
    median_prefill_device_seconds: float
    median_first_token_readback_seconds: float
    median_ttft_seconds: float
    prefill_device_spread_ratio: float
    first_token_readback_spread_ratio: float
    ttft_spread_ratio: float

    def phase_median(self, phase: str) -> float:
        if phase == PHASE_PREFILL:
            return self.median_prefill_device_seconds
        if phase == PHASE_READBACK:
            return self.median_first_token_readback_seconds
        if phase == PHASE_TTFT:
            return self.median_ttft_seconds
        raise TtftControlPolicyError(f"unknown TTFT phase: {phase}")

    def phase_spread(self, phase: str) -> float:
        if phase == PHASE_PREFILL:
            return self.prefill_device_spread_ratio
        if phase == PHASE_READBACK:
            return self.first_token_readback_spread_ratio
        if phase == PHASE_TTFT:
            return self.ttft_spread_ratio
        raise TtftControlPolicyError(f"unknown TTFT phase: {phase}")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "root_index": self.root_index,
            "route": self.route,
            "sample_count": self.sample_count,
            "first_token_id": self.first_token_id,
            "median_prefill_device_seconds": self.median_prefill_device_seconds,
            "median_first_token_readback_seconds": self.median_first_token_readback_seconds,
            "median_ttft_seconds": self.median_ttft_seconds,
            "prefill_device_spread_ratio": self.prefill_device_spread_ratio,
            "first_token_readback_spread_ratio": self.first_token_readback_spread_ratio,
            "ttft_spread_ratio": self.ttft_spread_ratio,
        }


@dataclass(frozen=True)
class TtftControlDecision:
    control_valid: bool
    admission_eligible: bool
    disposition: str
    selected_mechanisms: tuple[str, ...]
    reasons: tuple[str, ...]
    root_summaries: tuple[TtftControlRootSummary, ...]
    duplicate_root_ratios: Mapping[str, Mapping[str, float]]
    cache_capacity_ratios: Mapping[str, float]
    parked_trace_ratios: Mapping[str, float]
    combined_ratios: Mapping[str, float]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "control_valid": self.control_valid,
            "admission_eligible": self.admission_eligible,
            "disposition": self.disposition,
            "selected_mechanisms": list(self.selected_mechanisms),
            "reasons": list(self.reasons),
            "root_summaries": [summary.to_mapping() for summary in self.root_summaries],
            "duplicate_root_ratios": {
                route: dict(phase_ratios)
                for route, phase_ratios in self.duplicate_root_ratios.items()
            },
            "cache_capacity_ratios": dict(self.cache_capacity_ratios),
            "parked_trace_ratios": dict(self.parked_trace_ratios),
            "combined_ratios": dict(self.combined_ratios),
        }


def evaluate_ttft_control(
    profile: TtftControlProfile,
    evidence: TtftControlEvidence,
) -> TtftControlDecision:
    """Validate matched roots and select only a stable material mechanism."""
    reasons: list[str] = []
    _validate_identity(profile, evidence, reasons)
    if evidence.physical_pcie_device_ids != profile.expected_pcie_device_ids:
        _append_once(reasons, "physical-device-identity")
    if evidence.mesh_shape != profile.mesh_shape:
        _append_once(reasons, "mesh-shape")
    if evidence.host_cpu_affinity != profile.expected_cpu_affinity:
        _append_once(reasons, "host-cpu-affinity")
    if not evidence.cleanup_complete:
        _append_once(reasons, "cleanup-incomplete")

    samples_by_root: dict[int, list[TtftControlSample]] = {
        root_index: [] for root_index in range(1, REQUIRED_ROOT_COUNT + 1)
    }
    observed_tokens: set[int] = set()
    for sample in evidence.samples:
        observed_tokens.add(sample.first_token_id)
        if sample.root_index not in samples_by_root:
            _append_once(reasons, "root-index")
            continue
        samples_by_root[sample.root_index].append(sample)
        expected_route = profile.route_schedule[sample.root_index - 1]
        if sample.route != expected_route:
            _append_once(reasons, f"root-{sample.root_index}-route")
        expected_trace_state = sample.route == CAPACITY_CACHE_TRACED
        if sample.trace_parked != expected_trace_state:
            _append_once(reasons, f"root-{sample.root_index}-trace-state")
        expected_cache_blocks = (
            profile.request_cache_blocks
            if sample.route == REQUEST_CACHE_UNTRACED
            else profile.trace_capacity_blocks
        )
        if sample.cache_blocks != expected_cache_blocks:
            _append_once(reasons, f"root-{sample.root_index}-cache-width")
        if sample.page_table_blocks != profile.page_table_blocks:
            _append_once(reasons, f"root-{sample.root_index}-page-table-width")
        if not _valid_duration(sample.prefill_device_seconds):
            _append_once(reasons, "prefill-device-duration")
        if not _valid_duration(sample.first_token_readback_seconds):
            _append_once(reasons, "first-token-readback-duration")
        if not _valid_duration(sample.ttft_seconds):
            _append_once(reasons, "ttft-duration")
        if all((
            _valid_duration(sample.prefill_device_seconds),
            _valid_duration(sample.first_token_readback_seconds),
            _valid_duration(sample.ttft_seconds),
        )) and not math.isclose(
            sample.prefill_device_seconds + sample.first_token_readback_seconds,
            sample.ttft_seconds,
            rel_tol=0.0,
            abs_tol=profile.phase_sum_tolerance_seconds,
        ):
            _append_once(reasons, "ttft-phase-sum")

    if len(observed_tokens) != 1:
        _append_once(reasons, "first-token-parity")

    root_summaries: list[TtftControlRootSummary] = []
    for root_index in range(1, REQUIRED_ROOT_COUNT + 1):
        root_samples = samples_by_root[root_index]
        if len(root_samples) != profile.retained_samples:
            _append_once(reasons, f"root-{root_index}-sample-count")
            continue
        expected_indices = tuple(range(1, profile.retained_samples + 1))
        actual_indices = tuple(sorted(sample.sample_index for sample in root_samples))
        if actual_indices != expected_indices:
            _append_once(reasons, f"root-{root_index}-sample-indices")
            continue
        if any(
            not _valid_duration(value)
            for sample in root_samples
            for value in (
                sample.prefill_device_seconds,
                sample.first_token_readback_seconds,
                sample.ttft_seconds,
            )
        ):
            continue
        summary = _summarize_root(root_index, root_samples)
        root_summaries.append(summary)
        for phase in PHASES:
            if summary.phase_spread(phase) > profile.maximum_within_root_spread_ratio:
                _append_once(reasons, f"root-{root_index}-{phase}-instability")

    duplicate_root_ratios = _duplicate_root_ratios(root_summaries)
    for route, phase_ratios in duplicate_root_ratios.items():
        for phase, ratio in phase_ratios.items():
            if ratio > profile.maximum_duplicate_root_ratio:
                _append_once(reasons, f"{route}-{phase}-duplicate-instability")

    ratios_available = len(root_summaries) == REQUIRED_ROOT_COUNT and all(
        sum(summary.route == route for summary in root_summaries) == ROOTS_PER_ROUTE
        for route in set(EXPECTED_ROUTE_SCHEDULE)
    )
    if len(root_summaries) == REQUIRED_ROOT_COUNT and not ratios_available:
        _append_once(reasons, "route-summary-count")
    cache_capacity_ratios: dict[str, float] = {}
    parked_trace_ratios: dict[str, float] = {}
    combined_ratios: dict[str, float] = {}
    if ratios_available:
        route_medians = _route_phase_medians(root_summaries)
        for phase in PHASES:
            request_value = route_medians[REQUEST_CACHE_UNTRACED][phase]
            capacity_value = route_medians[CAPACITY_CACHE_UNTRACED][phase]
            traced_value = route_medians[CAPACITY_CACHE_TRACED][phase]
            cache_capacity_ratios[phase] = capacity_value / request_value
            parked_trace_ratios[phase] = traced_value / capacity_value
            combined_ratios[phase] = traced_value / request_value

    control_valid = not reasons and ratios_available
    selected_mechanisms: list[str] = []
    disposition = UNEXPLAINED_MATERIAL_REGRESSION
    if control_valid:
        if cache_capacity_ratios[PHASE_TTFT] > profile.material_regression_ratio:
            selected_mechanisms.append(CACHE_CAPACITY_MECHANISM)
        if parked_trace_ratios[PHASE_TTFT] > profile.material_regression_ratio:
            selected_mechanisms.append(PARKED_TRACE_MECHANISM)
        if selected_mechanisms:
            disposition = SELECTED_MECHANISM
        elif combined_ratios[PHASE_TTFT] <= profile.material_regression_ratio:
            disposition = NO_MATERIAL_REGRESSION

    return TtftControlDecision(
        control_valid=control_valid,
        admission_eligible=False,
        disposition=disposition,
        selected_mechanisms=tuple(selected_mechanisms) if control_valid else (),
        reasons=tuple(reasons),
        root_summaries=tuple(root_summaries),
        duplicate_root_ratios=duplicate_root_ratios,
        cache_capacity_ratios=cache_capacity_ratios,
        parked_trace_ratios=parked_trace_ratios,
        combined_ratios=combined_ratios,
    )


def _validate_identity(
    profile: TtftControlProfile,
    evidence: TtftControlEvidence,
    reasons: list[str],
) -> None:
    identity = evidence.identity
    for label, actual, expected in (
        ("authority-identity", identity.authority_blake3, profile.authority_blake3),
        (
            "checkpoint-identity",
            identity.checkpoint_revision,
            profile.checkpoint_revision,
        ),
        (
            "model-source-identity",
            identity.model_source_revision,
            profile.model_source_revision,
        ),
        (
            "tokenizer-identity",
            identity.tokenizer_revision,
            profile.tokenizer_revision,
        ),
    ):
        if actual != expected:
            _append_once(reasons, label)
    for label, value in (
        ("runtime-identity", identity.runtime_identity),
        ("package-identity", identity.package_identity),
        ("model-identity", identity.model_identity),
    ):
        if not value:
            _append_once(reasons, label)


def _summarize_root(
    root_index: int,
    samples: Sequence[TtftControlSample],
) -> TtftControlRootSummary:
    prefill = [sample.prefill_device_seconds for sample in samples]
    readback = [sample.first_token_readback_seconds for sample in samples]
    ttft = [sample.ttft_seconds for sample in samples]
    return TtftControlRootSummary(
        root_index=root_index,
        route=samples[0].route,
        sample_count=len(samples),
        first_token_id=samples[0].first_token_id,
        median_prefill_device_seconds=statistics.median(prefill),
        median_first_token_readback_seconds=statistics.median(readback),
        median_ttft_seconds=statistics.median(ttft),
        prefill_device_spread_ratio=_spread_ratio(prefill),
        first_token_readback_spread_ratio=_spread_ratio(readback),
        ttft_spread_ratio=_spread_ratio(ttft),
    )


def _duplicate_root_ratios(
    summaries: Sequence[TtftControlRootSummary],
) -> dict[str, dict[str, float]]:
    by_route: dict[str, list[TtftControlRootSummary]] = {
        route: [] for route in set(EXPECTED_ROUTE_SCHEDULE)
    }
    for summary in summaries:
        by_route.setdefault(summary.route, []).append(summary)
    result: dict[str, dict[str, float]] = {}
    for route in sorted(by_route):
        route_summaries = by_route[route]
        if len(route_summaries) != ROOTS_PER_ROUTE:
            continue
        result[route] = {
            phase: _spread_ratio([
                summary.phase_median(phase) for summary in route_summaries
            ])
            for phase in PHASES
        }
    return result


def _route_phase_medians(
    summaries: Sequence[TtftControlRootSummary],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for route in set(EXPECTED_ROUTE_SCHEDULE):
        route_summaries = [summary for summary in summaries if summary.route == route]
        if len(route_summaries) != ROOTS_PER_ROUTE:
            raise TtftControlPolicyError(
                f"route {route} lacks {ROOTS_PER_ROUTE} summaries"
            )
        result[route] = {
            phase: statistics.median(
                summary.phase_median(phase) for summary in route_summaries
            )
            for phase in PHASES
        }
    return result


def _spread_ratio(values: Sequence[float]) -> float:
    if not values or any(not _valid_duration(value) for value in values):
        raise TtftControlPolicyError("spread values must be finite and positive")
    return max(values) / min(values)


def _valid_duration(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _require_exact_keys(
    source: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(source)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TtftControlPolicyError(
            f"{label} keys changed: missing={missing}, extra={extra}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TtftControlPolicyError(f"{label} must be a mapping")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TtftControlPolicyError(f"{label} must be a boolean")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TtftControlPolicyError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TtftControlPolicyError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TtftControlPolicyError(f"{label} must be a non-negative integer")
    return value


def _require_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TtftControlPolicyError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TtftControlPolicyError(f"{label} must be finite and positive")
    return result


def _require_hex_string(value: Any, label: str, length: int) -> str:
    result = _require_nonempty_string(value, label)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise TtftControlPolicyError(f"{label} must be lowercase hexadecimal")
    return result


def _require_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TtftControlPolicyError(f"{label} must be an integer array")
    return tuple(_require_nonnegative_int(item, label) for item in value)


def _require_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TtftControlPolicyError(f"{label} must be a string array")
    result = tuple(_require_nonempty_string(item, label) for item in value)
    if len(result) != REQUIRED_ROOT_COUNT:
        raise TtftControlPolicyError("TTFT control route count changed")
    return result
