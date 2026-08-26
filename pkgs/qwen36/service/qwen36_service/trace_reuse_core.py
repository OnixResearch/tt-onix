"""Pure activation policy for Qwen P150x2 same-width trace reuse."""

# r[impl tenstorrent.qwen38_p150x2.same_width_trace_reuse]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROFILE_SCHEMA_VERSION = 1
TRACE_REUSE_WIDTH = 4
MAXIMUM_RESIDENT_TRACE_COUNT = 1
BLAKE3_LENGTH = 64
REVISION_LENGTH = 40
NOT_MEASURED = "not-measured"
TRACE_REUSE_MEASUREMENT_MODE = "same-width-trace-reuse"
PRODUCTION_MEASUREMENT_MODE = "physical-batched-service"
EXPECTED_EXECUTION_ORDER = ("recapture", "reuse", "reuse", "recapture")


class TraceReusePolicyError(ValueError):
    """The trace-reuse profile is malformed or used outside its scope."""


@dataclass(frozen=True)
class TraceReuseActivation:
    """One explicit activation decision for served evidence and adapter policy."""

    enabled: bool
    admission_eligible: bool
    measurement_mode: str


@dataclass(frozen=True)
class TraceReuseProfile:
    """Closed trace-reuse policy projected from Nickel."""

    profile_id: str
    physically_admitted: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    receipt_blake3: str
    tokenizer_revision: str
    width: int
    execution_order: tuple[str, ...]
    generated_tokens: int
    measured_runs: int
    prompts: tuple[str, ...]
    warmup_runs: int
    maximum_duplicate_root_ratio: float
    maximum_resident_trace_count: int
    maximum_reused_trace_prepare_seconds: float
    maximum_tail_latency_ratio: float
    maximum_ttft_ratio: float
    minimum_decode_rate_retention_ratio: float
    minimum_request_wall_speedup_ratio: float

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> "TraceReuseProfile":
        _require_keys(
            root,
            {
                "schema_version",
                "profile_id",
                "physically_admitted",
                "identities",
                "workload",
                "admission",
            },
            "profile",
        )
        if (
            _require_int(root["schema_version"], "schema_version")
            != PROFILE_SCHEMA_VERSION
        ):
            raise TraceReusePolicyError("schema_version is unsupported")
        identities = _require_mapping(root["identities"], "identities")
        workload = _require_mapping(root["workload"], "workload")
        admission = _require_mapping(root["admission"], "admission")
        _require_keys(
            identities,
            {
                "authority_blake3",
                "checkpoint_revision",
                "model_source_revision",
                "receipt_blake3",
                "tokenizer_revision",
            },
            "identities",
        )
        _require_keys(
            workload,
            {
                "width",
                "execution_order",
                "generated_tokens",
                "measured_runs",
                "prompts",
                "warmup_runs",
            },
            "workload",
        )
        _require_keys(
            admission,
            {
                "maximum_duplicate_root_ratio",
                "maximum_resident_trace_count",
                "maximum_reused_trace_prepare_seconds",
                "maximum_tail_latency_ratio",
                "maximum_ttft_ratio",
                "minimum_decode_rate_retention_ratio",
                "minimum_request_wall_speedup_ratio",
                "require_clean_reset",
                "require_exact_token_parity",
                "require_physical_evidence",
                "require_stable_binding",
                "require_zero_measured_recaptures",
            },
            "admission",
        )
        for field in (
            "require_clean_reset",
            "require_exact_token_parity",
            "require_physical_evidence",
            "require_stable_binding",
            "require_zero_measured_recaptures",
        ):
            if _require_bool(admission[field], f"admission.{field}") is not True:
                raise TraceReusePolicyError(f"admission.{field} must be true")
        width = _require_int(workload["width"], "workload.width")
        if width != TRACE_REUSE_WIDTH:
            raise TraceReusePolicyError("workload width is unsupported")
        execution_order = _require_string_tuple(
            workload["execution_order"], "workload.execution_order"
        )
        if execution_order != EXPECTED_EXECUTION_ORDER:
            raise TraceReusePolicyError("workload execution order is unsupported")
        physically_admitted = _require_bool(
            root["physically_admitted"], "physically_admitted"
        )
        receipt_blake3 = _require_receipt(
            identities["receipt_blake3"], "identities.receipt_blake3"
        )
        if physically_admitted and receipt_blake3 == NOT_MEASURED:
            raise TraceReusePolicyError(
                "physically admitted profile has no measured receipt"
            )
        resident_count = _require_positive_int(
            admission["maximum_resident_trace_count"],
            "admission.maximum_resident_trace_count",
        )
        if resident_count != MAXIMUM_RESIDENT_TRACE_COUNT:
            raise TraceReusePolicyError(
                "admission.maximum_resident_trace_count must be one"
            )
        return cls(
            profile_id=_require_string(root["profile_id"], "profile_id"),
            physically_admitted=physically_admitted,
            authority_blake3=_require_hex(
                identities["authority_blake3"],
                "identities.authority_blake3",
                BLAKE3_LENGTH,
            ),
            checkpoint_revision=_require_hex(
                identities["checkpoint_revision"],
                "identities.checkpoint_revision",
                REVISION_LENGTH,
            ),
            model_source_revision=_require_hex(
                identities["model_source_revision"],
                "identities.model_source_revision",
                REVISION_LENGTH,
            ),
            receipt_blake3=receipt_blake3,
            tokenizer_revision=_require_hex(
                identities["tokenizer_revision"],
                "identities.tokenizer_revision",
                REVISION_LENGTH,
            ),
            width=width,
            execution_order=execution_order,
            generated_tokens=_require_positive_int(
                workload["generated_tokens"], "workload.generated_tokens"
            ),
            measured_runs=_require_positive_int(
                workload["measured_runs"], "workload.measured_runs"
            ),
            prompts=_require_string_tuple(workload["prompts"], "workload.prompts"),
            warmup_runs=_require_positive_int(
                workload["warmup_runs"], "workload.warmup_runs"
            ),
            maximum_duplicate_root_ratio=_require_at_least_one(
                admission["maximum_duplicate_root_ratio"],
                "admission.maximum_duplicate_root_ratio",
            ),
            maximum_resident_trace_count=resident_count,
            maximum_reused_trace_prepare_seconds=_require_positive_number(
                admission["maximum_reused_trace_prepare_seconds"],
                "admission.maximum_reused_trace_prepare_seconds",
            ),
            maximum_tail_latency_ratio=_require_at_least_one(
                admission["maximum_tail_latency_ratio"],
                "admission.maximum_tail_latency_ratio",
            ),
            maximum_ttft_ratio=_require_at_least_one(
                admission["maximum_ttft_ratio"],
                "admission.maximum_ttft_ratio",
            ),
            minimum_decode_rate_retention_ratio=_require_fraction(
                admission["minimum_decode_rate_retention_ratio"],
                "admission.minimum_decode_rate_retention_ratio",
            ),
            minimum_request_wall_speedup_ratio=_require_above_one(
                admission["minimum_request_wall_speedup_ratio"],
                "admission.minimum_request_wall_speedup_ratio",
            ),
        )

    def activation(
        self, *, explicit_measurement: bool, width: int
    ) -> TraceReuseActivation:
        """Select reuse without reading ambient state or changing admission meaning."""
        if not isinstance(explicit_measurement, bool):
            raise TraceReusePolicyError("explicit measurement must be a boolean")
        if isinstance(width, bool) or not isinstance(width, int):
            raise TraceReusePolicyError("trace-reuse width must be an integer")
        if explicit_measurement and width != self.width:
            raise TraceReusePolicyError("trace-reuse width is outside the profile")
        return TraceReuseActivation(
            enabled=(
                explicit_measurement
                or (self.physically_admitted and width == self.width)
            ),
            admission_eligible=explicit_measurement,
            measurement_mode=(
                TRACE_REUSE_MEASUREMENT_MODE
                if explicit_measurement
                else PRODUCTION_MEASUREMENT_MODE
            ),
        )


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceReusePolicyError(f"{field} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise TraceReusePolicyError(
            f"{field} fields differ: expected {sorted(expected)}, got {sorted(observed)}"
        )


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TraceReusePolicyError(f"{field} must be a boolean")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceReusePolicyError(f"{field} must be an integer")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    result = _require_int(value, field)
    if result <= 0:
        raise TraceReusePolicyError(f"{field} must be positive")
    return result


def _require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceReusePolicyError(f"{field} must be a number")
    result = float(value)
    if result <= 0 or not math_is_finite(result):
        raise TraceReusePolicyError(f"{field} must be finite and positive")
    return result


def _require_at_least_one(value: Any, field: str) -> float:
    result = _require_positive_number(value, field)
    if result < 1:
        raise TraceReusePolicyError(f"{field} must be at least one")
    return result


def _require_above_one(value: Any, field: str) -> float:
    result = _require_positive_number(value, field)
    if result <= 1:
        raise TraceReusePolicyError(f"{field} must exceed one")
    return result


def _require_fraction(value: Any, field: str) -> float:
    result = _require_positive_number(value, field)
    if result > 1:
        raise TraceReusePolicyError(f"{field} must not exceed one")
    return result


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceReusePolicyError(f"{field} must be a non-empty string")
    return value


def _require_string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TraceReusePolicyError(f"{field} must be a non-empty array")
    return tuple(_require_string(item, f"{field} item") for item in value)


def _require_hex(value: Any, field: str, length: int) -> str:
    result = _require_string(value, field)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise TraceReusePolicyError(f"{field} must be lowercase hexadecimal")
    return result


def _require_receipt(value: Any, field: str) -> str:
    result = _require_string(value, field)
    if result == NOT_MEASURED:
        return result
    return _require_hex(result, field, BLAKE3_LENGTH)


def math_is_finite(value: float) -> bool:
    """Avoid an ambient numerical dependency for one finite-value check."""
    return value == value and value not in (float("inf"), float("-inf"))
