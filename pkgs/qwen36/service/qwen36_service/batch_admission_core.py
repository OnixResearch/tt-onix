"""Pure policy for bounded Qwen P150x2 batch admission."""

# r[impl tenstorrent.qwen38_p150x2.batch_performance_admission]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROFILE_SCHEMA_VERSION = 1
BASELINE_WIDTH = 1
CANDIDATE_WIDTH = 4
B8_CANDIDATE_WIDTH = 8
ALLOWED_CANDIDATE_WIDTHS = (CANDIDATE_WIDTH, B8_CANDIDATE_WIDTH)
BLAKE3_LENGTH = 64
REVISION_LENGTH = 40
NOT_MEASURED = "not-measured"
ADMISSION_MEASUREMENT_MODE = "batch-admission"
PRODUCTION_MEASUREMENT_MODE = "physical-batched-service"
EXPECTED_EXECUTION_ORDER = ("single", "batch", "batch", "single")


class BatchAdmissionPolicyError(ValueError):
    """The batch-admission policy is malformed or used outside its scope."""


@dataclass(frozen=True)
class BatchAdmissionProfile:
    """Closed batch admission policy projected from Nickel."""

    profile_id: str
    physically_admitted: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    receipt_blake3: str
    tokenizer_revision: str
    baseline_width: int
    candidate_width: int
    execution_order: tuple[str, ...]
    generated_tokens: int
    measured_runs: int
    prompts: tuple[str, ...]
    warmup_runs: int
    maximum_tail_latency_ratio: float
    maximum_ttft_ratio: float
    minimum_aggregate_decode_speedup_ratio: float
    minimum_per_user_decode_retention_ratio: float

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> "BatchAdmissionProfile":
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
            raise BatchAdmissionPolicyError("schema_version is unsupported")
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
                "baseline_width",
                "candidate_width",
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
                "maximum_tail_latency_ratio",
                "maximum_ttft_ratio",
                "minimum_aggregate_decode_speedup_ratio",
                "minimum_per_user_decode_retention_ratio",
                "require_active_prefix",
                "require_clean_reset",
                "require_distinct_state_slices",
                "require_exact_token_parity",
                "require_physical_evidence",
            },
            "admission",
        )
        for field in (
            "require_active_prefix",
            "require_clean_reset",
            "require_distinct_state_slices",
            "require_exact_token_parity",
            "require_physical_evidence",
        ):
            if _require_bool(admission[field], f"admission.{field}") is not True:
                raise BatchAdmissionPolicyError(f"admission.{field} must be true")
        baseline_width = _require_int(
            workload["baseline_width"], "workload.baseline_width"
        )
        candidate_width = _require_int(
            workload["candidate_width"], "workload.candidate_width"
        )
        if baseline_width != BASELINE_WIDTH:
            raise BatchAdmissionPolicyError("workload widths are unsupported")
        if candidate_width not in ALLOWED_CANDIDATE_WIDTHS:
            raise BatchAdmissionPolicyError("workload widths are unsupported")
        execution_order = _require_string_tuple(
            workload["execution_order"], "workload.execution_order"
        )
        if execution_order != EXPECTED_EXECUTION_ORDER:
            raise BatchAdmissionPolicyError("workload execution order is unsupported")
        prompts = _require_string_tuple(workload["prompts"], "workload.prompts")
        physically_admitted = _require_bool(
            root["physically_admitted"], "physically_admitted"
        )
        receipt_blake3 = _require_receipt(
            identities["receipt_blake3"], "identities.receipt_blake3"
        )
        if physically_admitted and receipt_blake3 == NOT_MEASURED:
            raise BatchAdmissionPolicyError(
                "physically admitted profile has no measured receipt"
            )
        profile = cls(
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
            baseline_width=baseline_width,
            candidate_width=candidate_width,
            execution_order=execution_order,
            generated_tokens=_require_positive_int(
                workload["generated_tokens"], "workload.generated_tokens"
            ),
            measured_runs=_require_positive_int(
                workload["measured_runs"], "workload.measured_runs"
            ),
            prompts=prompts,
            warmup_runs=_require_positive_int(
                workload["warmup_runs"], "workload.warmup_runs"
            ),
            maximum_tail_latency_ratio=_require_at_least_one(
                admission["maximum_tail_latency_ratio"],
                "admission.maximum_tail_latency_ratio",
            ),
            maximum_ttft_ratio=_require_at_least_one(
                admission["maximum_ttft_ratio"],
                "admission.maximum_ttft_ratio",
            ),
            minimum_aggregate_decode_speedup_ratio=_require_above_one(
                admission["minimum_aggregate_decode_speedup_ratio"],
                "admission.minimum_aggregate_decode_speedup_ratio",
            ),
            minimum_per_user_decode_retention_ratio=_require_fraction(
                admission["minimum_per_user_decode_retention_ratio"],
                "admission.minimum_per_user_decode_retention_ratio",
            ),
        )
        return profile

    def measurement(self, *, enabled: bool, width: int) -> tuple[str, bool]:
        """Select evidence mode without reading ambient state."""
        if not enabled:
            return PRODUCTION_MEASUREMENT_MODE, False
        if isinstance(width, bool) or not isinstance(width, int):
            raise BatchAdmissionPolicyError("batch admission width must be an integer")
        if width not in (self.baseline_width, self.candidate_width):
            raise BatchAdmissionPolicyError(
                "batch admission width is outside the profile"
            )
        return ADMISSION_MEASUREMENT_MODE, True


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BatchAdmissionPolicyError(f"{field} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise BatchAdmissionPolicyError(
            f"{field} fields differ: expected {sorted(expected)}, got {sorted(observed)}"
        )


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise BatchAdmissionPolicyError(f"{field} must be a boolean")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BatchAdmissionPolicyError(f"{field} must be an integer")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    result = _require_int(value, field)
    if result <= 0:
        raise BatchAdmissionPolicyError(f"{field} must be positive")
    return result


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchAdmissionPolicyError(f"{field} must be a number")
    result = float(value)
    if result <= 0 or result == float("inf") or result != result:
        raise BatchAdmissionPolicyError(f"{field} must be finite and positive")
    return result


def _require_at_least_one(value: Any, field: str) -> float:
    result = _require_number(value, field)
    if result < 1:
        raise BatchAdmissionPolicyError(f"{field} must be at least one")
    return result


def _require_above_one(value: Any, field: str) -> float:
    result = _require_number(value, field)
    if result <= 1:
        raise BatchAdmissionPolicyError(f"{field} must exceed one")
    return result


def _require_fraction(value: Any, field: str) -> float:
    result = _require_number(value, field)
    if result > 1:
        raise BatchAdmissionPolicyError(f"{field} must not exceed one")
    return result


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BatchAdmissionPolicyError(f"{field} must be a non-empty string")
    return value


def _require_string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BatchAdmissionPolicyError(f"{field} must be a non-empty array")
    result = tuple(_require_string(item, f"{field} item") for item in value)
    return result


def _require_hex(value: Any, field: str, length: int) -> str:
    result = _require_string(value, field)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise BatchAdmissionPolicyError(f"{field} must be lowercase hexadecimal")
    return result


def _require_receipt(value: Any, field: str) -> str:
    if value == NOT_MEASURED:
        return NOT_MEASURED
    return _require_hex(value, field, BLAKE3_LENGTH)
