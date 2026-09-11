"""Pure policy for the diagnostic Qwen3.8 MTP target probe."""

# r[impl tenstorrent.qwen38_p150x2.mtp_verification_probe]

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROFILE_SCHEMA_VERSION = 2
EXPECTED_PCIE_DEVICE_IDS = (0, 1)
EXPECTED_VERIFICATION_WIDTHS = (1, 2, 4)
EXPECTED_EXECUTION_PATH = "traced-hybrid-packed-decode"
EXPECTED_NATIVE_WIDTH_LIMIT = 4
EXPECTED_MINIMUM_SAMPLES_PER_WIDTH = 5
MINIMUM_STABILITY_RATIO = 1.0
MINIMUM_AMORTIZATION_RATIO = 1.0
MAXIMUM_PCC = 1.0
MINIMUM_TOKEN_ID = 0
FIRST_SAMPLE_INDEX = 1
REVISION_HEX_LENGTH = 40
BLAKE3_HEX_LENGTH = 64


class MtpProbePolicyError(ValueError):
    """The target-probe input does not satisfy its closed policy."""


@dataclass(frozen=True)
class MtpProbeIdentity:
    runtime_identity: str
    package_identity: str
    model_identity: str


@dataclass(frozen=True)
class MtpProbeProfile:
    profile_id: str
    diagnostic_only: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    verification_widths: tuple[int, ...]
    execution_path: str
    native_width_limit: int
    minimum_samples_per_width: int
    minimum_state_pcc: float
    maximum_state_nmse: float
    minimum_logits_pcc: float
    maximum_logits_nmse: float
    maximum_sample_spread_ratio: float
    minimum_end_to_end_amortization_ratio: float
    prompt: str

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> MtpProbeProfile:
        """Parse one deterministic Nickel profile with closed-record checks."""
        _require_exact_keys(
            source,
            {
                "schema_version",
                "profile_id",
                "diagnostic_only",
                "identities",
                "expected_pcie_device_ids",
                "verification_widths",
                "execution_path",
                "native_width_limit",
                "minimum_samples_per_width",
                "minimum_state_pcc",
                "maximum_state_nmse",
                "minimum_logits_pcc",
                "maximum_logits_nmse",
                "maximum_sample_spread_ratio",
                "minimum_end_to_end_amortization_ratio",
                "prompt",
            },
            "MTP probe profile",
        )
        if source["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise MtpProbePolicyError(
                "MTP probe profile schema version is not supported"
            )
        identities = _require_mapping(source["identities"], "identities")
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
        profile = cls(
            profile_id=_require_non_empty_string(source["profile_id"], "profile_id"),
            diagnostic_only=_require_bool(source["diagnostic_only"], "diagnostic_only"),
            authority_blake3=_require_lower_hex(
                identities["authority_blake3"],
                "authority_blake3",
                BLAKE3_HEX_LENGTH,
            ),
            checkpoint_revision=_require_revision(
                identities["checkpoint_revision"], "checkpoint_revision"
            ),
            model_source_revision=_require_revision(
                identities["model_source_revision"], "model_source_revision"
            ),
            tokenizer_revision=_require_revision(
                identities["tokenizer_revision"], "tokenizer_revision"
            ),
            expected_pcie_device_ids=_require_int_tuple(
                source["expected_pcie_device_ids"],
                "expected_pcie_device_ids",
                allow_zero=True,
            ),
            verification_widths=_require_int_tuple(
                source["verification_widths"], "verification_widths", allow_zero=False
            ),
            execution_path=_require_non_empty_string(
                source["execution_path"], "execution_path"
            ),
            native_width_limit=_require_positive_int(
                source["native_width_limit"], "native_width_limit"
            ),
            minimum_samples_per_width=_require_positive_int(
                source["minimum_samples_per_width"], "minimum_samples_per_width"
            ),
            minimum_state_pcc=_require_unit_float(
                source["minimum_state_pcc"], "minimum_state_pcc"
            ),
            maximum_state_nmse=_require_non_negative_float(
                source["maximum_state_nmse"], "maximum_state_nmse"
            ),
            minimum_logits_pcc=_require_unit_float(
                source["minimum_logits_pcc"], "minimum_logits_pcc"
            ),
            maximum_logits_nmse=_require_non_negative_float(
                source["maximum_logits_nmse"], "maximum_logits_nmse"
            ),
            maximum_sample_spread_ratio=_require_positive_float(
                source["maximum_sample_spread_ratio"], "maximum_sample_spread_ratio"
            ),
            minimum_end_to_end_amortization_ratio=_require_positive_float(
                source["minimum_end_to_end_amortization_ratio"],
                "minimum_end_to_end_amortization_ratio",
            ),
            prompt=_require_non_empty_string(source["prompt"], "prompt"),
        )
        if not profile.diagnostic_only:
            raise MtpProbePolicyError("MTP target probing must remain diagnostic-only")
        if profile.expected_pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
            raise MtpProbePolicyError(
                "MTP target probing requires physical devices 0 and 1"
            )
        if profile.verification_widths != EXPECTED_VERIFICATION_WIDTHS:
            raise MtpProbePolicyError("MTP target probing requires widths 1, 2, and 4")
        if profile.execution_path != EXPECTED_EXECUTION_PATH:
            raise MtpProbePolicyError(
                "MTP target probing requires the traced hybrid packed-decode path"
            )
        if profile.native_width_limit != EXPECTED_NATIVE_WIDTH_LIMIT:
            raise MtpProbePolicyError("native verifier width limit must be four")
        if max(profile.verification_widths) > profile.native_width_limit:
            raise MtpProbePolicyError("verification width exceeds the native limit")
        if profile.minimum_samples_per_width != EXPECTED_MINIMUM_SAMPLES_PER_WIDTH:
            raise MtpProbePolicyError(
                "MTP target probing requires five samples per width"
            )
        if profile.maximum_sample_spread_ratio < MINIMUM_STABILITY_RATIO:
            raise MtpProbePolicyError("sample spread ratio cannot be less than one")
        if profile.minimum_end_to_end_amortization_ratio < MINIMUM_AMORTIZATION_RATIO:
            raise MtpProbePolicyError(
                "end-to-end amortization ratio cannot be less than one"
            )
        return profile


@dataclass(frozen=True)
class MtpProbeSample:
    width: int
    sample_index: int
    identity: MtpProbeIdentity
    physical_pcie_device_ids: tuple[int, ...]
    sequential_token_ids: tuple[int, ...]
    verification_token_ids: tuple[int, ...]
    sequential_seconds: float
    verification_trace_capture_seconds: float
    verification_device_seconds: float
    verification_readback_seconds: float
    logits_pcc: float
    logits_nmse: float
    recurrent_state_pcc: float
    recurrent_state_nmse: float
    convolution_state_pcc: float
    convolution_state_nmse: float
    paged_kv_state_pcc: float
    paged_kv_state_nmse: float


@dataclass(frozen=True)
class MtpWidthSummary:
    width: int
    sample_count: int
    median_sequential_seconds: float
    median_verification_trace_capture_seconds: float
    median_verification_device_seconds: float
    median_verification_readback_seconds: float
    median_verification_end_to_end_seconds: float
    device_amortization_ratio: float
    end_to_end_amortization_ratio: float
    sequential_spread_ratio: float
    verification_device_spread_ratio: float
    verification_end_to_end_spread_ratio: float
    minimum_logits_pcc: float
    maximum_logits_nmse: float
    minimum_recurrent_state_pcc: float
    maximum_recurrent_state_nmse: float
    minimum_convolution_state_pcc: float
    maximum_convolution_state_nmse: float
    minimum_paged_kv_state_pcc: float
    maximum_paged_kv_state_nmse: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "sample_count": self.sample_count,
            "median_sequential_seconds": self.median_sequential_seconds,
            "median_verification_trace_capture_seconds": self.median_verification_trace_capture_seconds,
            "median_verification_device_seconds": self.median_verification_device_seconds,
            "median_verification_readback_seconds": self.median_verification_readback_seconds,
            "median_verification_end_to_end_seconds": self.median_verification_end_to_end_seconds,
            "device_amortization_ratio": self.device_amortization_ratio,
            "end_to_end_amortization_ratio": self.end_to_end_amortization_ratio,
            "sequential_spread_ratio": self.sequential_spread_ratio,
            "verification_device_spread_ratio": self.verification_device_spread_ratio,
            "verification_end_to_end_spread_ratio": self.verification_end_to_end_spread_ratio,
            "minimum_logits_pcc": self.minimum_logits_pcc,
            "maximum_logits_nmse": self.maximum_logits_nmse,
            "minimum_recurrent_state_pcc": self.minimum_recurrent_state_pcc,
            "maximum_recurrent_state_nmse": self.maximum_recurrent_state_nmse,
            "minimum_convolution_state_pcc": self.minimum_convolution_state_pcc,
            "maximum_convolution_state_nmse": self.maximum_convolution_state_nmse,
            "minimum_paged_kv_state_pcc": self.minimum_paged_kv_state_pcc,
            "maximum_paged_kv_state_nmse": self.maximum_paged_kv_state_nmse,
        }


@dataclass(frozen=True)
class MtpProbeDecision:
    target_verification_valid: bool
    mtp_selected: bool
    admission_eligible: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    width_summaries: tuple[MtpWidthSummary, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "target_verification_valid": self.target_verification_valid,
            "mtp_selected": self.mtp_selected,
            "admission_eligible": self.admission_eligible,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "widths": [summary.to_mapping() for summary in self.width_summaries],
        }


def evaluate_mtp_probe(
    profile: MtpProbeProfile,
    expected_identity: MtpProbeIdentity,
    samples: Sequence[MtpProbeSample],
) -> MtpProbeDecision:
    """Validate matched observations and summarize target-only amortization."""
    reasons: list[str] = []
    grouped: dict[int, list[MtpProbeSample]] = {
        width: [] for width in profile.verification_widths
    }
    for sample in samples:
        if sample.width not in grouped:
            _append_once(reasons, "unexpected-verification-width")
            continue
        grouped[sample.width].append(sample)

    summaries: list[MtpWidthSummary] = []
    for width in profile.verification_widths:
        width_samples = sorted(grouped[width], key=lambda item: item.sample_index)
        width_label = f"width-{width}"
        if len(width_samples) < profile.minimum_samples_per_width:
            _append_once(reasons, f"{width_label}-sample-count")
        expected_indices = tuple(
            range(FIRST_SAMPLE_INDEX, FIRST_SAMPLE_INDEX + len(width_samples))
        )
        observed_indices = tuple(sample.sample_index for sample in width_samples)
        if observed_indices != expected_indices:
            _append_once(reasons, f"{width_label}-sample-order")

        valid_durations = True
        for sample in width_samples:
            if sample.identity != expected_identity:
                _append_once(reasons, f"{width_label}-identity-mismatch")
            if sample.physical_pcie_device_ids != profile.expected_pcie_device_ids:
                _append_once(reasons, f"{width_label}-physical-device-mismatch")
            if not _valid_tokens(sample.sequential_token_ids, width):
                _append_once(reasons, f"{width_label}-sequential-token-shape")
            if not _valid_tokens(sample.verification_token_ids, width):
                _append_once(reasons, f"{width_label}-verification-token-shape")
            if sample.sequential_token_ids != sample.verification_token_ids:
                _append_once(reasons, f"{width_label}-token-parity")
            durations = (
                sample.sequential_seconds,
                sample.verification_trace_capture_seconds,
                sample.verification_device_seconds,
                sample.verification_readback_seconds,
            )
            if any(not math.isfinite(value) or value <= 0 for value in durations):
                valid_durations = False
                _append_once(reasons, f"{width_label}-invalid-duration")
            state_values = (
                sample.logits_pcc,
                sample.logits_nmse,
                sample.recurrent_state_pcc,
                sample.recurrent_state_nmse,
                sample.convolution_state_pcc,
                sample.convolution_state_nmse,
                sample.paged_kv_state_pcc,
                sample.paged_kv_state_nmse,
            )
            if any(not math.isfinite(value) for value in state_values):
                _append_once(reasons, f"{width_label}-non-finite-state-metric")
                continue
            if not 0 <= sample.logits_pcc <= MAXIMUM_PCC:
                _append_once(reasons, f"{width_label}-invalid-logits-pcc")
            elif sample.logits_pcc < profile.minimum_logits_pcc:
                _append_once(reasons, f"{width_label}-logits-pcc")
            if sample.logits_nmse < 0:
                _append_once(reasons, f"{width_label}-invalid-logits-nmse")
            elif sample.logits_nmse > profile.maximum_logits_nmse:
                _append_once(reasons, f"{width_label}-logits-nmse")
            if not 0 <= sample.recurrent_state_pcc <= MAXIMUM_PCC:
                _append_once(reasons, f"{width_label}-invalid-recurrent-state-pcc")
            elif sample.recurrent_state_pcc < profile.minimum_state_pcc:
                _append_once(reasons, f"{width_label}-recurrent-state-pcc")
            if sample.recurrent_state_nmse < 0:
                _append_once(reasons, f"{width_label}-invalid-recurrent-state-nmse")
            elif sample.recurrent_state_nmse > profile.maximum_state_nmse:
                _append_once(reasons, f"{width_label}-recurrent-state-nmse")
            if not 0 <= sample.convolution_state_pcc <= MAXIMUM_PCC:
                _append_once(reasons, f"{width_label}-invalid-convolution-state-pcc")
            elif sample.convolution_state_pcc < profile.minimum_state_pcc:
                _append_once(reasons, f"{width_label}-convolution-state-pcc")
            if sample.convolution_state_nmse < 0:
                _append_once(reasons, f"{width_label}-invalid-convolution-state-nmse")
            elif sample.convolution_state_nmse > profile.maximum_state_nmse:
                _append_once(reasons, f"{width_label}-convolution-state-nmse")
            if not 0 <= sample.paged_kv_state_pcc <= MAXIMUM_PCC:
                _append_once(reasons, f"{width_label}-invalid-paged-kv-state-pcc")
            elif sample.paged_kv_state_pcc < profile.minimum_state_pcc:
                _append_once(reasons, f"{width_label}-paged-kv-state-pcc")
            if sample.paged_kv_state_nmse < 0:
                _append_once(reasons, f"{width_label}-invalid-paged-kv-state-nmse")
            elif sample.paged_kv_state_nmse > profile.maximum_state_nmse:
                _append_once(reasons, f"{width_label}-paged-kv-state-nmse")

        if width_samples and valid_durations:
            summary = _summarize_width(width, width_samples)
            summaries.append(summary)
            # Full-logit host readback is diagnostic. A future MTP loop keeps
            # target logits on device for acceptance, so stability gates the
            # traced replay and its matched sequential baseline.
            spread_values = (
                summary.sequential_spread_ratio,
                summary.verification_device_spread_ratio,
            )
            if any(
                value > profile.maximum_sample_spread_ratio for value in spread_values
            ):
                _append_once(reasons, f"{width_label}-sample-instability")
            if (
                width == profile.native_width_limit
                and summary.end_to_end_amortization_ratio
                <= profile.minimum_end_to_end_amortization_ratio
            ):
                _append_once(reasons, f"{width_label}-no-amortization")

    target_verification_valid = not reasons
    blockers = [
        "mtp-drafter-not-measured",
        "local-acceptance-not-measured",
        "rejection-rollback-not-implemented",
        "end-to-end-admission-missing",
    ]
    if not target_verification_valid:
        blockers.insert(0, "target-verification-invalid")
    return MtpProbeDecision(
        target_verification_valid=target_verification_valid,
        mtp_selected=False,
        admission_eligible=False,
        reasons=tuple(reasons),
        blockers=tuple(blockers),
        width_summaries=tuple(summaries),
    )


def _summarize_width(width: int, samples: Sequence[MtpProbeSample]) -> MtpWidthSummary:
    sequential = [sample.sequential_seconds for sample in samples]
    trace_capture = [sample.verification_trace_capture_seconds for sample in samples]
    device = [sample.verification_device_seconds for sample in samples]
    readback = [sample.verification_readback_seconds for sample in samples]
    end_to_end = [
        sample.verification_device_seconds + sample.verification_readback_seconds
        for sample in samples
    ]
    median_sequential = statistics.median(sequential)
    median_device = statistics.median(device)
    median_readback = statistics.median(readback)
    median_end_to_end = statistics.median(end_to_end)
    return MtpWidthSummary(
        width=width,
        sample_count=len(samples),
        median_sequential_seconds=median_sequential,
        median_verification_trace_capture_seconds=statistics.median(trace_capture),
        median_verification_device_seconds=median_device,
        median_verification_readback_seconds=median_readback,
        median_verification_end_to_end_seconds=median_end_to_end,
        device_amortization_ratio=median_sequential / median_device,
        end_to_end_amortization_ratio=median_sequential / median_end_to_end,
        sequential_spread_ratio=_spread_ratio(sequential),
        verification_device_spread_ratio=_spread_ratio(device),
        verification_end_to_end_spread_ratio=_spread_ratio(end_to_end),
        minimum_logits_pcc=min(sample.logits_pcc for sample in samples),
        maximum_logits_nmse=max(sample.logits_nmse for sample in samples),
        minimum_recurrent_state_pcc=min(
            sample.recurrent_state_pcc for sample in samples
        ),
        maximum_recurrent_state_nmse=max(
            sample.recurrent_state_nmse for sample in samples
        ),
        minimum_convolution_state_pcc=min(
            sample.convolution_state_pcc for sample in samples
        ),
        maximum_convolution_state_nmse=max(
            sample.convolution_state_nmse for sample in samples
        ),
        minimum_paged_kv_state_pcc=min(sample.paged_kv_state_pcc for sample in samples),
        maximum_paged_kv_state_nmse=max(
            sample.paged_kv_state_nmse for sample in samples
        ),
    )


def _spread_ratio(values: Sequence[float]) -> float:
    return max(values) / min(values)


def _valid_tokens(tokens: tuple[int, ...], width: int) -> bool:
    return len(tokens) == width and all(
        not isinstance(token, bool)
        and isinstance(token, int)
        and token >= MINIMUM_TOKEN_ID
        for token in tokens
    )


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MtpProbePolicyError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise MtpProbePolicyError(
            f"{name} keys differ: missing={missing}, extra={extra}"
        )


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MtpProbePolicyError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MtpProbePolicyError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MtpProbePolicyError(f"{name} must be a positive integer")
    return value


def _require_int_tuple(value: Any, name: str, *, allow_zero: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MtpProbePolicyError(f"{name} must be a non-empty array")
    minimum = 0 if allow_zero else 1
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < minimum
        for item in value
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise MtpProbePolicyError(f"{name} must contain {qualifier} integers")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise MtpProbePolicyError(f"{name} must be sorted and unique")
    return result


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpProbePolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise MtpProbePolicyError(f"{name} must be finite and positive")
    return result


def _require_non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpProbePolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MtpProbePolicyError(f"{name} must be finite and non-negative")
    return result


def _require_unit_float(value: Any, name: str) -> float:
    result = _require_non_negative_float(value, name)
    if result > MAXIMUM_PCC:
        raise MtpProbePolicyError(f"{name} must not exceed one")
    return result


def _require_revision(value: Any, name: str) -> str:
    return _require_lower_hex(value, name, REVISION_HEX_LENGTH)


def _require_lower_hex(value: Any, name: str, length: int) -> str:
    result = _require_non_empty_string(value, name)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise MtpProbePolicyError(
            f"{name} must be a {length}-character lowercase hex value"
        )
    return result
