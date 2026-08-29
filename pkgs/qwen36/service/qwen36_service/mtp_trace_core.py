"""Pure policy and matched evidence evaluation for traced MTP drafting."""

# r[impl tenstorrent.qwen38_p150x2.mtp_draft_trace]

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .mtp_drafter_core import (
    BLAKE3_HEX_LENGTH,
    EXPECTED_MINIMUM_PROMPT_COUNT,
    EXPECTED_PCIE_DEVICE_IDS,
    FIRST_PROMPT_INDEX,
    MAXIMUM_DRAFT_TOKEN_COUNT,
    MINIMUM_DRAFT_TOKEN_COUNT,
    MINIMUM_RATIO,
    REVISION_HEX_LENGTH,
    MtpDrafterIdentity,
    MtpDrafterPolicyError,
    accepted_prefix_length,
)

PROFILE_SCHEMA_VERSION = 1
EAGER_DRAFT = "eager"
TRACED_DRAFT = "traced"
ROUTE_BLOCK = (EAGER_DRAFT, TRACED_DRAFT, TRACED_DRAFT, EAGER_DRAFT)
EXPECTED_ROUTE_BLOCK_COUNT = 3
EXPECTED_ROUTE_SCHEDULE = ROUTE_BLOCK * EXPECTED_ROUTE_BLOCK_COUNT
EXPECTED_ROOTS_PER_PROMPT = len(EXPECTED_ROUTE_SCHEDULE)
EXPECTED_ROOTS_PER_ROUTE = EXPECTED_ROOTS_PER_PROMPT // 2
MINIMUM_SAMPLES_PER_ROUTE = 5
FIRST_ROOT_INDEX = 1
ZERO_DURATION_SECONDS = 0.0


class MtpTracePolicyError(ValueError):
    """The MTP trace input does not satisfy its closed policy."""


@dataclass(frozen=True)
class MtpTraceProfile:
    profile_id: str
    diagnostic_only: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    draft_token_count: int
    minimum_prompt_count: int
    route_block_count: int
    minimum_samples_per_route: int
    route_schedule: tuple[str, ...]
    maximum_duplicate_root_ratio: float
    maximum_traced_draft_ratio: float
    minimum_emitted_token_speedup: float
    maximum_trace_capture_seconds: float
    prompts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> MtpTraceProfile:
        """Parse one closed exported Nickel MTP trace profile."""
        _require_exact_keys(
            source,
            {
                "schema_version",
                "profile_id",
                "diagnostic_only",
                "identities",
                "expected_pcie_device_ids",
                "draft_token_count",
                "minimum_prompt_count",
                "route_block_count",
                "minimum_samples_per_route",
                "route_schedule",
                "maximum_duplicate_root_ratio",
                "maximum_traced_draft_ratio",
                "minimum_emitted_token_speedup",
                "maximum_trace_capture_seconds",
                "prompts",
            },
            "MTP trace profile",
        )
        if source["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise MtpTracePolicyError("MTP trace schema version is unsupported")
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
            profile_id=_require_string(source["profile_id"], "profile_id"),
            diagnostic_only=_require_bool(source["diagnostic_only"], "diagnostic_only"),
            authority_blake3=_require_lower_hex(
                identities["authority_blake3"],
                "authority_blake3",
                BLAKE3_HEX_LENGTH,
            ),
            checkpoint_revision=_require_lower_hex(
                identities["checkpoint_revision"],
                "checkpoint_revision",
                REVISION_HEX_LENGTH,
            ),
            model_source_revision=_require_lower_hex(
                identities["model_source_revision"],
                "model_source_revision",
                REVISION_HEX_LENGTH,
            ),
            tokenizer_revision=_require_lower_hex(
                identities["tokenizer_revision"],
                "tokenizer_revision",
                REVISION_HEX_LENGTH,
            ),
            expected_pcie_device_ids=_require_int_tuple(
                source["expected_pcie_device_ids"],
                "expected_pcie_device_ids",
            ),
            draft_token_count=_require_positive_int(
                source["draft_token_count"], "draft_token_count"
            ),
            minimum_prompt_count=_require_positive_int(
                source["minimum_prompt_count"], "minimum_prompt_count"
            ),
            route_block_count=_require_positive_int(
                source["route_block_count"], "route_block_count"
            ),
            minimum_samples_per_route=_require_positive_int(
                source["minimum_samples_per_route"],
                "minimum_samples_per_route",
            ),
            route_schedule=_require_string_tuple(
                source["route_schedule"], "route_schedule", unique=False
            ),
            maximum_duplicate_root_ratio=_require_positive_float(
                source["maximum_duplicate_root_ratio"],
                "maximum_duplicate_root_ratio",
            ),
            maximum_traced_draft_ratio=_require_positive_float(
                source["maximum_traced_draft_ratio"],
                "maximum_traced_draft_ratio",
            ),
            minimum_emitted_token_speedup=_require_positive_float(
                source["minimum_emitted_token_speedup"],
                "minimum_emitted_token_speedup",
            ),
            maximum_trace_capture_seconds=_require_positive_float(
                source["maximum_trace_capture_seconds"],
                "maximum_trace_capture_seconds",
            ),
            prompts=_require_string_tuple(source["prompts"], "prompts", unique=True),
        )
        if not profile.diagnostic_only:
            raise MtpTracePolicyError("MTP trace control must be diagnostic-only")
        if profile.expected_pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
            raise MtpTracePolicyError("MTP trace control requires devices 0 and 1")
        if not (
            MINIMUM_DRAFT_TOKEN_COUNT
            <= profile.draft_token_count
            <= MAXIMUM_DRAFT_TOKEN_COUNT
        ):
            raise MtpTracePolicyError(
                "MTP trace draft count must lie between "
                f"{MINIMUM_DRAFT_TOKEN_COUNT} and {MAXIMUM_DRAFT_TOKEN_COUNT}"
            )
        if profile.minimum_prompt_count != EXPECTED_MINIMUM_PROMPT_COUNT:
            raise MtpTracePolicyError("MTP trace control requires five prompts")
        if len(profile.prompts) != profile.minimum_prompt_count:
            raise MtpTracePolicyError("MTP trace prompt count differs from policy")
        if profile.route_block_count != EXPECTED_ROUTE_BLOCK_COUNT:
            raise MtpTracePolicyError("MTP trace route block count changed")
        if profile.minimum_samples_per_route != EXPECTED_ROOTS_PER_ROUTE:
            raise MtpTracePolicyError("MTP trace retained sample count changed")
        if profile.minimum_samples_per_route < MINIMUM_SAMPLES_PER_ROUTE:
            raise MtpTracePolicyError("MTP trace requires at least five route samples")
        if profile.route_schedule != EXPECTED_ROUTE_SCHEDULE:
            raise MtpTracePolicyError("MTP trace route schedule changed")
        if profile.maximum_duplicate_root_ratio < MINIMUM_RATIO:
            raise MtpTracePolicyError("duplicate-root ratio cannot be less than one")
        if (
            not ZERO_DURATION_SECONDS
            < profile.maximum_traced_draft_ratio
            < MINIMUM_RATIO
        ):
            raise MtpTracePolicyError("traced draft ratio must be between zero and one")
        if profile.minimum_emitted_token_speedup < MINIMUM_RATIO:
            raise MtpTracePolicyError("speedup bound cannot be less than one")
        return profile


@dataclass(frozen=True)
class MtpTraceBinding:
    token_buffer_identity: str
    position_buffer_identity: str
    rope_buffer_identity: str
    feedback_buffer_identity: str
    kv_buffer_identity: str
    feedback_output_identity: str
    index_output_identity: str
    value_output_identity: str
    feedback_output_shape: tuple[int, ...]
    feedback_output_dtype: str
    index_output_shape: tuple[int, ...]
    index_output_dtype: str
    value_output_shape: tuple[int, ...]
    value_output_dtype: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "token_buffer_identity": self.token_buffer_identity,
            "position_buffer_identity": self.position_buffer_identity,
            "rope_buffer_identity": self.rope_buffer_identity,
            "feedback_buffer_identity": self.feedback_buffer_identity,
            "kv_buffer_identity": self.kv_buffer_identity,
            "feedback_output_identity": self.feedback_output_identity,
            "index_output_identity": self.index_output_identity,
            "value_output_identity": self.value_output_identity,
            "feedback_output_shape": list(self.feedback_output_shape),
            "feedback_output_dtype": self.feedback_output_dtype,
            "index_output_shape": list(self.index_output_shape),
            "index_output_dtype": self.index_output_dtype,
            "value_output_shape": list(self.value_output_shape),
            "value_output_dtype": self.value_output_dtype,
        }


def validate_mtp_trace_binding(
    expected: MtpTraceBinding, observed: MtpTraceBinding
) -> tuple[str, ...]:
    """Return every changed trace-bound field in deterministic order."""
    fields = (
        "token_buffer_identity",
        "position_buffer_identity",
        "rope_buffer_identity",
        "feedback_buffer_identity",
        "kv_buffer_identity",
        "feedback_output_identity",
        "index_output_identity",
        "value_output_identity",
        "feedback_output_shape",
        "feedback_output_dtype",
        "index_output_shape",
        "index_output_dtype",
        "value_output_shape",
        "value_output_dtype",
    )
    return tuple(
        field
        for field in fields
        if getattr(expected, field) != getattr(observed, field)
    )


@dataclass(frozen=True)
class MtpTraceSample:
    prompt_index: int
    root_index: int
    route: str
    identity: MtpDrafterIdentity
    physical_pcie_device_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    mtp_prefill_seconds: float
    draft_decode_seconds: float
    draft_selection_seconds: float
    trace_capture_seconds: float
    target_sequential_seconds: float
    target_verification_seconds: float
    target_verification_readback_seconds: float
    rollback_seconds: float
    mtp_cache_recovery_seconds: float
    trace_binding_valid: bool
    capture_idempotence_valid: bool
    mtp_kv_prefix_valid: bool
    rollback_valid: bool
    target_output_valid: bool

    @property
    def total_draft_seconds(self) -> float:
        return self.mtp_prefill_seconds + self.draft_decode_seconds


@dataclass(frozen=True)
class MtpTraceDecision:
    trace_valid: bool
    traced_drafting_selected: bool
    admission_eligible: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    traced_draft_ratio: float
    total_draft_ratio: float
    aggregate_emitted_token_speedup: float
    mean_accepted_drafts: float
    eager_draft_median_seconds: float
    traced_draft_median_seconds: float
    trace_capture_median_seconds: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "trace_valid": self.trace_valid,
            "traced_drafting_selected": self.traced_drafting_selected,
            "admission_eligible": self.admission_eligible,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "traced_draft_ratio": self.traced_draft_ratio,
            "total_draft_ratio": self.total_draft_ratio,
            "aggregate_emitted_token_speedup": self.aggregate_emitted_token_speedup,
            "mean_accepted_drafts": self.mean_accepted_drafts,
            "eager_draft_median_seconds": self.eager_draft_median_seconds,
            "traced_draft_median_seconds": self.traced_draft_median_seconds,
            "trace_capture_median_seconds": self.trace_capture_median_seconds,
        }


def evaluate_mtp_trace(
    profile: MtpTraceProfile,
    expected_identity: MtpDrafterIdentity,
    samples: Sequence[MtpTraceSample],
) -> MtpTraceDecision:
    """Evaluate matched trace bindings, parity, stability, and economics."""
    reasons: list[str] = []
    ordered = sorted(
        samples, key=lambda sample: (sample.prompt_index, sample.root_index)
    )
    expected_sample_count = profile.minimum_prompt_count * EXPECTED_ROOTS_PER_PROMPT
    if len(ordered) != expected_sample_count:
        _append_once(reasons, "sample-count")

    by_prompt: dict[int, list[MtpTraceSample]] = {}
    for sample in ordered:
        by_prompt.setdefault(sample.prompt_index, []).append(sample)
        label = f"prompt-{sample.prompt_index}-root-{sample.root_index}"
        if sample.identity != expected_identity:
            _append_once(reasons, f"{label}-identity")
        if sample.physical_pcie_device_ids != profile.expected_pcie_device_ids:
            _append_once(reasons, f"{label}-physical-devices")
        if sample.route not in (EAGER_DRAFT, TRACED_DRAFT):
            _append_once(reasons, f"{label}-route")
        try:
            computed_acceptance = accepted_prefix_length(
                sample.draft_token_ids, sample.target_token_ids
            )
        except MtpDrafterPolicyError:
            _append_once(reasons, f"{label}-token-shape")
            continue
        if sample.accepted_prefix_length != computed_acceptance:
            _append_once(reasons, f"{label}-acceptance-alignment")
        if not sample.rollback_valid:
            _append_once(reasons, f"{label}-target-rollback")
        if not sample.target_output_valid:
            _append_once(reasons, f"{label}-target-output")
        if not sample.mtp_kv_prefix_valid:
            _append_once(reasons, f"{label}-mtp-kv-prefix")
        if sample.route == TRACED_DRAFT:
            if not sample.trace_binding_valid:
                _append_once(reasons, f"{label}-trace-binding")
            if not sample.capture_idempotence_valid:
                _append_once(reasons, f"{label}-capture-idempotence")
            if (
                not math.isfinite(sample.trace_capture_seconds)
                or sample.trace_capture_seconds <= ZERO_DURATION_SECONDS
                or sample.trace_capture_seconds > profile.maximum_trace_capture_seconds
            ):
                _append_once(reasons, f"{label}-trace-capture")
        elif sample.trace_capture_seconds != ZERO_DURATION_SECONDS:
            _append_once(reasons, f"{label}-eager-trace-capture")
        if not _sample_timing_valid(sample):
            _append_once(reasons, f"{label}-duration")

    expected_prompts = tuple(
        range(FIRST_PROMPT_INDEX, FIRST_PROMPT_INDEX + profile.minimum_prompt_count)
    )
    if tuple(sorted(by_prompt)) != expected_prompts:
        _append_once(reasons, "prompt-order")

    eager_draft_medians: list[float] = []
    traced_draft_medians: list[float] = []
    eager_total_medians: list[float] = []
    traced_total_medians: list[float] = []
    trace_capture_medians: list[float] = []
    baseline_seconds: list[float] = []
    speculative_seconds: list[float] = []
    accepted_counts: list[int] = []

    for prompt_index in expected_prompts:
        roots = sorted(
            by_prompt.get(prompt_index, []), key=lambda sample: sample.root_index
        )
        if len(roots) != EXPECTED_ROOTS_PER_PROMPT:
            _append_once(reasons, f"prompt-{prompt_index}-root-count")
            continue
        expected_root_indices = tuple(
            range(FIRST_ROOT_INDEX, FIRST_ROOT_INDEX + EXPECTED_ROOTS_PER_PROMPT)
        )
        if tuple(sample.root_index for sample in roots) != expected_root_indices:
            _append_once(reasons, f"prompt-{prompt_index}-root-order")
        if tuple(sample.route for sample in roots) != profile.route_schedule:
            _append_once(reasons, f"prompt-{prompt_index}-route-order")
        if any(not _sample_timing_valid(sample) for sample in roots):
            continue

        reference = roots[0]
        for sample in roots[1:]:
            if (
                sample.draft_token_ids != reference.draft_token_ids
                or sample.target_token_ids != reference.target_token_ids
                or sample.accepted_prefix_length != reference.accepted_prefix_length
            ):
                _append_once(reasons, f"prompt-{prompt_index}-route-parity")

        eager_roots = [sample for sample in roots if sample.route == EAGER_DRAFT]
        traced_roots = [sample for sample in roots if sample.route == TRACED_DRAFT]
        if (
            len(eager_roots) != profile.minimum_samples_per_route
            or len(traced_roots) != profile.minimum_samples_per_route
        ):
            _append_once(reasons, f"prompt-{prompt_index}-route-count")
            continue
        for route, route_roots in (
            (EAGER_DRAFT, eager_roots),
            (TRACED_DRAFT, traced_roots),
        ):
            timing_groups = (
                [sample.mtp_prefill_seconds for sample in route_roots],
                [sample.draft_decode_seconds for sample in route_roots],
                [sample.draft_selection_seconds for sample in route_roots],
                [sample.total_draft_seconds for sample in route_roots],
                [sample.target_verification_seconds for sample in route_roots],
                [sample.target_verification_readback_seconds for sample in route_roots],
                [sample.target_sequential_seconds for sample in route_roots],
            )
            if any(
                _spread_ratio(values) > profile.maximum_duplicate_root_ratio
                for values in timing_groups
            ):
                _append_once(reasons, f"prompt-{prompt_index}-{route}-instability")

        eager_draft_medians.append(
            statistics.median(sample.draft_decode_seconds for sample in eager_roots)
        )
        traced_draft_medians.append(
            statistics.median(sample.draft_decode_seconds for sample in traced_roots)
        )
        eager_total_medians.append(
            statistics.median(sample.total_draft_seconds for sample in eager_roots)
        )
        traced_total_medians.append(
            statistics.median(sample.total_draft_seconds for sample in traced_roots)
        )
        trace_capture_medians.append(
            statistics.median(sample.trace_capture_seconds for sample in traced_roots)
        )

        accepted = reference.accepted_prefix_length
        emitted_tokens = accepted + 1
        baseline_seconds.append(
            statistics.median(
                sample.target_sequential_seconds for sample in traced_roots
            )
            / (profile.draft_token_count + 1)
            * emitted_tokens
        )
        speculative_seconds.append(
            statistics.median(sample.total_draft_seconds for sample in traced_roots)
            + statistics.median(
                sample.target_verification_seconds for sample in traced_roots
            )
            + statistics.median(
                sample.target_verification_readback_seconds for sample in traced_roots
            )
            + statistics.median(sample.rollback_seconds for sample in traced_roots)
            + statistics.median(
                sample.mtp_cache_recovery_seconds for sample in traced_roots
            )
        )
        accepted_counts.append(accepted)

    if eager_draft_medians and len(eager_draft_medians) == profile.minimum_prompt_count:
        traced_draft_ratio = sum(traced_draft_medians) / sum(eager_draft_medians)
        total_draft_ratio = sum(traced_total_medians) / sum(eager_total_medians)
        aggregate_speedup = sum(baseline_seconds) / sum(speculative_seconds)
        mean_accepted = statistics.fmean(accepted_counts)
        eager_draft_median = statistics.median(eager_draft_medians)
        traced_draft_median = statistics.median(traced_draft_medians)
        trace_capture_median = statistics.median(trace_capture_medians)
        if traced_draft_ratio > profile.maximum_traced_draft_ratio:
            _append_once(reasons, "traced-draft-gain")
        if aggregate_speedup <= profile.minimum_emitted_token_speedup:
            _append_once(reasons, "emitted-token-economics")
    else:
        traced_draft_ratio = ZERO_DURATION_SECONDS
        total_draft_ratio = ZERO_DURATION_SECONDS
        aggregate_speedup = ZERO_DURATION_SECONDS
        mean_accepted = ZERO_DURATION_SECONDS
        eager_draft_median = ZERO_DURATION_SECONDS
        traced_draft_median = ZERO_DURATION_SECONDS
        trace_capture_median = ZERO_DURATION_SECONDS

    trace_valid = not reasons
    blockers = ["production-admission-missing", "traced-decode-admission-missing"]
    if not trace_valid:
        blockers.insert(0, "mtp-draft-trace-invalid")
    return MtpTraceDecision(
        trace_valid=trace_valid,
        traced_drafting_selected=trace_valid,
        admission_eligible=False,
        reasons=tuple(reasons),
        blockers=tuple(blockers),
        traced_draft_ratio=traced_draft_ratio,
        total_draft_ratio=total_draft_ratio,
        aggregate_emitted_token_speedup=aggregate_speedup,
        mean_accepted_drafts=mean_accepted,
        eager_draft_median_seconds=eager_draft_median,
        traced_draft_median_seconds=traced_draft_median,
        trace_capture_median_seconds=trace_capture_median,
    )


def _sample_timing_valid(sample: MtpTraceSample) -> bool:
    required = (
        sample.mtp_prefill_seconds,
        sample.draft_decode_seconds,
        sample.draft_selection_seconds,
        sample.target_sequential_seconds,
        sample.target_verification_seconds,
        sample.target_verification_readback_seconds,
    )
    optional = (
        sample.rollback_seconds,
        sample.mtp_cache_recovery_seconds,
    )
    return (
        all(
            math.isfinite(value) and value > ZERO_DURATION_SECONDS for value in required
        )
        and all(
            math.isfinite(value) and value >= ZERO_DURATION_SECONDS
            for value in optional
        )
        and sample.draft_selection_seconds <= sample.total_draft_seconds
    )


def _spread_ratio(values: Sequence[float]) -> float:
    if not values or any(
        not math.isfinite(value) or value <= ZERO_DURATION_SECONDS for value in values
    ):
        return math.inf
    return max(values) / min(values)


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MtpTracePolicyError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise MtpTracePolicyError(
            f"{name} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MtpTracePolicyError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MtpTracePolicyError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MtpTracePolicyError(f"{name} must be a positive integer")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpTracePolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= ZERO_DURATION_SECONDS:
        raise MtpTracePolicyError(f"{name} must be finite and positive")
    return result


def _require_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MtpTracePolicyError(f"{name} must be a non-empty array")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise MtpTracePolicyError(f"{name} contains an invalid integer")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise MtpTracePolicyError(f"{name} must be sorted and unique")
    return result


def _require_string_tuple(value: Any, name: str, *, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MtpTracePolicyError(f"{name} must be a non-empty array")
    result = tuple(_require_string(item, name) for item in value)
    if unique and len(set(result)) != len(result):
        raise MtpTracePolicyError(f"{name} must be unique")
    return result


def _require_lower_hex(value: Any, name: str, length: int) -> str:
    result = _require_string(value, name)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise MtpTracePolicyError(
            f"{name} must be a {length}-character lowercase hex value"
        )
    return result
