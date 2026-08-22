"""Pure policy and paired evidence evaluation for MTP draft selection."""

# r[impl tenstorrent.qwen38_p150x2.mtp_draft_selection]

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .mtp_drafter_core import (
    BLAKE3_HEX_LENGTH,
    EXPECTED_DRAFT_TOKEN_COUNT,
    EXPECTED_MINIMUM_PROMPT_COUNT,
    EXPECTED_PCIE_DEVICE_IDS,
    FIRST_PROMPT_INDEX,
    MINIMUM_RATIO,
    REVISION_HEX_LENGTH,
    MtpDrafterIdentity,
    MtpDrafterPolicyError,
    accepted_prefix_length,
)

PROFILE_SCHEMA_VERSION = 1
FULL_LOGIT_READBACK = "full-logit-readback"
SHARD_LOCAL_TOP1 = "shard-local-top1"
EXPECTED_ROUTE_SCHEDULE = (
    FULL_LOGIT_READBACK,
    SHARD_LOCAL_TOP1,
    SHARD_LOCAL_TOP1,
    FULL_LOGIT_READBACK,
)
EXPECTED_ROOTS_PER_PROMPT = len(EXPECTED_ROUTE_SCHEDULE)
EXPECTED_ROOTS_PER_ROUTE = 2
FIRST_ROOT_INDEX = 1


class MtpSelectionPolicyError(ValueError):
    """The MTP selector input does not satisfy its closed policy."""


@dataclass(frozen=True)
class MtpSelectionProfile:
    profile_id: str
    diagnostic_only: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    draft_token_count: int
    minimum_prompt_count: int
    route_schedule: tuple[str, ...]
    maximum_duplicate_root_ratio: float
    maximum_selector_ratio: float
    minimum_emitted_token_speedup: float
    prompts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> MtpSelectionProfile:
        """Parse one closed exported Nickel selector profile."""
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
                "route_schedule",
                "maximum_duplicate_root_ratio",
                "maximum_selector_ratio",
                "minimum_emitted_token_speedup",
                "prompts",
            },
            "MTP selection profile",
        )
        if source["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise MtpSelectionPolicyError("MTP selection schema version is unsupported")
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
            route_schedule=_require_string_tuple(
                source["route_schedule"], "route_schedule", unique=False
            ),
            maximum_duplicate_root_ratio=_require_positive_float(
                source["maximum_duplicate_root_ratio"],
                "maximum_duplicate_root_ratio",
            ),
            maximum_selector_ratio=_require_positive_float(
                source["maximum_selector_ratio"], "maximum_selector_ratio"
            ),
            minimum_emitted_token_speedup=_require_positive_float(
                source["minimum_emitted_token_speedup"],
                "minimum_emitted_token_speedup",
            ),
            prompts=_require_string_tuple(source["prompts"], "prompts", unique=True),
        )
        if not profile.diagnostic_only:
            raise MtpSelectionPolicyError("MTP selection must be diagnostic-only")
        if profile.expected_pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
            raise MtpSelectionPolicyError("MTP selection requires devices 0 and 1")
        if profile.draft_token_count != EXPECTED_DRAFT_TOKEN_COUNT:
            raise MtpSelectionPolicyError("MTP selection requires three drafts")
        if profile.minimum_prompt_count != EXPECTED_MINIMUM_PROMPT_COUNT:
            raise MtpSelectionPolicyError("MTP selection requires five prompts")
        if len(profile.prompts) != profile.minimum_prompt_count:
            raise MtpSelectionPolicyError(
                "MTP selection prompt count differs from policy"
            )
        if profile.route_schedule != EXPECTED_ROUTE_SCHEDULE:
            raise MtpSelectionPolicyError("MTP selection route schedule changed")
        if profile.maximum_duplicate_root_ratio < MINIMUM_RATIO:
            raise MtpSelectionPolicyError(
                "duplicate-root ratio cannot be less than one"
            )
        if not 0.0 < profile.maximum_selector_ratio < MINIMUM_RATIO:
            raise MtpSelectionPolicyError("selector ratio must be between zero and one")
        if profile.minimum_emitted_token_speedup < MINIMUM_RATIO:
            raise MtpSelectionPolicyError("speedup bound cannot be less than one")
        return profile


@dataclass(frozen=True)
class MtpSelectionSample:
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
    target_sequential_seconds: float
    target_verification_seconds: float
    target_verification_readback_seconds: float
    rollback_seconds: float
    mtp_cache_reprefill_seconds: float
    rollback_valid: bool
    target_output_valid: bool

    @property
    def total_draft_seconds(self) -> float:
        return self.mtp_prefill_seconds + self.draft_decode_seconds


@dataclass(frozen=True)
class MtpSelectionDecision:
    selector_valid: bool
    bounded_selector_selected: bool
    admission_eligible: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    selector_ratio: float
    total_draft_ratio: float
    aggregate_emitted_token_speedup: float
    mean_accepted_drafts: float
    full_selector_median_seconds: float
    bounded_selector_median_seconds: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "selector_valid": self.selector_valid,
            "bounded_selector_selected": self.bounded_selector_selected,
            "admission_eligible": self.admission_eligible,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "selector_ratio": self.selector_ratio,
            "total_draft_ratio": self.total_draft_ratio,
            "aggregate_emitted_token_speedup": self.aggregate_emitted_token_speedup,
            "mean_accepted_drafts": self.mean_accepted_drafts,
            "full_selector_median_seconds": self.full_selector_median_seconds,
            "bounded_selector_median_seconds": self.bounded_selector_median_seconds,
        }


def evaluate_mtp_selection(
    profile: MtpSelectionProfile,
    expected_identity: MtpDrafterIdentity,
    samples: Sequence[MtpSelectionSample],
) -> MtpSelectionDecision:
    """Evaluate paired selector causality, parity, stability, and economics."""
    reasons: list[str] = []
    ordered = sorted(
        samples, key=lambda sample: (sample.prompt_index, sample.root_index)
    )
    expected_sample_count = profile.minimum_prompt_count * EXPECTED_ROOTS_PER_PROMPT
    if len(ordered) != expected_sample_count:
        _append_once(reasons, "sample-count")

    by_prompt: dict[int, list[MtpSelectionSample]] = {}
    for sample in ordered:
        by_prompt.setdefault(sample.prompt_index, []).append(sample)
        label = f"prompt-{sample.prompt_index}-root-{sample.root_index}"
        if sample.identity != expected_identity:
            _append_once(reasons, f"{label}-identity")
        if sample.physical_pcie_device_ids != profile.expected_pcie_device_ids:
            _append_once(reasons, f"{label}-physical-devices")
        if sample.route not in (FULL_LOGIT_READBACK, SHARD_LOCAL_TOP1):
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
            _append_once(reasons, f"{label}-rollback")
        if not sample.target_output_valid:
            _append_once(reasons, f"{label}-target-output")
        required_durations = (
            sample.mtp_prefill_seconds,
            sample.draft_decode_seconds,
            sample.draft_selection_seconds,
            sample.target_sequential_seconds,
            sample.target_verification_seconds,
            sample.target_verification_readback_seconds,
        )
        optional_durations = (
            sample.rollback_seconds,
            sample.mtp_cache_reprefill_seconds,
        )
        if any(
            not math.isfinite(value) or value <= 0.0 for value in required_durations
        ):
            _append_once(reasons, f"{label}-duration")
        if any(not math.isfinite(value) or value < 0.0 for value in optional_durations):
            _append_once(reasons, f"{label}-duration")
        if sample.draft_selection_seconds > sample.total_draft_seconds:
            _append_once(reasons, f"{label}-selection-duration")

    expected_prompts = tuple(
        range(FIRST_PROMPT_INDEX, FIRST_PROMPT_INDEX + profile.minimum_prompt_count)
    )
    if tuple(sorted(by_prompt)) != expected_prompts:
        _append_once(reasons, "prompt-order")

    full_selector_medians: list[float] = []
    bounded_selector_medians: list[float] = []
    full_draft_medians: list[float] = []
    bounded_draft_medians: list[float] = []
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

        full_roots = [sample for sample in roots if sample.route == FULL_LOGIT_READBACK]
        bounded_roots = [sample for sample in roots if sample.route == SHARD_LOCAL_TOP1]
        if (
            len(full_roots) != EXPECTED_ROOTS_PER_ROUTE
            or len(bounded_roots) != EXPECTED_ROOTS_PER_ROUTE
        ):
            _append_once(reasons, f"prompt-{prompt_index}-route-count")
            continue
        for route, route_roots in (
            (FULL_LOGIT_READBACK, full_roots),
            (SHARD_LOCAL_TOP1, bounded_roots),
        ):
            timing_groups = (
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

        full_selector_medians.append(
            statistics.median(sample.draft_selection_seconds for sample in full_roots)
        )
        bounded_selector_medians.append(
            statistics.median(
                sample.draft_selection_seconds for sample in bounded_roots
            )
        )
        full_draft_medians.append(
            statistics.median(sample.total_draft_seconds for sample in full_roots)
        )
        bounded_draft_medians.append(
            statistics.median(sample.total_draft_seconds for sample in bounded_roots)
        )

        accepted = reference.accepted_prefix_length
        emitted_tokens = accepted + 1
        bounded_target_sequential = statistics.median(
            sample.target_sequential_seconds for sample in bounded_roots
        )
        baseline_seconds.append(
            bounded_target_sequential / (profile.draft_token_count + 1) * emitted_tokens
        )
        speculative_seconds.append(
            statistics.median(sample.total_draft_seconds for sample in bounded_roots)
            + statistics.median(
                sample.target_verification_seconds for sample in bounded_roots
            )
            + statistics.median(
                sample.target_verification_readback_seconds for sample in bounded_roots
            )
            + statistics.median(sample.rollback_seconds for sample in bounded_roots)
            + statistics.median(
                sample.mtp_cache_reprefill_seconds for sample in bounded_roots
            )
        )
        accepted_counts.append(accepted)

    if (
        full_selector_medians
        and len(full_selector_medians) == profile.minimum_prompt_count
    ):
        full_selector_total = sum(full_selector_medians)
        bounded_selector_total = sum(bounded_selector_medians)
        selector_ratio = bounded_selector_total / full_selector_total
        total_draft_ratio = sum(bounded_draft_medians) / sum(full_draft_medians)
        aggregate_speedup = sum(baseline_seconds) / sum(speculative_seconds)
        mean_accepted = statistics.fmean(accepted_counts)
        full_selector_median = statistics.median(full_selector_medians)
        bounded_selector_median = statistics.median(bounded_selector_medians)
        if selector_ratio > profile.maximum_selector_ratio:
            _append_once(reasons, "selector-gain")
        if aggregate_speedup <= profile.minimum_emitted_token_speedup:
            _append_once(reasons, "emitted-token-economics")
    else:
        selector_ratio = 0.0
        total_draft_ratio = 0.0
        aggregate_speedup = 0.0
        mean_accepted = 0.0
        full_selector_median = 0.0
        bounded_selector_median = 0.0

    selector_valid = not reasons
    blockers = ["production-admission-missing", "traced-decode-admission-missing"]
    if not selector_valid:
        blockers.insert(0, "mtp-selector-invalid")
    return MtpSelectionDecision(
        selector_valid=selector_valid,
        bounded_selector_selected=selector_valid,
        admission_eligible=False,
        reasons=tuple(reasons),
        blockers=tuple(blockers),
        selector_ratio=selector_ratio,
        total_draft_ratio=total_draft_ratio,
        aggregate_emitted_token_speedup=aggregate_speedup,
        mean_accepted_drafts=mean_accepted,
        full_selector_median_seconds=full_selector_median,
        bounded_selector_median_seconds=bounded_selector_median,
    )


def _sample_timing_valid(sample: MtpSelectionSample) -> bool:
    required = (
        sample.mtp_prefill_seconds,
        sample.draft_decode_seconds,
        sample.draft_selection_seconds,
        sample.target_sequential_seconds,
        sample.target_verification_seconds,
        sample.target_verification_readback_seconds,
    )
    optional = (sample.rollback_seconds, sample.mtp_cache_reprefill_seconds)
    return (
        all(math.isfinite(value) and value > 0.0 for value in required)
        and all(math.isfinite(value) and value >= 0.0 for value in optional)
        and sample.draft_selection_seconds <= sample.total_draft_seconds
    )


def _spread_ratio(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        return math.inf
    return max(values) / min(values)


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MtpSelectionPolicyError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise MtpSelectionPolicyError(
            f"{name} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MtpSelectionPolicyError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MtpSelectionPolicyError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MtpSelectionPolicyError(f"{name} must be a positive integer")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpSelectionPolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MtpSelectionPolicyError(f"{name} must be finite and positive")
    return result


def _require_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MtpSelectionPolicyError(f"{name} must be a non-empty array")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise MtpSelectionPolicyError(f"{name} contains an invalid integer")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise MtpSelectionPolicyError(f"{name} must be sorted and unique")
    return result


def _require_string_tuple(value: Any, name: str, *, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MtpSelectionPolicyError(f"{name} must be a non-empty array")
    result = tuple(_require_string(item, name) for item in value)
    if unique and len(set(result)) != len(result):
        raise MtpSelectionPolicyError(f"{name} must be unique")
    return result


def _require_lower_hex(value: Any, name: str, length: int) -> str:
    result = _require_string(value, name)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise MtpSelectionPolicyError(
            f"{name} must be a {length}-character lowercase hex value"
        )
    return result
