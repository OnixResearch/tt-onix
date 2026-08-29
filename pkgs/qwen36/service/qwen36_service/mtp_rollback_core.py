"""Pure policy and matched evidence evaluation for MTP K/V rollback."""

# r[impl tenstorrent.qwen38_p150x2.mtp_kv_rollback]
# r[impl tenstorrent.qwen38_p150x2.mtp_recovery_stability]

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
    MINIMUM_RATIO,
    REVISION_HEX_LENGTH,
    MtpDrafterIdentity,
    MtpDrafterPolicyError,
    accepted_prefix_length,
)

PROFILE_SCHEMA_VERSION = 2
# Rollback control stays pinned to the validated default draft width.
EXPECTED_ROLLBACK_DRAFT_COUNT = 3
REPREFILL_RECOVERY = "reprefill"
SNAPSHOT_RECOVERY = "snapshot"
PERSISTENT_TRACED_RECOVERY = "persistent-traced"
EXPECTED_PROCESS_TRACE_CAPTURE_COUNT = 1
ROUTE_BLOCK = (
    REPREFILL_RECOVERY,
    SNAPSHOT_RECOVERY,
    SNAPSHOT_RECOVERY,
    REPREFILL_RECOVERY,
)
EXPECTED_ROUTE_BLOCK_COUNT = 3
EXPECTED_ROUTE_SCHEDULE = ROUTE_BLOCK * EXPECTED_ROUTE_BLOCK_COUNT
EXPECTED_ROOTS_PER_PROMPT = len(EXPECTED_ROUTE_SCHEDULE)
EXPECTED_ROOTS_PER_ROUTE = EXPECTED_ROOTS_PER_PROMPT // 2
MINIMUM_SAMPLES_PER_ROUTE = 5
EXPECTED_SNAPSHOT_ROW_COUNT = EXPECTED_ROLLBACK_DRAFT_COUNT - 1
EXPECTED_COMMITTED_ADVANCE_COUNT = 1
REQUIRED_ACCEPTANCE_LENGTHS = (0, EXPECTED_ROLLBACK_DRAFT_COUNT)
FIRST_ROOT_INDEX = 1
MINIMUM_TOKEN_ID = 0
ZERO_DURATION_SECONDS = 0.0


class MtpRollbackPolicyError(ValueError):
    """The MTP rollback input does not satisfy its closed policy."""


@dataclass(frozen=True)
class MtpRollbackProfile:
    profile_id: str
    diagnostic_only: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    recovery_execution_mode: str
    required_process_trace_capture_count: int
    draft_token_count: int
    minimum_prompt_count: int
    route_block_count: int
    minimum_samples_per_route: int
    route_schedule: tuple[str, ...]
    snapshot_row_count: int
    committed_advance_count: int
    required_acceptance_lengths: tuple[int, ...]
    maximum_duplicate_root_ratio: float
    maximum_snapshot_recovery_ratio: float
    prompts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> MtpRollbackProfile:
        """Parse one closed exported Nickel rollback profile."""
        _require_exact_keys(
            source,
            {
                "schema_version",
                "profile_id",
                "diagnostic_only",
                "identities",
                "expected_pcie_device_ids",
                "recovery_execution_mode",
                "required_process_trace_capture_count",
                "draft_token_count",
                "minimum_prompt_count",
                "route_block_count",
                "minimum_samples_per_route",
                "route_schedule",
                "snapshot_row_count",
                "committed_advance_count",
                "required_acceptance_lengths",
                "maximum_duplicate_root_ratio",
                "maximum_snapshot_recovery_ratio",
                "prompts",
            },
            "MTP rollback profile",
        )
        if source["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise MtpRollbackPolicyError("MTP rollback schema version is unsupported")
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
            recovery_execution_mode=_require_string(
                source["recovery_execution_mode"], "recovery_execution_mode"
            ),
            required_process_trace_capture_count=_require_positive_int(
                source["required_process_trace_capture_count"],
                "required_process_trace_capture_count",
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
            snapshot_row_count=_require_positive_int(
                source["snapshot_row_count"], "snapshot_row_count"
            ),
            committed_advance_count=_require_positive_int(
                source["committed_advance_count"], "committed_advance_count"
            ),
            required_acceptance_lengths=_require_int_tuple(
                source["required_acceptance_lengths"],
                "required_acceptance_lengths",
            ),
            maximum_duplicate_root_ratio=_require_positive_float(
                source["maximum_duplicate_root_ratio"],
                "maximum_duplicate_root_ratio",
            ),
            maximum_snapshot_recovery_ratio=_require_positive_float(
                source["maximum_snapshot_recovery_ratio"],
                "maximum_snapshot_recovery_ratio",
            ),
            prompts=_require_string_tuple(source["prompts"], "prompts", unique=True),
        )
        if not profile.diagnostic_only:
            raise MtpRollbackPolicyError("MTP rollback control must be diagnostic-only")
        if profile.expected_pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
            raise MtpRollbackPolicyError(
                "MTP rollback control requires devices 0 and 1"
            )
        if profile.recovery_execution_mode != PERSISTENT_TRACED_RECOVERY:
            raise MtpRollbackPolicyError("MTP recovery must use a persistent trace")
        if (
            profile.required_process_trace_capture_count
            != EXPECTED_PROCESS_TRACE_CAPTURE_COUNT
        ):
            raise MtpRollbackPolicyError("MTP recovery must capture one process trace")
        if profile.draft_token_count != EXPECTED_ROLLBACK_DRAFT_COUNT:
            raise MtpRollbackPolicyError("MTP rollback control requires three drafts")
        if profile.minimum_prompt_count != EXPECTED_MINIMUM_PROMPT_COUNT:
            raise MtpRollbackPolicyError("MTP rollback control requires five prompts")
        if len(profile.prompts) != profile.minimum_prompt_count:
            raise MtpRollbackPolicyError(
                "MTP rollback prompt count differs from policy"
            )
        if profile.route_block_count != EXPECTED_ROUTE_BLOCK_COUNT:
            raise MtpRollbackPolicyError("MTP rollback route block count changed")
        if profile.minimum_samples_per_route != EXPECTED_ROOTS_PER_ROUTE:
            raise MtpRollbackPolicyError("MTP rollback retained sample count changed")
        if profile.minimum_samples_per_route < MINIMUM_SAMPLES_PER_ROUTE:
            raise MtpRollbackPolicyError(
                "MTP rollback requires at least five route samples"
            )
        if profile.route_schedule != EXPECTED_ROUTE_SCHEDULE:
            raise MtpRollbackPolicyError("MTP rollback route schedule changed")
        if profile.snapshot_row_count != EXPECTED_SNAPSHOT_ROW_COUNT:
            raise MtpRollbackPolicyError("MTP rollback snapshot row count changed")
        if profile.committed_advance_count != EXPECTED_COMMITTED_ADVANCE_COUNT:
            raise MtpRollbackPolicyError(
                "MTP rollback must advance one committed token"
            )
        if profile.required_acceptance_lengths != REQUIRED_ACCEPTANCE_LENGTHS:
            raise MtpRollbackPolicyError("MTP rollback acceptance coverage changed")
        if profile.maximum_duplicate_root_ratio < MINIMUM_RATIO:
            raise MtpRollbackPolicyError("duplicate-root ratio cannot be less than one")
        if (
            not ZERO_DURATION_SECONDS
            < profile.maximum_snapshot_recovery_ratio
            < MINIMUM_RATIO
        ):
            raise MtpRollbackPolicyError(
                "snapshot recovery ratio must be between zero and one"
            )
        return profile


@dataclass(frozen=True)
class MtpRollbackSample:
    prompt_index: int
    root_index: int
    route: str
    identity: MtpDrafterIdentity
    physical_pcie_device_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    committed_token_id: int
    next_draft_token_id: int
    recovery_seconds: float
    recovery_execution_mode: str
    process_trace_capture_count: int
    trace_binding_identity: str
    trace_binding_valid: bool
    snapshot_row_count: int
    committed_advance_count: int
    next_draft_valid: bool
    feedback_state_valid: bool
    kv_prefix_valid: bool
    target_output_valid: bool
    target_rollback_valid: bool


@dataclass(frozen=True)
class MtpRollbackDecision:
    rollback_valid: bool
    snapshot_rollback_selected: bool
    admission_eligible: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    snapshot_recovery_ratio: float
    reprefill_recovery_median_seconds: float
    snapshot_recovery_median_seconds: float
    mean_accepted_drafts: float
    observed_acceptance_lengths: tuple[int, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rollback_valid": self.rollback_valid,
            "snapshot_rollback_selected": self.snapshot_rollback_selected,
            "admission_eligible": self.admission_eligible,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "snapshot_recovery_ratio": self.snapshot_recovery_ratio,
            "reprefill_recovery_median_seconds": (
                self.reprefill_recovery_median_seconds
            ),
            "snapshot_recovery_median_seconds": (self.snapshot_recovery_median_seconds),
            "mean_accepted_drafts": self.mean_accepted_drafts,
            "observed_acceptance_lengths": list(self.observed_acceptance_lengths),
        }


def evaluate_mtp_rollback(
    profile: MtpRollbackProfile,
    expected_identity: MtpDrafterIdentity,
    samples: Sequence[MtpRollbackSample],
) -> MtpRollbackDecision:
    """Evaluate exact continuation state, stable recovery, and recovery gain."""
    reasons: list[str] = []
    ordered = sorted(
        samples, key=lambda sample: (sample.prompt_index, sample.root_index)
    )
    expected_sample_count = profile.minimum_prompt_count * EXPECTED_ROOTS_PER_PROMPT
    if len(ordered) != expected_sample_count:
        _append_once(reasons, "sample-count")

    by_prompt: dict[int, list[MtpRollbackSample]] = {}
    for sample in ordered:
        by_prompt.setdefault(sample.prompt_index, []).append(sample)
        label = f"prompt-{sample.prompt_index}-root-{sample.root_index}"
        if sample.identity != expected_identity:
            _append_once(reasons, f"{label}-identity")
        if sample.physical_pcie_device_ids != profile.expected_pcie_device_ids:
            _append_once(reasons, f"{label}-physical-devices")
        if sample.route not in (REPREFILL_RECOVERY, SNAPSHOT_RECOVERY):
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
        if not _valid_token(sample.committed_token_id):
            _append_once(reasons, f"{label}-committed-token")
        if not _valid_token(sample.next_draft_token_id):
            _append_once(reasons, f"{label}-next-draft-token")
        if (
            not math.isfinite(sample.recovery_seconds)
            or sample.recovery_seconds <= ZERO_DURATION_SECONDS
        ):
            _append_once(reasons, f"{label}-duration")
        if sample.recovery_execution_mode != profile.recovery_execution_mode:
            _append_once(reasons, f"{label}-recovery-execution-mode")
        if (
            sample.process_trace_capture_count
            != profile.required_process_trace_capture_count
        ):
            _append_once(reasons, f"{label}-process-trace-capture-count")
        if not sample.trace_binding_identity:
            _append_once(reasons, f"{label}-trace-binding-identity")
        if not sample.trace_binding_valid:
            _append_once(reasons, f"{label}-trace-binding")
        expected_rows = (
            profile.snapshot_row_count if sample.route == SNAPSHOT_RECOVERY else 0
        )
        if sample.snapshot_row_count != expected_rows:
            _append_once(reasons, f"{label}-snapshot-row-count")
        if sample.committed_advance_count != profile.committed_advance_count:
            _append_once(reasons, f"{label}-committed-advance-count")
        if not sample.next_draft_valid:
            _append_once(reasons, f"{label}-next-draft-state")
        if not sample.feedback_state_valid:
            _append_once(reasons, f"{label}-feedback-state")
        if not sample.kv_prefix_valid:
            _append_once(reasons, f"{label}-kv-prefix")
        if not sample.target_output_valid:
            _append_once(reasons, f"{label}-target-output")
        if not sample.target_rollback_valid:
            _append_once(reasons, f"{label}-target-rollback")

    binding_identities = {
        sample.trace_binding_identity for sample in ordered if sample.trace_binding_identity
    }
    if len(binding_identities) != 1:
        _append_once(reasons, "process-trace-binding-parity")

    expected_prompts = tuple(
        range(FIRST_PROMPT_INDEX, FIRST_PROMPT_INDEX + profile.minimum_prompt_count)
    )
    if tuple(sorted(by_prompt)) != expected_prompts:
        _append_once(reasons, "prompt-order")

    reprefill_medians: list[float] = []
    snapshot_medians: list[float] = []
    acceptance_lengths: list[int] = []
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

        reference = roots[0]
        for sample in roots[1:]:
            if sample.trace_binding_identity != reference.trace_binding_identity:
                _append_once(reasons, f"prompt-{prompt_index}-trace-binding-parity")
            if (
                sample.draft_token_ids != reference.draft_token_ids
                or sample.target_token_ids != reference.target_token_ids
                or sample.accepted_prefix_length != reference.accepted_prefix_length
                or sample.committed_token_id != reference.committed_token_id
                or sample.next_draft_token_id != reference.next_draft_token_id
            ):
                _append_once(reasons, f"prompt-{prompt_index}-route-parity")
        reprefill_roots = [
            sample for sample in roots if sample.route == REPREFILL_RECOVERY
        ]
        snapshot_roots = [
            sample for sample in roots if sample.route == SNAPSHOT_RECOVERY
        ]
        if (
            len(reprefill_roots) != profile.minimum_samples_per_route
            or len(snapshot_roots) != profile.minimum_samples_per_route
        ):
            _append_once(reasons, f"prompt-{prompt_index}-route-count")
            continue
        for route, route_roots in (
            (REPREFILL_RECOVERY, reprefill_roots),
            (SNAPSHOT_RECOVERY, snapshot_roots),
        ):
            values = [sample.recovery_seconds for sample in route_roots]
            if _spread_ratio(values) > profile.maximum_duplicate_root_ratio:
                _append_once(reasons, f"prompt-{prompt_index}-{route}-instability")
        reprefill_medians.append(
            statistics.median(sample.recovery_seconds for sample in reprefill_roots)
        )
        snapshot_medians.append(
            statistics.median(sample.recovery_seconds for sample in snapshot_roots)
        )
        acceptance_lengths.append(reference.accepted_prefix_length)

    observed_acceptance_lengths = tuple(sorted(set(acceptance_lengths)))
    if not set(profile.required_acceptance_lengths).issubset(
        observed_acceptance_lengths
    ):
        _append_once(reasons, "acceptance-coverage")

    if reprefill_medians and len(reprefill_medians) == profile.minimum_prompt_count:
        snapshot_ratio = sum(snapshot_medians) / sum(reprefill_medians)
        reprefill_median = statistics.median(reprefill_medians)
        snapshot_median = statistics.median(snapshot_medians)
        mean_accepted = statistics.fmean(acceptance_lengths)
        if snapshot_ratio > profile.maximum_snapshot_recovery_ratio:
            _append_once(reasons, "snapshot-recovery-gain")
    else:
        snapshot_ratio = ZERO_DURATION_SECONDS
        reprefill_median = ZERO_DURATION_SECONDS
        snapshot_median = ZERO_DURATION_SECONDS
        mean_accepted = ZERO_DURATION_SECONDS

    rollback_valid = not reasons
    blockers = ["mtp-admission-missing", "production-admission-missing"]
    if not rollback_valid:
        blockers.insert(0, "mtp-recovery-stability-invalid")
    return MtpRollbackDecision(
        rollback_valid=rollback_valid,
        snapshot_rollback_selected=rollback_valid,
        admission_eligible=False,
        reasons=tuple(reasons),
        blockers=tuple(blockers),
        snapshot_recovery_ratio=snapshot_ratio,
        reprefill_recovery_median_seconds=reprefill_median,
        snapshot_recovery_median_seconds=snapshot_median,
        mean_accepted_drafts=mean_accepted,
        observed_acceptance_lengths=observed_acceptance_lengths,
    )


def _valid_token(value: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= MINIMUM_TOKEN_ID
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
        raise MtpRollbackPolicyError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise MtpRollbackPolicyError(
            f"{name} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MtpRollbackPolicyError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MtpRollbackPolicyError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MtpRollbackPolicyError(f"{name} must be a positive integer")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpRollbackPolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= ZERO_DURATION_SECONDS:
        raise MtpRollbackPolicyError(f"{name} must be finite and positive")
    return result


def _require_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MtpRollbackPolicyError(f"{name} must be a non-empty array")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise MtpRollbackPolicyError(f"{name} contains an invalid integer")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise MtpRollbackPolicyError(f"{name} must be sorted and unique")
    return result


def _require_string_tuple(value: Any, name: str, *, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MtpRollbackPolicyError(f"{name} must be a non-empty array")
    result = tuple(_require_string(item, name) for item in value)
    if unique and len(set(result)) != len(result):
        raise MtpRollbackPolicyError(f"{name} must be unique")
    return result


def _require_lower_hex(value: Any, name: str, length: int) -> str:
    result = _require_string(value, name)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise MtpRollbackPolicyError(
            f"{name} must be a {length}-character lowercase hex value"
        )
    return result
