"""Pure policy and evidence evaluation for the Qwen3.8 MTP drafter."""

# r[impl tenstorrent.qwen38_p150x2.mtp_drafter_probe]

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROFILE_SCHEMA_VERSION = 1
EXPECTED_PCIE_DEVICE_IDS = (0, 1)
MINIMUM_DRAFT_TOKEN_COUNT = 2
MAXIMUM_DRAFT_TOKEN_COUNT = 6
EXPECTED_MINIMUM_PROMPT_COUNT = 5
MINIMUM_RATIO = 1.0
MINIMUM_TOKEN_ID = 0
FIRST_PROMPT_INDEX = 1
REVISION_HEX_LENGTH = 40
BLAKE3_HEX_LENGTH = 64
MTP_HIDDEN_SIZE = 5120
MTP_FUSED_INPUT_SIZE = 10240
MTP_INTERMEDIATE_SIZE = 17408
MTP_HEAD_SIZE = 256
MTP_QUERY_GATE_SIZE = 12288
MTP_KV_SIZE = 1024
MTP_ATTENTION_OUTPUT_SIZE = 6144

EXPECTED_MTP_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "mtp.fc.weight": (MTP_HIDDEN_SIZE, MTP_FUSED_INPUT_SIZE),
    "mtp.layers.0.input_layernorm.weight": (MTP_HIDDEN_SIZE,),
    "mtp.layers.0.mlp.down_proj.weight": (
        MTP_HIDDEN_SIZE,
        MTP_INTERMEDIATE_SIZE,
    ),
    "mtp.layers.0.mlp.gate_proj.weight": (
        MTP_INTERMEDIATE_SIZE,
        MTP_HIDDEN_SIZE,
    ),
    "mtp.layers.0.mlp.up_proj.weight": (
        MTP_INTERMEDIATE_SIZE,
        MTP_HIDDEN_SIZE,
    ),
    "mtp.layers.0.post_attention_layernorm.weight": (MTP_HIDDEN_SIZE,),
    "mtp.layers.0.self_attn.k_norm.weight": (MTP_HEAD_SIZE,),
    "mtp.layers.0.self_attn.k_proj.weight": (MTP_KV_SIZE, MTP_HIDDEN_SIZE),
    "mtp.layers.0.self_attn.o_proj.weight": (
        MTP_HIDDEN_SIZE,
        MTP_ATTENTION_OUTPUT_SIZE,
    ),
    "mtp.layers.0.self_attn.q_norm.weight": (MTP_HEAD_SIZE,),
    "mtp.layers.0.self_attn.q_proj.weight": (
        MTP_QUERY_GATE_SIZE,
        MTP_HIDDEN_SIZE,
    ),
    "mtp.layers.0.self_attn.v_proj.weight": (MTP_KV_SIZE, MTP_HIDDEN_SIZE),
    "mtp.norm.weight": (MTP_HIDDEN_SIZE,),
    "mtp.pre_fc_norm_embedding.weight": (MTP_HIDDEN_SIZE,),
    "mtp.pre_fc_norm_hidden.weight": (MTP_HIDDEN_SIZE,),
}


class MtpDrafterPolicyError(ValueError):
    """The MTP drafter input does not satisfy its closed policy."""


@dataclass(frozen=True)
class MtpDrafterIdentity:
    runtime_identity: str
    package_identity: str
    model_identity: str


@dataclass(frozen=True)
class MtpDrafterProfile:
    profile_id: str
    diagnostic_only: bool
    authority_blake3: str
    checkpoint_revision: str
    model_source_revision: str
    tokenizer_revision: str
    expected_pcie_device_ids: tuple[int, ...]
    draft_token_count: int
    minimum_prompt_count: int
    maximum_sample_spread_ratio: float
    minimum_emitted_token_speedup: float
    prompts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> MtpDrafterProfile:
        """Parse one closed exported Nickel profile."""
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
                "maximum_sample_spread_ratio",
                "minimum_emitted_token_speedup",
                "prompts",
            },
            "MTP drafter profile",
        )
        if source["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise MtpDrafterPolicyError("MTP drafter schema version is unsupported")
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
                allow_zero=True,
            ),
            draft_token_count=_require_positive_int(
                source["draft_token_count"], "draft_token_count"
            ),
            minimum_prompt_count=_require_positive_int(
                source["minimum_prompt_count"], "minimum_prompt_count"
            ),
            maximum_sample_spread_ratio=_require_positive_float(
                source["maximum_sample_spread_ratio"],
                "maximum_sample_spread_ratio",
            ),
            minimum_emitted_token_speedup=_require_positive_float(
                source["minimum_emitted_token_speedup"],
                "minimum_emitted_token_speedup",
            ),
            prompts=_require_string_tuple(source["prompts"], "prompts"),
        )
        if not profile.diagnostic_only:
            raise MtpDrafterPolicyError("MTP drafter probing must be diagnostic-only")
        if profile.expected_pcie_device_ids != EXPECTED_PCIE_DEVICE_IDS:
            raise MtpDrafterPolicyError("MTP drafter probing requires devices 0 and 1")
        if not (
            MINIMUM_DRAFT_TOKEN_COUNT
            <= profile.draft_token_count
            <= MAXIMUM_DRAFT_TOKEN_COUNT
        ):
            raise MtpDrafterPolicyError(
                "MTP draft count must lie between "
                f"{MINIMUM_DRAFT_TOKEN_COUNT} and {MAXIMUM_DRAFT_TOKEN_COUNT}"
            )
        if profile.minimum_prompt_count != EXPECTED_MINIMUM_PROMPT_COUNT:
            raise MtpDrafterPolicyError("MTP drafter probing requires five prompts")
        if len(profile.prompts) != profile.minimum_prompt_count:
            raise MtpDrafterPolicyError("MTP drafter prompt count differs from policy")
        if profile.maximum_sample_spread_ratio < MINIMUM_RATIO:
            raise MtpDrafterPolicyError("sample spread cannot be less than one")
        if profile.minimum_emitted_token_speedup < MINIMUM_RATIO:
            raise MtpDrafterPolicyError("speedup bound cannot be less than one")
        return profile


@dataclass(frozen=True)
class MtpDrafterSample:
    prompt_index: int
    identity: MtpDrafterIdentity
    physical_pcie_device_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_prefix_length: int
    mtp_prefill_seconds: float
    draft_seconds: float
    target_sequential_seconds: float
    target_verification_seconds: float
    target_verification_readback_seconds: float
    rollback_seconds: float
    mtp_cache_reprefill_seconds: float
    rollback_valid: bool
    target_output_valid: bool


@dataclass(frozen=True)
class MtpDrafterDecision:
    drafter_valid: bool
    mtp_selected: bool
    admission_eligible: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    mean_accepted_drafts: float
    aggregate_emitted_token_speedup: float
    median_emitted_token_speedup: float
    median_draft_seconds: float
    median_target_verification_seconds: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "drafter_valid": self.drafter_valid,
            "mtp_selected": self.mtp_selected,
            "admission_eligible": self.admission_eligible,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "mean_accepted_drafts": self.mean_accepted_drafts,
            "aggregate_emitted_token_speedup": self.aggregate_emitted_token_speedup,
            "median_emitted_token_speedup": self.median_emitted_token_speedup,
            "median_draft_seconds": self.median_draft_seconds,
            "median_target_verification_seconds": self.median_target_verification_seconds,
        }


def validate_mtp_weight_shapes(
    observed: Mapping[str, tuple[int, ...]],
) -> None:
    """Reject missing, extra, or shape-drifted MTP tensors."""
    expected_keys = set(EXPECTED_MTP_WEIGHT_SHAPES)
    observed_keys = set(observed)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise MtpDrafterPolicyError(
            f"MTP weight keys differ: missing={missing}, extra={extra}"
        )
    for name, expected_shape in EXPECTED_MTP_WEIGHT_SHAPES.items():
        if tuple(observed[name]) != expected_shape:
            raise MtpDrafterPolicyError(
                f"MTP weight shape differs for {name}: "
                f"expected={expected_shape}, observed={tuple(observed[name])}"
            )


def accepted_prefix_length(drafts: Sequence[int], target_tokens: Sequence[int]) -> int:
    """Return the exact greedy accepted draft prefix."""
    if not MINIMUM_DRAFT_TOKEN_COUNT <= len(drafts) <= MAXIMUM_DRAFT_TOKEN_COUNT:
        raise MtpDrafterPolicyError("draft token count is outside the policy range")
    if len(drafts) != len(target_tokens):
        raise MtpDrafterPolicyError("target token count differs from draft count")
    for index, (draft, target) in enumerate(zip(drafts, target_tokens, strict=True)):
        if not _valid_token(draft) or not _valid_token(target):
            raise MtpDrafterPolicyError("draft or target token is invalid")
        if draft != target:
            return index
    return len(drafts)


def evaluate_mtp_drafter(
    profile: MtpDrafterProfile,
    expected_identity: MtpDrafterIdentity,
    samples: Sequence[MtpDrafterSample],
) -> MtpDrafterDecision:
    """Evaluate aligned local acceptance, rollback, stability, and economics."""
    reasons: list[str] = []
    ordered = sorted(samples, key=lambda sample: sample.prompt_index)
    if len(ordered) < profile.minimum_prompt_count:
        _append_once(reasons, "prompt-count")
    expected_indices = tuple(
        range(FIRST_PROMPT_INDEX, FIRST_PROMPT_INDEX + len(ordered))
    )
    if tuple(sample.prompt_index for sample in ordered) != expected_indices:
        _append_once(reasons, "prompt-order")

    accepted = []
    draft_times = []
    verify_times = []
    sequential_per_token = []
    speedups = []
    baseline_times = []
    speculative_times = []
    for sample in ordered:
        label = f"prompt-{sample.prompt_index}"
        if sample.identity != expected_identity:
            _append_once(reasons, f"{label}-identity")
        if sample.physical_pcie_device_ids != profile.expected_pcie_device_ids:
            _append_once(reasons, f"{label}-physical-devices")
        try:
            computed_acceptance = accepted_prefix_length(
                sample.draft_token_ids,
                sample.target_token_ids,
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
        durations = (
            sample.mtp_prefill_seconds,
            sample.draft_seconds,
            sample.target_sequential_seconds,
            sample.target_verification_seconds,
            sample.target_verification_readback_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            _append_once(reasons, f"{label}-duration")
            continue
        optional_durations = (
            sample.rollback_seconds,
            sample.mtp_cache_reprefill_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in optional_durations):
            _append_once(reasons, f"{label}-duration")
            continue
        emitted_tokens = computed_acceptance + 1
        baseline_seconds = (
            sample.target_sequential_seconds
            / (profile.draft_token_count + 1)
            * emitted_tokens
        )
        speculative_seconds = (
            sample.draft_seconds
            + sample.target_verification_seconds
            + sample.target_verification_readback_seconds
            + sample.rollback_seconds
            + sample.mtp_cache_reprefill_seconds
        )
        speedups.append(baseline_seconds / speculative_seconds)
        baseline_times.append(baseline_seconds)
        speculative_times.append(speculative_seconds)
        accepted.append(computed_acceptance)
        draft_times.append(sample.draft_seconds)
        verify_times.append(sample.target_verification_seconds)
        sequential_per_token.append(
            sample.target_sequential_seconds / (profile.draft_token_count + 1)
        )

    if len(accepted) == len(ordered) and accepted:
        spread_groups = (draft_times, verify_times, sequential_per_token)
        if any(
            _spread_ratio(values) > profile.maximum_sample_spread_ratio
            for values in spread_groups
        ):
            _append_once(reasons, "sample-instability")
        aggregate_speedup = sum(baseline_times) / sum(speculative_times)
        median_speedup = statistics.median(speedups)
        if aggregate_speedup <= profile.minimum_emitted_token_speedup:
            _append_once(reasons, "emitted-token-economics")
        mean_accepted = statistics.fmean(accepted)
        median_draft = statistics.median(draft_times)
        median_verify = statistics.median(verify_times)
    else:
        mean_accepted = 0.0
        aggregate_speedup = 0.0
        median_speedup = 0.0
        median_draft = 0.0
        median_verify = 0.0

    drafter_valid = not reasons
    blockers = ["production-admission-missing"]
    if not drafter_valid:
        blockers.insert(0, "mtp-drafter-invalid")
    return MtpDrafterDecision(
        drafter_valid=drafter_valid,
        mtp_selected=drafter_valid,
        admission_eligible=False,
        reasons=tuple(reasons),
        blockers=tuple(blockers),
        mean_accepted_drafts=mean_accepted,
        aggregate_emitted_token_speedup=aggregate_speedup,
        median_emitted_token_speedup=median_speedup,
        median_draft_seconds=median_draft,
        median_target_verification_seconds=median_verify,
    )


def _spread_ratio(values: Sequence[float]) -> float:
    return max(values) / min(values)


def _valid_token(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= MINIMUM_TOKEN_ID
    )


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MtpDrafterPolicyError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise MtpDrafterPolicyError(
            f"{name} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MtpDrafterPolicyError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MtpDrafterPolicyError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MtpDrafterPolicyError(f"{name} must be a positive integer")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MtpDrafterPolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise MtpDrafterPolicyError(f"{name} must be finite and positive")
    return result


def _require_int_tuple(value: Any, name: str, *, allow_zero: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MtpDrafterPolicyError(f"{name} must be a non-empty array")
    minimum = 0 if allow_zero else 1
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < minimum
        for item in value
    ):
        raise MtpDrafterPolicyError(f"{name} contains an invalid integer")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise MtpDrafterPolicyError(f"{name} must be sorted and unique")
    return result


def _require_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MtpDrafterPolicyError(f"{name} must be a non-empty array")
    result = tuple(_require_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise MtpDrafterPolicyError(f"{name} must be unique")
    return result


def _require_lower_hex(value: Any, name: str, length: int) -> str:
    result = _require_string(value, name)
    if len(result) != length or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise MtpDrafterPolicyError(
            f"{name} must be a {length}-character lowercase hex value"
        )
    return result
