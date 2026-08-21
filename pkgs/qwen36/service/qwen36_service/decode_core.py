"""Pure traced-decode policy for Qwen3.8-27B on P150x2."""

# r[impl tenstorrent.qwen38_p150x2.persistent_decode_buffers]
# r[impl tenstorrent.qwen38_p150x2.shard_local_argmax]
# r[impl tenstorrent.qwen38_p150x2.decode_performance_admission]
# r[impl tenstorrent.qwen38_p150x2.traced_decode_ttft]

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

PROFILE_SCHEMA_VERSION = 1
TRACE_COMMAND_QUEUE_ID = 0
SINGLE_DECODE_SLOT_COUNT = 1
REVISION_HEX_LENGTH = 40
BLAKE3_HEX_LENGTH = 64
UNMEASURED_CANDIDATE_RECEIPT = "not-measured"


class DecodePolicyError(ValueError):
    """The traced-decode input does not satisfy the profile contract."""


def required_page_table_blocks(
    *,
    prompt_tokens: int,
    generation_tokens: int,
    kv_block_size: int,
    capacity_blocks: int,
) -> int:
    """Return the bounded block count required by one generation request."""
    values = (
        ("prompt token count", prompt_tokens),
        ("generation token count", generation_tokens),
        ("KV block size", kv_block_size),
        ("page-table capacity", capacity_blocks),
    )
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DecodePolicyError(f"{name} must be a positive integer")
    total_tokens = prompt_tokens + generation_tokens
    required_blocks = (total_tokens + kv_block_size - 1) // kv_block_size
    if required_blocks > capacity_blocks:
        raise DecodePolicyError(
            f"request requires {required_blocks} KV blocks; capacity is {capacity_blocks}"
        )
    return required_blocks


class SlotState(Enum):
    EMPTY = "empty"
    ADMITTED = "admitted"
    ACTIVE = "active"
    RESETTING = "resetting"
    QUARANTINED = "quarantined"


class SlotEvent(Enum):
    ADMIT = "admit"
    ACTIVATE = "activate"
    BEGIN_RESET = "begin-reset"
    COMPLETE_RESET = "complete-reset"
    RESET_FAILED = "reset-failed"
    REINITIALIZE = "reinitialize"


@dataclass(frozen=True)
class DecodeSlot:
    slot_id: str
    ownership_generation: int
    buffer_generation: int
    state: SlotState
    owner_request_id: str | None = None
    quarantine_reason: str | None = None


def transition_slot(
    slot: DecodeSlot,
    event: SlotEvent,
    *,
    request_id: str | None = None,
    new_buffer_generation: int | None = None,
    quarantine_reason: str | None = None,
) -> DecodeSlot:
    """Apply one valid slot transition and reject every other transition."""
    if event is SlotEvent.ADMIT:
        if slot.state is not SlotState.EMPTY or slot.owner_request_id is not None:
            raise DecodePolicyError("only an empty unowned slot can admit a request")
        if not request_id:
            raise DecodePolicyError("slot admission requires a request identity")
        return DecodeSlot(
            slot_id=slot.slot_id,
            ownership_generation=slot.ownership_generation + 1,
            buffer_generation=slot.buffer_generation,
            state=SlotState.ADMITTED,
            owner_request_id=request_id,
        )
    if event is SlotEvent.ACTIVATE:
        if slot.state is not SlotState.ADMITTED or request_id != slot.owner_request_id:
            raise DecodePolicyError("only the admitted owner can activate a slot")
        return DecodeSlot(
            slot_id=slot.slot_id,
            ownership_generation=slot.ownership_generation,
            buffer_generation=slot.buffer_generation,
            state=SlotState.ACTIVE,
            owner_request_id=slot.owner_request_id,
        )
    if event is SlotEvent.BEGIN_RESET:
        if slot.state not in (SlotState.ADMITTED, SlotState.ACTIVE):
            raise DecodePolicyError("only an owned slot can begin reset")
        if request_id != slot.owner_request_id:
            raise DecodePolicyError("only the slot owner can begin reset")
        return DecodeSlot(
            slot_id=slot.slot_id,
            ownership_generation=slot.ownership_generation,
            buffer_generation=slot.buffer_generation,
            state=SlotState.RESETTING,
            owner_request_id=slot.owner_request_id,
        )
    if event is SlotEvent.COMPLETE_RESET:
        if slot.state is not SlotState.RESETTING:
            raise DecodePolicyError("only a resetting slot can complete reset")
        return DecodeSlot(
            slot_id=slot.slot_id,
            ownership_generation=slot.ownership_generation,
            buffer_generation=slot.buffer_generation,
            state=SlotState.EMPTY,
        )
    if event is SlotEvent.RESET_FAILED:
        if slot.state is not SlotState.RESETTING:
            raise DecodePolicyError("only a resetting slot can enter quarantine")
        if not quarantine_reason:
            raise DecodePolicyError("slot quarantine requires a reason")
        return DecodeSlot(
            slot_id=slot.slot_id,
            ownership_generation=slot.ownership_generation,
            buffer_generation=slot.buffer_generation,
            state=SlotState.QUARANTINED,
            quarantine_reason=quarantine_reason,
        )
    if event is SlotEvent.REINITIALIZE:
        if slot.state is not SlotState.QUARANTINED:
            raise DecodePolicyError("only a quarantined slot can be reinitialized")
        if (
            new_buffer_generation is None
            or new_buffer_generation <= slot.buffer_generation
        ):
            raise DecodePolicyError(
                "reinitialization requires a newer buffer generation"
            )
        return DecodeSlot(
            slot_id=slot.slot_id,
            ownership_generation=slot.ownership_generation,
            buffer_generation=new_buffer_generation,
            state=SlotState.EMPTY,
        )
    raise DecodePolicyError(f"unsupported slot event: {event.value}")


@dataclass(frozen=True)
class DecodeProfile:
    profile_id: str
    physically_admitted: bool
    runtime_source_revision: str
    checkpoint_revision: str
    tokenizer_revision: str
    authority_blake3: str
    baseline_receipt_blake3: str
    candidate_receipt_blake3: str
    trace_enabled: bool
    trace_capture_timeout_seconds: int
    request_timeout_seconds: int
    trace_command_queue_id: int
    slot_count: int
    maximum_sequence_length: int
    kv_block_size: int
    trace_region_size_bytes: int
    prompts: tuple[str, ...]
    warmup_runs: int
    measured_runs: int
    generated_tokens: int
    minimum_median_decode_speedup_ratio: float
    maximum_ttft_regression_ratio: float
    maximum_tail_latency_regression_ratio: float
    require_clean_reset: bool
    require_exact_token_parity: bool
    require_physical_evidence: bool

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> DecodeProfile:
        """Parse the deterministic Nickel export with closed-record checks."""
        _require_exact_keys(
            source,
            {
                "schema_version",
                "profile_id",
                "physically_admitted",
                "identities",
                "trace",
                "buffers",
                "workload",
                "admission",
            },
            "decode profile",
        )
        if source["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise DecodePolicyError("decode profile schema version is not supported")
        identities = _require_mapping(source["identities"], "identities")
        trace = _require_mapping(source["trace"], "trace")
        buffers = _require_mapping(source["buffers"], "buffers")
        workload = _require_mapping(source["workload"], "workload")
        admission = _require_mapping(source["admission"], "admission")
        _require_exact_keys(
            identities,
            {
                "authority_blake3",
                "baseline_receipt_blake3",
                "candidate_receipt_blake3",
                "checkpoint_revision",
                "model_source_revision",
                "tokenizer_revision",
            },
            "identities",
        )
        _require_exact_keys(
            trace,
            {
                "capture_timeout_seconds",
                "request_timeout_seconds",
                "enabled",
                "slot_count",
                "command_queue_id",
            },
            "trace",
        )
        _require_exact_keys(
            buffers,
            {"kv_block_size", "maximum_sequence_length", "trace_region_size_bytes"},
            "buffers",
        )
        _require_exact_keys(
            workload,
            {"prompts", "warmup_runs", "measured_runs", "generated_tokens"},
            "workload",
        )
        _require_exact_keys(
            admission,
            {
                "maximum_tail_latency_regression_ratio",
                "maximum_ttft_regression_ratio",
                "minimum_median_decode_speedup_ratio",
                "require_clean_reset",
                "require_exact_token_parity",
                "require_physical_evidence",
            },
            "admission",
        )
        prompts_source = workload["prompts"]
        if not isinstance(prompts_source, list) or not prompts_source:
            raise DecodePolicyError("workload prompts must be a non-empty array")
        prompts = tuple(
            _require_non_empty_string(item, "workload prompt")
            for item in prompts_source
        )
        result = cls(
            profile_id=_require_non_empty_string(source["profile_id"], "profile_id"),
            physically_admitted=_require_bool(
                source["physically_admitted"], "physically_admitted"
            ),
            runtime_source_revision=_require_hex(
                identities["model_source_revision"],
                "model_source_revision",
                REVISION_HEX_LENGTH,
            ),
            checkpoint_revision=_require_hex(
                identities["checkpoint_revision"],
                "checkpoint_revision",
                REVISION_HEX_LENGTH,
            ),
            tokenizer_revision=_require_hex(
                identities["tokenizer_revision"],
                "tokenizer_revision",
                REVISION_HEX_LENGTH,
            ),
            authority_blake3=_require_hex(
                identities["authority_blake3"],
                "authority_blake3",
                BLAKE3_HEX_LENGTH,
            ),
            baseline_receipt_blake3=_require_hex(
                identities["baseline_receipt_blake3"],
                "baseline_receipt_blake3",
                BLAKE3_HEX_LENGTH,
            ),
            candidate_receipt_blake3=_require_candidate_receipt(
                identities["candidate_receipt_blake3"]
            ),
            trace_enabled=_require_bool(trace["enabled"], "trace.enabled"),
            trace_capture_timeout_seconds=_require_positive_int(
                trace["capture_timeout_seconds"], "trace.capture_timeout_seconds"
            ),
            request_timeout_seconds=_require_positive_int(
                trace["request_timeout_seconds"], "trace.request_timeout_seconds"
            ),
            trace_command_queue_id=_require_non_negative_int(
                trace["command_queue_id"], "trace.command_queue_id"
            ),
            slot_count=_require_positive_int(trace["slot_count"], "trace.slot_count"),
            maximum_sequence_length=_require_positive_int(
                buffers["maximum_sequence_length"], "buffers.maximum_sequence_length"
            ),
            kv_block_size=_require_positive_int(
                buffers["kv_block_size"], "buffers.kv_block_size"
            ),
            trace_region_size_bytes=_require_positive_int(
                buffers["trace_region_size_bytes"], "buffers.trace_region_size_bytes"
            ),
            prompts=prompts,
            warmup_runs=_require_positive_int(
                workload["warmup_runs"], "workload.warmup_runs"
            ),
            measured_runs=_require_positive_int(
                workload["measured_runs"], "workload.measured_runs"
            ),
            generated_tokens=_require_positive_int(
                workload["generated_tokens"], "workload.generated_tokens"
            ),
            minimum_median_decode_speedup_ratio=_require_positive_float(
                admission["minimum_median_decode_speedup_ratio"],
                "admission.minimum_median_decode_speedup_ratio",
            ),
            maximum_ttft_regression_ratio=_require_positive_float(
                admission["maximum_ttft_regression_ratio"],
                "admission.maximum_ttft_regression_ratio",
            ),
            maximum_tail_latency_regression_ratio=_require_positive_float(
                admission["maximum_tail_latency_regression_ratio"],
                "admission.maximum_tail_latency_regression_ratio",
            ),
            require_clean_reset=_require_bool(
                admission["require_clean_reset"], "admission.require_clean_reset"
            ),
            require_exact_token_parity=_require_bool(
                admission["require_exact_token_parity"],
                "admission.require_exact_token_parity",
            ),
            require_physical_evidence=_require_bool(
                admission["require_physical_evidence"],
                "admission.require_physical_evidence",
            ),
        )
        if result.slot_count != SINGLE_DECODE_SLOT_COUNT:
            raise DecodePolicyError(
                "the P150x2 service requires exactly one decode slot"
            )
        if result.trace_command_queue_id != TRACE_COMMAND_QUEUE_ID:
            raise DecodePolicyError("the decode trace must use command queue zero")
        if result.generated_tokens > result.maximum_sequence_length:
            raise DecodePolicyError("generated token bound exceeds the sequence bound")
        if (
            result.physically_admitted
            and result.candidate_receipt_blake3 == UNMEASURED_CANDIDATE_RECEIPT
        ):
            raise DecodePolicyError("physical admission requires a candidate receipt")
        return result


@dataclass(frozen=True)
class TraceBinding:
    runtime_identity: str
    package_identity: str
    model_identity: str
    profile_id: str
    slot_id: str
    buffer_generation: int
    token_buffer_identity: str
    position_buffer_identity: str
    rope_buffer_identity: str
    page_table_buffer_identity: str
    kv_buffer_identity: str
    gdn_buffer_identity: str
    output_buffer_identity: str
    output_shape: tuple[int, ...]
    output_dtype: str


def validate_trace_binding(
    expected: TraceBinding, observed: TraceBinding
) -> tuple[str, ...]:
    """Return every mismatched binding field in deterministic field order."""
    mismatches = []
    for field in fields(TraceBinding):
        if getattr(expected, field.name) != getattr(observed, field.name):
            mismatches.append(field.name)
    return tuple(mismatches)


def request_deadline_expired(current_time: float, deadline: float) -> bool:
    """Decide timeout status from explicit monotonic time facts."""
    if not math.isfinite(current_time) or not math.isfinite(deadline):
        raise DecodePolicyError("request time facts must be finite")
    return current_time >= deadline


@dataclass(frozen=True)
class ShardWinner:
    shard_start: int
    shard_stop: int
    token_index: int
    maximum_logit: float


def select_global_winner(winners: Sequence[ShardWinner], vocabulary_size: int) -> int:
    """Validate complete vocabulary ranges and select the lowest-index global maximum."""
    if vocabulary_size <= 0:
        raise DecodePolicyError("vocabulary size must be positive")
    if not winners:
        raise DecodePolicyError("at least one shard winner is required")
    ordered = sorted(
        winners, key=lambda winner: (winner.shard_start, winner.shard_stop)
    )
    expected_start = 0
    seen_ranges: set[tuple[int, int]] = set()
    for winner in ordered:
        shard_range = (winner.shard_start, winner.shard_stop)
        if shard_range in seen_ranges:
            raise DecodePolicyError("duplicate shard range")
        seen_ranges.add(shard_range)
        if (
            winner.shard_start != expected_start
            or winner.shard_stop <= winner.shard_start
        ):
            raise DecodePolicyError(
                "shard ranges must be complete, ordered, and non-overlapping"
            )
        if not winner.shard_start <= winner.token_index < winner.shard_stop:
            raise DecodePolicyError("shard winner token is outside its range")
        if not math.isfinite(winner.maximum_logit):
            raise DecodePolicyError("shard winner logit must be finite")
        expected_start = winner.shard_stop
    if expected_start != vocabulary_size:
        raise DecodePolicyError("shard ranges do not cover the vocabulary")
    return min(
        ordered, key=lambda winner: (-winner.maximum_logit, winner.token_index)
    ).token_index


@dataclass(frozen=True)
class DecodeMetrics:
    median_decode_tokens_per_second: float
    median_ttft_seconds: float
    p99_token_latency_seconds: float


@dataclass(frozen=True)
class DecodeEvidence:
    physical: bool
    identities_match: bool
    exact_tokens_match: bool
    clean_reset: bool
    baseline: DecodeMetrics
    candidate: DecodeMetrics


@dataclass(frozen=True)
class DecodeAdmission:
    admitted: bool
    reasons: tuple[str, ...]


def admit_decode_candidate(
    profile: DecodeProfile, evidence: DecodeEvidence
) -> DecodeAdmission:
    """Evaluate complete matched evidence without performing external effects."""
    reasons: list[str] = []
    metrics = (
        evidence.baseline.median_decode_tokens_per_second,
        evidence.baseline.median_ttft_seconds,
        evidence.baseline.p99_token_latency_seconds,
        evidence.candidate.median_decode_tokens_per_second,
        evidence.candidate.median_ttft_seconds,
        evidence.candidate.p99_token_latency_seconds,
    )
    if any(not math.isfinite(metric) or metric <= 0 for metric in metrics):
        reasons.append("metrics-not-finite-positive")
    if profile.require_physical_evidence and not evidence.physical:
        reasons.append("physical-evidence-missing")
    if not evidence.identities_match:
        reasons.append("identity-mismatch")
    if profile.require_exact_token_parity and not evidence.exact_tokens_match:
        reasons.append("token-parity-failed")
    if profile.require_clean_reset and not evidence.clean_reset:
        reasons.append("cleanup-failed")
    if not reasons:
        speedup = (
            evidence.candidate.median_decode_tokens_per_second
            / evidence.baseline.median_decode_tokens_per_second
        )
        ttft_ratio = (
            evidence.candidate.median_ttft_seconds
            / evidence.baseline.median_ttft_seconds
        )
        tail_ratio = (
            evidence.candidate.p99_token_latency_seconds
            / evidence.baseline.p99_token_latency_seconds
        )
        if speedup < profile.minimum_median_decode_speedup_ratio:
            reasons.append("decode-speedup-below-bound")
        if ttft_ratio > profile.maximum_ttft_regression_ratio:
            reasons.append("ttft-regression-above-bound")
        if tail_ratio > profile.maximum_tail_latency_regression_ratio:
            reasons.append("tail-latency-regression-above-bound")
    return DecodeAdmission(admitted=not reasons, reasons=tuple(reasons))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DecodePolicyError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise DecodePolicyError(
            f"{name} fields differ: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DecodePolicyError(f"{name} must be a boolean")
    return value


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DecodePolicyError(f"{name} must be a non-empty string")
    return value


def _require_hex(value: Any, name: str, expected_length: int) -> str:
    text = _require_non_empty_string(value, name)
    if len(text) != expected_length:
        raise DecodePolicyError(f"{name} has the wrong hexadecimal length")
    if any(character not in "0123456789abcdef" for character in text):
        raise DecodePolicyError(f"{name} must be lowercase hexadecimal")
    return text


def _require_candidate_receipt(value: Any) -> str:
    if value == UNMEASURED_CANDIDATE_RECEIPT:
        return UNMEASURED_CANDIDATE_RECEIPT
    return _require_hex(value, "candidate_receipt_blake3", BLAKE3_HEX_LENGTH)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DecodePolicyError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecodePolicyError(f"{name} must be a non-negative integer")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DecodePolicyError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DecodePolicyError(f"{name} must be finite and positive")
    return result
