"""Pure validation and aggregation for Qwen prefill program profiles."""

# r[impl tenstorrent.qwen38_p150x2.prefill_program_profile]
# r[impl tenstorrent.qwen38_p150x2.host_timeline_profile]

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

PREFILL_PROGRAM_PROFILE_SCHEMA_VERSION = 1
MEASUREMENT_MODE_ADMISSION = "admission"
MEASUREMENT_MODE_DIAGNOSTIC_HOST_TIMELINE = "diagnostic-host-timeline"
MEASUREMENT_MODE_DIAGNOSTIC_PREFILL_PROFILE = "diagnostic-prefill-profile"


class PrefillProfileError(ValueError):
    """A prefill program profile violates its deterministic contract."""


def select_measurement_mode(
    *, prefill_program_profiled: bool, host_timeline_profiled: bool
) -> tuple[str, bool]:
    """Return the response mode and admission eligibility for one request."""
    if prefill_program_profiled and host_timeline_profiled:
        raise PrefillProfileError("prefill program and host timeline profiles conflict")
    if prefill_program_profiled:
        return MEASUREMENT_MODE_DIAGNOSTIC_PREFILL_PROFILE, False
    if host_timeline_profiled:
        return MEASUREMENT_MODE_DIAGNOSTIC_HOST_TIMELINE, False
    return MEASUREMENT_MODE_ADMISSION, True


@dataclass(frozen=True)
class RealtimeProgramObservation:
    """Vendor-neutral observation from one chip and one program execution."""

    runtime_id: int
    chip_id: int
    start_timestamp: int
    end_timestamp: int
    frequency_cycles_per_ns: float
    kernel_sources: tuple[str, ...]


def summarize_prefill_programs(
    observations: Sequence[RealtimeProgramObservation],
    *,
    expected_chip_ids: Sequence[int],
) -> dict[str, object]:
    """Validate observations and return a deterministic per-signature summary."""
    admitted_chips = _validate_expected_chip_ids(expected_chip_ids)
    if not observations:
        raise PrefillProfileError("prefill profile must contain program observations")

    totals_by_chip = {
        chip_id: {"record_count": 0, "total_duration_ns": 0.0}
        for chip_id in admitted_chips
    }
    signature_rows: dict[tuple[str, ...], dict[int, dict[str, float | int]]] = {}
    runtime_ids: set[int] = set()

    for observation in observations:
        duration_ns, signature = _validate_observation(observation, admitted_chips)
        runtime_ids.add(observation.runtime_id)
        chip_total = totals_by_chip[observation.chip_id]
        chip_total["record_count"] = int(chip_total["record_count"]) + 1
        chip_total["total_duration_ns"] = (
            float(chip_total["total_duration_ns"]) + duration_ns
        )

        by_chip = signature_rows.setdefault(signature, {})
        signature_chip = by_chip.setdefault(
            observation.chip_id,
            {
                "maximum_duration_ns": 0.0,
                "record_count": 0,
                "total_duration_ns": 0.0,
            },
        )
        signature_chip["record_count"] = int(signature_chip["record_count"]) + 1
        signature_chip["total_duration_ns"] = (
            float(signature_chip["total_duration_ns"]) + duration_ns
        )
        signature_chip["maximum_duration_ns"] = max(
            float(signature_chip["maximum_duration_ns"]),
            duration_ns,
        )

    signatures = []
    for signature, by_chip in signature_rows.items():
        chip_rows = [
            {
                "chip_id": chip_id,
                "maximum_duration_ns": float(by_chip[chip_id]["maximum_duration_ns"]),
                "record_count": int(by_chip[chip_id]["record_count"]),
                "total_duration_ns": float(by_chip[chip_id]["total_duration_ns"]),
            }
            for chip_id in sorted(by_chip)
        ]
        signatures.append({
            "kernel_sources": list(signature),
            "maximum_chip_total_duration_ns": max(
                row["total_duration_ns"] for row in chip_rows
            ),
            "record_count": sum(row["record_count"] for row in chip_rows),
            "by_chip": chip_rows,
        })
    signatures.sort(
        key=lambda row: (
            -float(row["maximum_chip_total_duration_ns"]),
            tuple(row["kernel_sources"]),
        )
    )

    return {
        "schema_version": PREFILL_PROGRAM_PROFILE_SCHEMA_VERSION,
        "record_count": len(observations),
        "runtime_ids": sorted(runtime_ids),
        "by_chip": [
            {
                "chip_id": chip_id,
                "record_count": int(totals_by_chip[chip_id]["record_count"]),
                "total_duration_ns": float(
                    totals_by_chip[chip_id]["total_duration_ns"]
                ),
            }
            for chip_id in admitted_chips
        ],
        "signatures": signatures,
    }


def _validate_expected_chip_ids(expected_chip_ids: Sequence[int]) -> tuple[int, ...]:
    if not expected_chip_ids:
        raise PrefillProfileError("expected chip IDs must not be empty")
    admitted_chips = tuple(expected_chip_ids)
    if any(
        isinstance(chip_id, bool) or not isinstance(chip_id, int) or chip_id < 0
        for chip_id in admitted_chips
    ):
        raise PrefillProfileError("expected chip IDs must be non-negative integers")
    if len(set(admitted_chips)) != len(admitted_chips):
        raise PrefillProfileError("expected chip IDs must be unique")
    if admitted_chips != tuple(sorted(admitted_chips)):
        raise PrefillProfileError("expected chip IDs must be sorted")
    return admitted_chips


def _validate_observation(
    observation: RealtimeProgramObservation,
    admitted_chips: tuple[int, ...],
) -> tuple[float, tuple[str, ...]]:
    integer_fields = (
        ("runtime ID", observation.runtime_id),
        ("chip ID", observation.chip_id),
        ("start timestamp", observation.start_timestamp),
        ("end timestamp", observation.end_timestamp),
    )
    for name, value in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PrefillProfileError(f"{name} must be a non-negative integer")
    if observation.runtime_id == 0:
        raise PrefillProfileError("runtime ID must be positive")
    if observation.chip_id not in admitted_chips:
        raise PrefillProfileError(
            f"chip ID {observation.chip_id} is outside the admitted profile"
        )
    if observation.end_timestamp <= observation.start_timestamp:
        raise PrefillProfileError("program end timestamp must follow its start")
    frequency = observation.frequency_cycles_per_ns
    if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
        raise PrefillProfileError("program frequency must be numeric")
    if not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        raise PrefillProfileError("program frequency must be finite and positive")
    if not observation.kernel_sources:
        raise PrefillProfileError("program kernel sources must not be empty")
    if any(
        not isinstance(source, str) or not source
        for source in observation.kernel_sources
    ):
        raise PrefillProfileError("program kernel sources must be non-empty strings")
    signature = tuple(sorted(set(observation.kernel_sources)))
    duration_ns = (observation.end_timestamp - observation.start_timestamp) / float(
        frequency
    )
    if not math.isfinite(duration_ns) or duration_ns <= 0.0:
        raise PrefillProfileError("program duration must be finite and positive")
    return duration_ns, signature
