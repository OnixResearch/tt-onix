"""Pure trace-capture hygiene gates for MTP trace evidence."""

# r[impl tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene]

from __future__ import annotations

from dataclasses import dataclass

REASON_RESIDENT_TRACE = "resident-trace"
REASON_PRIMING = "priming"
REASON_BUFFER_ADDRESS = "buffer-address"


class MtpTraceHygienePolicyError(ValueError):
    """The capture-provenance input violates its closed policy."""


@dataclass(frozen=True)
class TraceCaptureProvenance:
    """What a control truthfully observed around one trace capture.

    A captured trace bakes the device addresses it reads. Capturing while a
    second trace is resident corrupts the captured graph, priming from zeroed
    buffers writes state that later replay slices out of, and reallocating a
    trace-bound buffer silently invalidates every baked address.
    """

    resident_traces_before: int
    warm_execution_completed: bool
    capture_from_zeroed_priming: bool
    trace_bound_addresses_stable: bool


def validate_trace_capture_provenance(
    provenance: TraceCaptureProvenance,
) -> tuple[str, ...]:
    """Return the bounded hygiene reasons for one capture.

    An empty tuple admits the capture. Malformed field types fail closed with
    a policy error instead of a reason string.
    """
    if not isinstance(provenance, TraceCaptureProvenance):
        raise MtpTraceHygienePolicyError(
            "capture provenance must be a TraceCaptureProvenance"
        )
    resident = provenance.resident_traces_before
    if isinstance(resident, bool) or not isinstance(resident, int):
        raise MtpTraceHygienePolicyError(
            "resident trace count must be a non-negative integer"
        )
    if resident < 0:
        raise MtpTraceHygienePolicyError(
            "resident trace count must be a non-negative integer"
        )
    for label, value in (
        ("warm execution flag", provenance.warm_execution_completed),
        ("zeroed priming flag", provenance.capture_from_zeroed_priming),
        ("address stability flag", provenance.trace_bound_addresses_stable),
    ):
        if not isinstance(value, bool):
            raise MtpTraceHygienePolicyError(f"{label} must be boolean")

    reasons: list[str] = []
    if resident > 0:
        reasons.append(REASON_RESIDENT_TRACE)
    if not provenance.warm_execution_completed or (
        provenance.capture_from_zeroed_priming
    ):
        reasons.append(REASON_PRIMING)
    if not provenance.trace_bound_addresses_stable:
        reasons.append(REASON_BUFFER_ADDRESS)
    return tuple(reasons)
