"""TTNN callback adapter for bounded Qwen prefill program profiles."""

# r[impl tenstorrent.qwen38_p150x2.prefill_program_profile]

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from .prefill_profile_core import (
    PrefillProfileError,
    RealtimeProgramObservation,
    summarize_prefill_programs,
)

# Package-matched RT profiler tests use this drain after device quiescence.
DEFAULT_RECEIVER_DRAIN_SECONDS = 0.5


class PrefillProfilerAdapterError(RuntimeError):
    """The TTNN real-time profiler adapter cannot produce a valid profile."""


class RealtimePrefillProfiler:
    """Collect one serialized prefill region from the TTNN receiver thread."""

    def __init__(
        self,
        device_api: Any,
        *,
        expected_chip_ids: Sequence[int],
        receiver_drain_seconds: float = DEFAULT_RECEIVER_DRAIN_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(receiver_drain_seconds, bool)
            or not isinstance(receiver_drain_seconds, (int, float))
            or not math.isfinite(float(receiver_drain_seconds))
            or float(receiver_drain_seconds) <= 0.0
        ):
            raise PrefillProfilerAdapterError(
                "receiver drain seconds must be finite and positive"
            )
        required_api = (
            "IsProgramRealtimeProfilerActive",
            "RegisterProgramRealtimeProfilerCallback",
            "UnregisterProgramRealtimeProfilerCallback",
        )
        missing_api = [name for name in required_api if not hasattr(device_api, name)]
        if missing_api:
            raise PrefillProfilerAdapterError(
                f"TTNN real-time profiler API is unavailable: {', '.join(missing_api)}"
            )
        if not device_api.IsProgramRealtimeProfilerActive():
            raise PrefillProfilerAdapterError(
                "TTNN real-time profiler is inactive on the physical mesh"
            )

        self._device_api = device_api
        self._expected_chip_ids = tuple(expected_chip_ids)
        self._receiver_drain_seconds = float(receiver_drain_seconds)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._collecting = False
        self._closed = False
        self._callback_error: str | None = None
        self._observations: list[RealtimeProgramObservation] = []
        self._handle = device_api.RegisterProgramRealtimeProfilerCallback(
            self._collect_record
        )

    @property
    def receiver_drain_seconds(self) -> float:
        return self._receiver_drain_seconds

    def begin(self) -> None:
        """Drain prior callbacks, then start one empty profile region."""
        with self._lock:
            if self._closed:
                raise PrefillProfilerAdapterError("prefill profiler is closed")
            if self._collecting:
                raise PrefillProfilerAdapterError(
                    "prefill profiler region is already active"
                )
            self._observations.clear()
            self._callback_error = None
            self._collecting = True
        self._sleep(self._receiver_drain_seconds)
        with self._lock:
            if self._closed:
                raise PrefillProfilerAdapterError("prefill profiler closed during drain")
            self._observations.clear()
            self._callback_error = None

    def finish(self) -> dict[str, object]:
        """Drain the receiver, close the region, and summarize its records."""
        with self._lock:
            if self._closed:
                raise PrefillProfilerAdapterError("prefill profiler is closed")
            if not self._collecting:
                raise PrefillProfilerAdapterError("prefill profiler region is not active")
        self._sleep(self._receiver_drain_seconds)
        with self._lock:
            self._collecting = False
            callback_error = self._callback_error
            observations = tuple(self._observations)
            self._observations.clear()
            self._callback_error = None
        if callback_error is not None:
            raise PrefillProfilerAdapterError(
                f"real-time profiler callback failed: {callback_error}"
            )
        try:
            return summarize_prefill_programs(
                observations,
                expected_chip_ids=self._expected_chip_ids,
            )
        except PrefillProfileError as error:
            raise PrefillProfilerAdapterError(
                f"real-time profiler records are invalid: {error}"
            ) from error

    def cancel(self) -> None:
        """Discard one failed region without publishing partial evidence."""
        with self._lock:
            self._collecting = False
            self._observations.clear()
            self._callback_error = None

    def close(self) -> None:
        """Stop collection and unregister the vendor callback exactly once."""
        with self._lock:
            if self._closed:
                return
            self._collecting = False
            self._observations.clear()
            self._closed = True
        try:
            self._device_api.UnregisterProgramRealtimeProfilerCallback(self._handle)
        except (OSError, RuntimeError, ValueError) as error:
            raise PrefillProfilerAdapterError(
                "failed to unregister the real-time profiler callback"
            ) from error

    def _collect_record(self, record: Any) -> None:
        with self._lock:
            if not self._collecting or self._closed:
                return
            try:
                observation = RealtimeProgramObservation(
                    runtime_id=record.runtime_id,
                    chip_id=record.chip_id,
                    start_timestamp=record.start_timestamp,
                    end_timestamp=record.end_timestamp,
                    frequency_cycles_per_ns=record.frequency,
                    kernel_sources=tuple(record.kernel_sources),
                )
            except (AttributeError, TypeError, ValueError) as error:
                self._callback_error = f"{type(error).__name__}: {error}"
                return
            self._observations.append(observation)
