"""Host-thread watchdog for the gated MTP adapter.

A timeout does not stop work. The caller must not reuse device state while
that worker remains active. This module does not implement process recovery.
"""

import threading
from collections.abc import Callable
from typing import Any

from .mtp_serving_core import MtpServingPolicyError

ROUND_WATCHDOG_SECONDS = 60.0


class MtpServingRoundTimeout(MtpServingPolicyError):
    """A host watchdog expired, not a negative policy verdict.

    Keep the existing exception base for caller compatibility. The caller
    must exit the process before the supervisor can recover the device.
    """


def run_with_watchdog(work: Callable[[], Any], *, seconds: float, stage: str) -> Any:
    """Return a completed value or forward the exact worker exception."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise MtpServingPolicyError("seconds must be a number")
    if not 0 < seconds <= threading.TIMEOUT_MAX:
        raise MtpServingPolicyError(
            "seconds must be positive, finite, and within the thread timeout limit"
        )
    if not callable(work):
        raise MtpServingPolicyError("work must be callable")
    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["value"] = work()
        # Forward control signals too. An Exception-only catch loses them.
        except BaseException as error:  # noqa: BLE001 - Forward unchanged to the caller.
            outcome["error"] = error

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    if worker.is_alive():
        raise MtpServingRoundTimeout(
            f"speculative stage '{stage}' exceeded the {seconds:g}s watchdog. "
            "The device worker thread remains active. The process must exit."
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
