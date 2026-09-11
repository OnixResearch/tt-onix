"""Bound host waits for speculative device stages outside the pure core."""

import math
import os
import threading
from collections.abc import Callable
from typing import NoReturn, TypeVar

from .mtp_serving_core import MtpServingPolicyError

StageResult = TypeVar("StageResult")
STANDARD_ERROR_FILENO = 2
DEVICE_TIMEOUT_EXIT_CODE = os.EX_TEMPFAIL


class MtpServingRoundTimeout(RuntimeError):
    """A device worker remains active. The process must terminate."""


def run_with_watchdog(
    work: Callable[[], StageResult], *, seconds: float, stage: str
) -> StageResult:
    """Return the stage result or reject a timeout without reusing the device.

    This host-thread bound cannot interrupt a vendor call that holds the GIL.
    The process supervisor must also enforce an external runtime bound.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise MtpServingPolicyError("seconds must be a number")
    if not math.isfinite(seconds) or seconds <= 0:
        raise MtpServingPolicyError("seconds must be positive and finite")
    if not callable(work):
        raise MtpServingPolicyError("work must be callable")
    if not isinstance(stage, str) or not stage:
        raise MtpServingPolicyError("stage must be a nonempty string")
    outcome = {}

    def run() -> None:
        try:
            outcome["value"] = work()
        except BaseException as error:  # noqa: BLE001 -- propagate worker exits and interrupts to the caller
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    if worker.is_alive():
        raise MtpServingRoundTimeout(
            f"speculative stage '{stage}' exceeded the {seconds:.0f}s watchdog; "
            "the process must exit without device cleanup"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def run_device_stage(
    work: Callable[[], StageResult],
    *,
    seconds: float,
    stage: str,
    exit_process: Callable[[int], NoReturn] = os._exit,
) -> StageResult:
    """Terminate before cleanup can race an abandoned device worker."""
    try:
        return run_with_watchdog(work, seconds=seconds, stage=stage)
    except MtpServingRoundTimeout as error:
        try:
            os.write(STANDARD_ERROR_FILENO, f"{error}\n".encode())
        finally:
            exit_process(DEVICE_TIMEOUT_EXIT_CODE)
        raise AssertionError("process termination returned")
