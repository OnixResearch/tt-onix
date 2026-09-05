"""Host-thread checks. A watchdog timeout is not device cancellation."""

import threading
import unittest
from unittest.mock import Mock, patch

from qwen36_service import mtp_watchdog_adapter as watchdog_module
from qwen36_service.mtp_serving_core import MtpServingPolicyError

WORKER_WAIT_SECONDS = 5.0
TIMEOUT_PROBE_SECONDS = 0.05


class WorkerControlSignal(BaseException):
    """A synthetic process-control signal that must reach the caller."""


class MtpWatchdogTests(unittest.TestCase):
    def test_returns_values_without_truthiness_conversion(self):
        for value in (object(), None, False, 0):
            with self.subTest(value=value):
                observed = watchdog_module.run_with_watchdog(
                    lambda value=value: value,
                    seconds=WORKER_WAIT_SECONDS,
                    stage="completed",
                )
                self.assertIs(observed, value)

    def test_propagates_worker_errors_and_control_signals_unchanged(self):
        errors = (
            RuntimeError("worker failed"),
            KeyboardInterrupt(),
            SystemExit("worker exited"),
            WorkerControlSignal(),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):

                def fail(error=error):
                    raise error

                with self.assertRaises(type(error)) as raised:
                    watchdog_module.run_with_watchdog(
                        fail, seconds=WORKER_WAIT_SECONDS, stage="failed"
                    )
                self.assertIs(raised.exception, error)

    def test_timeout_leaves_work_alive_until_released(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocked():
            started.set()
            release.wait()
            finished.set()

        try:
            with self.assertRaises(watchdog_module.MtpServingRoundTimeout) as raised:
                watchdog_module.run_with_watchdog(
                    blocked, seconds=TIMEOUT_PROBE_SECONDS, stage="blocked-worker"
                )
            self.assertIn("blocked-worker", str(raised.exception))
            self.assertTrue(started.wait(WORKER_WAIT_SECONDS))
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            self.assertTrue(finished.wait(WORKER_WAIT_SECONDS))

    def test_rejects_invalid_deadlines_before_work_starts(self):
        invalid_seconds = (
            True,
            None,
            "1",
            0,
            -1,
            float("nan"),
            float("inf"),
            threading.TIMEOUT_MAX + 1,
        )
        for seconds in invalid_seconds:
            with self.subTest(seconds=seconds):
                work = Mock()
                with self.assertRaises(MtpServingPolicyError):
                    watchdog_module.run_with_watchdog(
                        work, seconds=seconds, stage="invalid-timeout"
                    )
                work.assert_not_called()

    def test_rejects_non_callable_work(self):
        with self.assertRaisesRegex(MtpServingPolicyError, "callable"):
            watchdog_module.run_with_watchdog(
                None, seconds=WORKER_WAIT_SECONDS, stage="invalid-work"
            )

    def test_thread_start_error_is_not_a_success(self):
        work = Mock()
        error = OSError("thread creation failed")
        with patch.object(threading.Thread, "start", side_effect=error):
            with self.assertRaises(OSError) as raised:
                watchdog_module.run_with_watchdog(
                    work, seconds=WORKER_WAIT_SECONDS, stage="thread-start"
                )
        self.assertIs(raised.exception, error)
        work.assert_not_called()


if __name__ == "__main__":
    unittest.main()
