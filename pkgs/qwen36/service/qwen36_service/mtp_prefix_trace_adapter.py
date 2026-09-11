"""Request-owned traces for fixed-address GDN prefix restoration."""

from collections.abc import Callable
from typing import Any

import ttnn

TRACE_COMMAND_QUEUE_ID = 0
TRACE_BLOCKING_REPLAY = False


class MtpPrefixTraceError(RuntimeError):
    """A prefix trace cannot preserve its request-owned lifecycle."""


class MtpPrefixTraces:
    """Borrow prefix buffers until close. The caller owns and restores model state.

    Capture these small traces after the verifier and before the draft trace.
    Release them before the caller releases any borrowed prefix or target buffer.
    """

    def __init__(
        self,
        *,
        mesh: Any,
        prefixes: tuple[Any, ...],
        restore_prefix: Callable[[Any], None],
        binding_ids: Callable[[], tuple[int, ...]],
    ) -> None:
        if not prefixes:
            raise MtpPrefixTraceError("prefix traces require at least one prefix")
        self._mesh = mesh
        self._prefixes = prefixes
        self._restore_prefix = restore_prefix
        self._binding_ids = binding_ids
        self._captured_bindings: tuple[int, ...] = ()
        self._trace_ids: list[Any] = []

    def capture(self) -> None:
        if self._trace_ids:
            raise MtpPrefixTraceError("prefix traces are already captured")
        self._captured_bindings = self._binding_ids()
        if not self._captured_bindings:
            raise MtpPrefixTraceError("prefix trace bindings are empty")
        try:
            for prefix in self._prefixes:
                # Compile every restore operation before recording its trace.
                self._restore_prefix(prefix)
                ttnn.synchronize_device(self._mesh)
                trace_id = ttnn.begin_trace_capture(
                    self._mesh, cq_id=TRACE_COMMAND_QUEUE_ID
                )
                self._trace_ids.append(trace_id)
                try:
                    self._restore_prefix(prefix)
                finally:
                    ttnn.end_trace_capture(
                        self._mesh, trace_id, cq_id=TRACE_COMMAND_QUEUE_ID
                    )
                ttnn.synchronize_device(self._mesh)
            if self._binding_ids() != self._captured_bindings:
                raise MtpPrefixTraceError(
                    "prefix trace bindings changed during capture"
                )
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
            self.close()
            raise

    def replay(self, accepted: int) -> None:
        if (
            isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or not 0 <= accepted < len(self._trace_ids)
            or len(self._trace_ids) != len(self._prefixes)
        ):
            raise MtpPrefixTraceError("prefix trace index or capture state is invalid")
        if self._binding_ids() != self._captured_bindings:
            raise MtpPrefixTraceError("prefix trace bindings changed before replay")
        ttnn.execute_trace(
            self._mesh,
            self._trace_ids[accepted],
            cq_id=TRACE_COMMAND_QUEUE_ID,
            blocking=TRACE_BLOCKING_REPLAY,
        )
        ttnn.synchronize_device(self._mesh)

    def close(self) -> None:
        while self._trace_ids:
            ttnn.release_trace(self._mesh, self._trace_ids[-1])
            self._trace_ids.pop()
        self._captured_bindings = ()
