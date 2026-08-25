"""Minimal systemd notification adapter for service readiness."""

# r[impl tenstorrent.qwen38_p150x2.startup_readiness]

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from typing import Any

NOTIFY_SOCKET_ENVIRONMENT_KEY = "NOTIFY_SOCKET"
ABSTRACT_SOCKET_PREFIX = "@"
ABSTRACT_SOCKET_ADDRESS_PREFIX = "\0"
READY_MESSAGE_PREFIX = "READY=1\nSTATUS="


class SystemdNotifyError(RuntimeError):
    """The required systemd readiness notification failed."""


def notify_ready(
    status: str,
    *,
    required: bool,
    environment: Mapping[str, str] = os.environ,
    socket_factory: Callable[..., Any] = socket.socket,
) -> bool:
    if not isinstance(status, str) or not status.strip():
        raise SystemdNotifyError("systemd readiness status must be non-empty")
    socket_path = environment.get(NOTIFY_SOCKET_ENVIRONMENT_KEY)
    if not socket_path:
        if required:
            raise SystemdNotifyError("required systemd notification socket is absent")
        return False
    address = (
        ABSTRACT_SOCKET_ADDRESS_PREFIX + socket_path[len(ABSTRACT_SOCKET_PREFIX) :]
        if socket_path.startswith(ABSTRACT_SOCKET_PREFIX)
        else socket_path
    )
    payload = (READY_MESSAGE_PREFIX + status).encode("utf-8")
    try:
        notify_socket = socket_factory(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            notify_socket.connect(address)
            sent_bytes = notify_socket.send(payload)
        finally:
            notify_socket.close()
    except OSError as error:
        raise SystemdNotifyError("systemd readiness notification failed") from error
    if sent_bytes != len(payload):
        raise SystemdNotifyError("systemd readiness notification was incomplete")
    return True
