#!/usr/bin/env python3
"""Report the read-only NVMe-to-Blackhole DMA-BUF capability boundary.

This script reads kernel configuration, kernel UAPI, and the tt-kmd UAPI. It
does not open a device, an NVMe namespace, or a DMA-BUF. It exists so a blocked
result is reproducible instead of inferred from a kernel version string.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any

DMABUF_REGISTRATION = re.compile(
    r"\bIORING_(?:REGISTER|UNREGISTER)_DMABUF\b|\bIORING_REGISTER_BUFFERS_DMABUF\b"
)
KMD_EXPORT = re.compile(r"\bTENSTORRENT_IOCTL_EXPORT_TLB_DMABUF\b")


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def kernel_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"state": "not-provided"}
    try:
        text = read_text(path)
    except OSError as error:
        return {"state": "unreadable", "reason": str(error)}
    values = {}
    for name in ("CONFIG_IO_URING", "CONFIG_DMABUF_MOVE_NOTIFY", "CONFIG_DMABUF_HEAPS"):
        match = re.search(rf"^{name}=(.*)$", text, re.MULTILINE)
        values[name] = match.group(1) if match else None
    return {"state": "observed", "path": str(path), "values": values}


def uapi_capability(
    path: Path | None, pattern: re.Pattern[str], label: str
) -> dict[str, Any]:
    if path is None:
        return {"state": "not-provided", "label": label}
    try:
        text = read_text(path)
    except OSError as error:
        return {
            "state": "unreadable",
            "label": label,
            "path": str(path),
            "reason": str(error),
        }
    match = pattern.search(text)
    return {
        "state": "present" if match else "absent",
        "label": label,
        "path": str(path),
        "symbol": match.group(0) if match else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-config", type=Path, default=Path("/proc/config.gz"))
    parser.add_argument("--kernel-uapi", type=Path, required=True)
    parser.add_argument("--kmd-uapi", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    config = kernel_config(arguments.kernel_config)
    kernel = uapi_capability(
        arguments.kernel_uapi,
        DMABUF_REGISTRATION,
        "kernel io_uring DMA-BUF registration",
    )
    kmd = uapi_capability(arguments.kmd_uapi, KMD_EXPORT, "tt-kmd TLB DMA-BUF export")
    reasons = []
    if kernel["state"] != "present":
        reasons.append(
            "the running kernel UAPI has no io_uring DMA-BUF registration symbol"
        )
    if config.get("state") == "observed":
        values = config.get("values", {})
        if values.get("CONFIG_IO_URING") not in {"y", "m"}:
            reasons.append("CONFIG_IO_URING is not enabled")
    if kmd["state"] != "present":
        reasons.append("the packaged tt-kmd UAPI does not expose TLB DMA-BUF export")
    reasons.append("no caller-provisioned disposable NVMe namespace was supplied")
    report = {
        "schema": "onix.nvme-dmabuf-capability.v1",
        "kernel_release": os.uname().release,
        "kernel_config": config,
        "kernel_io_uring_dmabuf": kernel,
        "kmd_tlb_dmabuf_export": kmd,
        "nvme_namespace": {"state": "not-supplied"},
        "transfer": {"state": "not-attempted"},
        "verified_bytes": {"state": "not-attempted"},
        "status": "blocked" if reasons else "ready-for-authorized-probe",
        "reasons": reasons,
        "non_claims": [
            "no device, NVMe namespace, DMA-BUF, or io_uring operation was performed",
            "matching symbols or hashes do not establish a safe transfer",
        ],
    }
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"kernel: {report['kernel_release']}")
        print(f"kernel io_uring DMA-BUF registration: {kernel['state']}")
        print(f"tt-kmd TLB DMA-BUF export: {kmd['state']}")
        for reason in reasons:
            print(f"blocked: {reason}")
        print(f"status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
