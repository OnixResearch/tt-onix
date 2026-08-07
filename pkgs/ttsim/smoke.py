#!/usr/bin/env python3
"""Run one bounded TT-NN operation path on a configured simulator."""

# r[impl tenstorrent.ttsim.single_chip_smoke]

import json
from pathlib import Path

import ttnn

DEVICE_ID = 0
TILE_EDGE_ELEMENTS = 32
SMOKE_SHAPE = (1, 1, TILE_EDGE_ELEMENTS, TILE_EDGE_ELEMENTS)
FIRMWARE_CACHE_DIRECTORY = Path.home() / ".cache" / "tt-metal-cache"
FIRMWARE_ARTIFACT_PATTERN = "*/firmware/**/*weakened.elf"


def emit_observation(event: str, **details: object) -> None:
    """Emit one machine-readable simulator observation."""
    payload = {"event": event, **details}
    print(f"TTSIM_OBSERVATION {json.dumps(payload, sort_keys=True)}", flush=True)


def run_smoke() -> None:
    """Open one simulated device, run TT-NN addition, and close it."""
    device = None
    try:
        device = ttnn.open_device(device_id=DEVICE_ID)
        emit_observation("device_opened", device_count=1, device_id=DEVICE_ID)
        firmware_artifacts = sorted(
            FIRMWARE_CACHE_DIRECTORY.glob(FIRMWARE_ARTIFACT_PATTERN)
        )
        if not firmware_artifacts:
            message = "TT-Metal device open produced no firmware ELF artifacts"
            raise RuntimeError(message)
        emit_observation(
            "firmware_compiled",
            artifact_count=len(firmware_artifacts),
            cache_directory=str(FIRMWARE_CACHE_DIRECTORY),
        )

        left = ttnn.ones(
            SMOKE_SHAPE,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        right = ttnn.ones(
            SMOKE_SHAPE,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        result = ttnn.add(left, right)
        ttnn.synchronize_device(device)
        emit_observation(
            "representative_operations_completed",
            operations=["ttnn.ones", "ttnn.add"],
            result_shape=list(result.shape),
        )
    finally:
        if device is not None:
            ttnn.close_device(device)
            emit_observation("clean_close")


if __name__ == "__main__":
    run_smoke()
