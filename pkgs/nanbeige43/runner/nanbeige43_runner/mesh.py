# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.nanbeige43_p150x2.mesh]
"""Mesh admission.

`p150x2` is the production mesh. It is fail-closed: PCIe devices exactly
[0, 1], `FABRIC_1D`, a `1x2` mesh, and both devices open. The check mirrors
`qwen36_service/server.py`.
`smoke-1x1` is the hardware-neutral smoke mode required by AGENTS.md: one
device, no fabric, and the same model code. It is the mode TTsim runs, but it
contains no simulator branch.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import ttnn

EXPECTED_PCIE_DEVICE_IDS = [0, 1]


@dataclass(frozen=True)
class MeshSpec:
    name: str
    shape: tuple[int, int]
    fabric: bool
    pcie_device_ids: list[int] | None


MESHES = {
    "p150x2": MeshSpec("p150x2", (1, 2), True, EXPECTED_PCIE_DEVICE_IDS),
    "smoke-1x1": MeshSpec("smoke-1x1", (1, 1), False, None),
}


@contextlib.contextmanager
def open_mesh(spec: MeshSpec, trace_region_size: int = 0):
    """`trace_region_size` 0 keeps the runtime default (dynamic trace allocation)."""
    observed_pcie = None
    if spec.pcie_device_ids is not None:
        observed_pcie = list(ttnn.get_pcie_device_ids())
        if observed_pcie != spec.pcie_device_ids:
            raise SystemExit(
                f"expected physical PCIe devices {spec.pcie_device_ids}, got {observed_pcie}"
            )
    if spec.fabric:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )
    mesh_device = None
    try:
        options = {"trace_region_size": trace_region_size} if trace_region_size else {}
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*spec.shape), num_command_queues=1, **options
        )
        expected_devices = spec.shape[0] * spec.shape[1]
        if mesh_device.get_num_devices() != expected_devices:
            raise RuntimeError(
                f"{spec.name} mesh opened {mesh_device.get_num_devices()} devices, expected {expected_devices}"
            )
        mesh_device.enable_program_cache()
        yield (
            mesh_device,
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "fabric": "FABRIC_1D" if spec.fabric else "disabled",
                "trace_region_size": trace_region_size,
                "pcie_device_ids": observed_pcie,
                "device_ids": [
                    int(device_id) for device_id in mesh_device.get_device_ids()
                ],
            },
        )
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        if spec.fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
