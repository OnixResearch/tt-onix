"""Diagnostic read-back check for the MTP head's persistent weight caches.

The draft head is the only component whose weight caches were written on the
0.74 runtime and reused on 0.77. Target tokens are runtime-stable, so the
draft regression localizes to this component's load or forward path. This
probe reads the adapter-owned weights back to the host and compares them
bitwise against a fresh dequantization of the checkpoint shard.

r[impl tenstorrent.qwen38_p150x2.mtp_drafter_probe]
"""

from __future__ import annotations

import torch

WEIGHT_TOLERANCE = 0.0


def evaluate_weight_integrity(
    expected: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    *,
    maximum_abs_diff: float = WEIGHT_TOLERANCE,
) -> dict[str, dict[str, object]]:
    """Compare host-expected tensors with device read-backs per weight."""
    if set(expected) != set(observed):
        raise ValueError(
            f"weight integrity key sets differ: {sorted(set(expected) ^ set(observed))}"
        )
    verdicts: dict[str, dict[str, object]] = {}
    for name, expected_tensor in expected.items():
        observed_tensor = observed[name]
        if expected_tensor.shape != observed_tensor.shape:
            verdicts[name] = {
                "passed": False,
                "max_abs_diff": float("inf"),
                "reason": "shape mismatch",
            }
            continue
        diff = (expected_tensor.float() - observed_tensor.float()).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        verdicts[name] = {
            "passed": max_abs_diff <= maximum_abs_diff,
            "max_abs_diff": max_abs_diff,
            "reason": "" if max_abs_diff <= maximum_abs_diff else "values drifted",
        }
    return verdicts


def collect_head_weight_observations(adapter) -> dict[str, torch.Tensor]:
    """Read the adapter-owned head weights back to one host tensor each."""
    import ttnn

    fc = ttnn.to_torch(
        adapter.fc_weight,
        mesh_composer=ttnn.ConcatMeshToTensor(adapter.mesh, dim=-1),
    )
    observations = {"mtp.fc.weight": fc.squeeze(0)}
    for name, device_tensor in (
        ("mtp.pre_fc_norm_embedding.weight", adapter.pre_embedding_norm),
        ("mtp.pre_fc_norm_hidden.weight", adapter.pre_hidden_norm),
        ("mtp.norm.weight", adapter.final_norm),
    ):
        observed = ttnn.to_torch(
            device_tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(adapter.mesh, dim=0),
        )
        observations[name] = observed[0, 0]
    return observations


def load_mtp_host_weights(model_path):
    """Re-derive the dequantized host head weights from the checkpoint."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    from .mtp_drafter_adapter import _dequantize_fp8_weights
    from .mtp_drafter_core import EXPECTED_MTP_WEIGHT_SHAPES

    index_path = Path(model_path) / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    shard_files = {
        weight_map[name] for name in EXPECTED_MTP_WEIGHT_SHAPES if name in weight_map
    }
    if len(shard_files) != 1:
        raise ValueError("MTP tensors do not share one checkpoint shard")
    shard_path = Path(model_path) / next(iter(shard_files))
    with safe_open(str(shard_path), framework="pt") as shard:
        weights = {name: shard.get_tensor(name) for name in EXPECTED_MTP_WEIGHT_SHAPES}
    return _dequantize_fp8_weights(weights)


def expected_head_weight_tensors(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply the loader's host transforms for the probed weights."""
    return {
        "mtp.fc.weight": weights["mtp.fc.weight"].to(torch.bfloat16).T.contiguous(),
        "mtp.pre_fc_norm_embedding.weight": (
            weights["mtp.pre_fc_norm_embedding.weight"].float() + 1.0
        ).to(torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": (
            weights["mtp.pre_fc_norm_hidden.weight"].float() + 1.0
        ).to(torch.bfloat16),
        "mtp.norm.weight": (weights["mtp.norm.weight"].float() + 1.0).to(
            torch.bfloat16
        ),
    }


def run_head_load_check(adapter) -> dict[str, object]:
    """Read the head weights back and compare them with the checkpoint."""
    host_weights = load_mtp_host_weights(adapter.model_path)
    expected = expected_head_weight_tensors(host_weights)
    observed = collect_head_weight_observations(adapter)
    verdicts = evaluate_weight_integrity(expected, observed)
    failed = sorted(name for name, verdict in verdicts.items() if not verdict["passed"])
    return {
        "diagnostic_only": True,
        "admission_eligible": False,
        "simulated": False,
        "weight_verdicts": verdicts,
        "weights_intact": not failed,
        "drifted_weights": failed,
        "non_claims": [
            "Intact weights do not validate the head forward path.",
            "This probe does not qualify draft acceptance or production behavior.",
        ],
    }
