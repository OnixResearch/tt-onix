# r[impl tenstorrent.rwkv7_p150a_finetuning.compiler]
# r[impl tenstorrent.rwkv7_p150a_finetuning.device]
# Focused direct-lowering probes for the single-P150A RWKV-7 LoRA bring-up.
# Runs inside the pinned TT-XLA runtime container. Prints one JSON result.
#
# The WKV7 equations mirror the CPU authority in
# pkgs/rwkv7-lora-reference/wkv7.py. The probe recomputes the CPU reference
# in FP32 inside the same process so every comparison uses identical seeded
# inputs.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

EXPECTED_VISIBLE_DEVICE_COUNT = 2
SELECTED_DEVICE_ORDINAL = 0
HLO_OP_PATTERN = re.compile(r"stablehlo\.([a-z_0-9]+)")
SEED = 7
LOSS_IGNORE_INDEX = -100
# Compile options match the pinned TT-Blacksmith training recipe for this
# runtime revision.
TT_COMPILE_OPTIONS = {
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_legacy_compile": True,
    "tt_use_aot_autograd": False,
}


def emit(status: str, probe: str, **extra) -> None:
    payload = {"status": status, "probe": probe}
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True))


def wkv7_forward(r, w, k, v, a, b, state):
    """Paper-matched naive WKV7 forward over a bounded static sequence."""
    batch, time_steps, heads, width = r.shape
    wactivated = (-(w.exp())).exp()
    out = r.new_zeros((batch, time_steps, heads, width))
    current = state
    for t in range(time_steps):
        kk = k[:, t, :]
        rr = r[:, t, :]
        vv = v[:, t, :]
        aa = a[:, t, :]
        bb = b[:, t, :]
        current = (
            current * wactivated[:, t, :, None, :]
            + torch.einsum("bhik,bhk,bhj->bhij", current, aa, bb)
            + torch.einsum("bhj,bhi->bhij", kk, vv)
        )
        out[:, t, :] = torch.einsum("bhj,bhij->bhi", rr, current)
    return out, current


def build_wkv7_inputs(shape, dtype, generator):
    batch = shape["batch"]
    time_steps = shape["time"]
    heads = shape["heads"]
    width = shape["width"]
    dims = (batch, time_steps, heads, width)
    inputs = {
        "r": torch.randn(dims, generator=generator),
        "w": torch.randn(dims, generator=generator) * 0.2,
        "k": torch.randn(dims, generator=generator),
        "v": torch.randn(dims, generator=generator),
        "a": torch.sigmoid(torch.randn(dims, generator=generator)),
        "b": torch.randn(dims, generator=generator),
        "state": torch.zeros((batch, heads, width, width)),
    }
    return {name: tensor.to(dtype) for name, tensor in inputs.items()}


class Wkv7ForwardModule(torch.nn.Module):
    def forward(self, r, w, k, v, a, b, state):
        return wkv7_forward(r, w, k, v, a, b, state)


class Wkv7BackwardModule(torch.nn.Module):
    """Forward plus scalar loss so autograd covers every WKV7 input."""

    def forward(self, r, w, k, v, a, b, state):
        out, final_state = wkv7_forward(r, w, k, v, a, b, state)
        return out.sum() + final_state.sum()


class LoraProjectionModule(torch.nn.Module):
    """Frozen base projection plus trainable rank-limited adapter pair."""

    def __init__(self, hidden, rank, dtype, generator):
        super().__init__()
        self.base = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        with torch.no_grad():
            self.base.weight.copy_(torch.randn(hidden, hidden, generator=generator) * 0.02)
        self.base.weight.requires_grad_(False)
        self.lora_a = torch.nn.Parameter(torch.randn(rank, hidden, generator=generator) * 0.02)
        self.lora_b = torch.nn.Parameter(torch.zeros(hidden, rank, dtype=dtype))

    def forward(self, value):
        adapter = value @ self.lora_a.t() @ self.lora_b.t()
        return self.base(value) + adapter


class MaskedLossModule(torch.nn.Module):
    def forward(self, logits, labels):
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        return torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, ignore_index=LOSS_IGNORE_INDEX
        )


def summarize_hlo(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    ops = HLO_OP_PATTERN.findall(text)
    histogram = {}
    for op in ops:
        histogram[op] = histogram.get(op, 0) + 1
    return {"stablehlo_op_count": len(ops), "op_histogram": histogram}


def prepare_device(hlo_path):
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_SAVE_TENSORS_FILE"] = hlo_path
    os.environ["XLA_SAVE_TENSORS_FMT"] = "hlo"
    xr.set_device_type("TT")
    supported = xm.get_xla_supported_devices("TT")
    if len(supported) != EXPECTED_VISIBLE_DEVICE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_VISIBLE_DEVICE_COUNT} visible TT devices, saw {len(supported)}"
        )
    device = xm.xla_device(SELECTED_DEVICE_ORDINAL)
    return device, supported


def max_abs_deviation(device_tensor, cpu_tensor):
    moved = device_tensor.detach().to("cpu").to(torch.float32)
    reference = cpu_tensor.detach().to(torch.float32)
    if moved.shape != reference.shape:
        raise RuntimeError(f"shape mismatch: {tuple(moved.shape)} vs {tuple(reference.shape)}")
    if not torch.isfinite(moved).all():
        raise RuntimeError("device tensor contains a nonfinite value")
    return float((moved - reference).abs().max().item())


def check_tolerance(deviation, tolerance):
    # atol-only gate: named absolute limits keep BF16 comparisons honest.
    return deviation <= tolerance["atol"]


def run_wkv7_forward(args):
    import torch_xla

    profile = args.profile
    shape = profile["probe"][args.shape_kind]
    dtype = torch.bfloat16 if profile["precision"]["model_dtype"] == "bf16" else torch.float32
    tolerance = profile["tolerances"]["wkv7_output"]
    state_tolerance = profile["tolerances"]["wkv7_state"]

    generator = torch.Generator().manual_seed(SEED)
    cpu_inputs = build_wkv7_inputs(shape, torch.float32, generator)
    with torch.no_grad():
        cpu_out, cpu_state = wkv7_forward(*cpu_inputs.values())

    device, supported = prepare_device(args.hlo)
    module = Wkv7ForwardModule().to(dtype).to(device)
    compile_start = time.monotonic()
    compiled = torch.compile(module, backend="tt", options=TT_COMPILE_OPTIONS)
    device_inputs = [
        tensor.to(device).requires_grad_(False) for tensor in cpu_inputs.values()
    ]
    device_inputs = [tensor.to(dtype) for tensor in device_inputs]
    with torch.no_grad():
        device_out, device_state = compiled(*device_inputs)
    torch_xla.sync(wait=True)
    compile_seconds = time.monotonic() - compile_start

    out_deviation = max_abs_deviation(device_out, cpu_out)
    state_deviation = max_abs_deviation(device_state, cpu_state)
    result = {
        "device": {
            "type": "TT",
            "ordinal": SELECTED_DEVICE_ORDINAL,
            "visible_count": len(supported),
            "supported_devices": supported,
        },
        "compile_seconds": compile_seconds,
        "shape": shape,
        "dtype": str(dtype).replace("torch.", ""),
        "deviations": {"wkv7_output": out_deviation, "wkv7_state": state_deviation},
        "tolerances_satisfied": check_tolerance(out_deviation, tolerance)
        and check_tolerance(state_deviation, state_tolerance),
        "finite": True,
        "hlo": summarize_hlo(args.hlo),
    }
    return result


def run_wkv7_backward(args):
    import torch_xla

    profile = args.profile
    shape = profile["probe"][args.shape_kind]
    dtype = torch.bfloat16 if profile["precision"]["model_dtype"] == "bf16" else torch.float32
    tolerance = profile["tolerances"]["adapter_gradient"]

    generator = torch.Generator().manual_seed(SEED)
    cpu_inputs = build_wkv7_inputs(shape, torch.float32, generator)
    cpu_leaves = [tensor.detach().clone().requires_grad_(True) for tensor in cpu_inputs.values()]
    cpu_loss = Wkv7BackwardModule()(*cpu_leaves)
    cpu_loss.backward()
    cpu_gradients = {
        name: leaf.grad.detach().clone()
        for name, leaf in zip(cpu_inputs.keys(), cpu_leaves)
    }

    device, supported = prepare_device(args.hlo)
    module = Wkv7BackwardModule().to(dtype).to(device)
    compile_start = time.monotonic()
    compiled = torch.compile(module, backend="tt", options=TT_COMPILE_OPTIONS)
    device_leaves = [
        tensor.to(device).to(dtype).requires_grad_(True) for tensor in cpu_inputs.values()
    ]
    device_loss = compiled(*device_leaves)
    device_loss.backward()
    torch_xla.sync(wait=True)
    compile_seconds = time.monotonic() - compile_start

    deviations = {}
    for name, leaf in zip(cpu_inputs.keys(), device_leaves):
        if leaf.grad is None:
            raise RuntimeError(f"input '{name}' received no gradient")
        deviations[name] = max_abs_deviation(leaf.grad, cpu_gradients[name])
    result = {
        "device": {
            "type": "TT",
            "ordinal": SELECTED_DEVICE_ORDINAL,
            "visible_count": len(supported),
            "supported_devices": supported,
        },
        "compile_seconds": compile_seconds,
        "shape": shape,
        "dtype": str(dtype).replace("torch.", ""),
        "loss_deviation": max_abs_deviation(device_loss, cpu_loss),
        "deviations": deviations,
        "tolerances_satisfied": all(
            check_tolerance(value, tolerance) for value in deviations.values()
        ),
        "finite": True,
        "hlo": summarize_hlo(args.hlo),
    }
    return result


def run_probe(args):
    handlers = {
        "wkv7-forward": run_wkv7_forward,
        "wkv7-backward": run_wkv7_backward,
    }
    handler = handlers.get(args.probe)
    if handler is None:
        raise RuntimeError(f"unknown probe '{args.probe}'")
    return handler(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--shape-kind", default="focused", choices=["focused", "representative"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--hlo", required=True)
    args = parser.parse_args()
    with open(args.profile, encoding="utf-8") as handle:
        args.profile = json.load(handle)
    try:
        result = run_probe(args)
        emit("pass", args.probe, **result)
        return 0
    except Exception as error:
        emit("fail", args.probe, detail=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
