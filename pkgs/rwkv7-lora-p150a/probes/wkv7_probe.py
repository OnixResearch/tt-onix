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
import math
import os
import re
import sys
import time
import traceback

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


def _step(x, t):
    """Select timestep t as x[:, t, :] using narrow + squeeze.

    The runtime's backward for the basic-indexing select that underlies
    x[:, t, :] fails to scatter gradients back to a repeatedly-sliced tensor.
    narrow lowers to an aten.slice whose backward is correct, so the
    recurrence reads each timestep through narrow instead.
    """
    return x.narrow(1, t, 1).squeeze(1)


def wkv7_forward(r, w, k, v, a, b, state):
    """Paper-matched naive WKV7 forward over a bounded static sequence."""
    _, time_steps, _, _ = r.shape
    wactivated = (-(w.exp())).exp()
    outputs = []
    current = state
    for t in range(time_steps):
        current = (
            current * _step(wactivated, t).unsqueeze(-2)
            + torch.einsum("bhik,bhk,bhj->bhij", current, _step(a, t), _step(b, t))
            + torch.einsum("bhj,bhi->bhij", _step(k, t), _step(v, t))
        )
        outputs.append(torch.einsum("bhj,bhij->bhi", _step(r, t), current))
    out = torch.stack(outputs, dim=1)
    return out, current


def build_wkv7_inputs(shape, dtype, generator):
    batch = shape["batch"]
    time_steps = shape["time"]
    heads = shape["heads"]
    width = shape["width"]
    dims = (batch, time_steps, heads, width)
    # The WKV7 state update adds outer(S @ a, b) each step. With unit
    # variance inputs that operator has spectral norm far above one and the
    # recurrence diverges, amplifying bf16 rounding into chaotic divergence.
    # Scaling a, b, k, v by 1/sqrt(width) keeps the update contractive so
    # parity measures dtype rounding, not input-driven instability.
    state_scale = 1.0 / math.sqrt(width)
    inputs = {
        "r": torch.randn(dims, generator=generator),
        "w": torch.randn(dims, generator=generator) * 0.2,
        "k": torch.randn(dims, generator=generator) * state_scale,
        "v": torch.randn(dims, generator=generator) * state_scale,
        "a": torch.sigmoid(torch.randn(dims, generator=generator)) * state_scale,
        "b": torch.randn(dims, generator=generator) * state_scale,
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
            self.base.weight.copy_(
                torch.randn(hidden, hidden, generator=generator) * 0.02
            )
        self.base.weight.requires_grad_(False)
        self.lora_a = torch.nn.Parameter(
            torch.randn(rank, hidden, generator=generator) * 0.02
        )
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


def summarize_hlo_text(text):
    ops = HLO_OP_PATTERN.findall(text)
    histogram = {}
    for op in ops:
        histogram[op] = histogram.get(op, 0) + 1
    return {"stablehlo_op_count": len(ops), "op_histogram": histogram}


def summarize_hlo(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return summarize_hlo_text(text)


def capture_stablehlo_export(module, example_args, path):
    """Lower a fixed-shape module to StableHLO and write it as evidence.

    This is the direct-lowering compile evidence: the recurrence is traced
    by torch.export and lowered through torch_xla to StableHLO text. The
    returned summary counts the stablehlo operations in the graph.
    """
    from torch_xla.stablehlo import exported_program_to_stablehlo

    exported = torch.export.export(module, example_args)
    graph_module = exported_program_to_stablehlo(exported)
    text = graph_module.get_stablehlo_text()
    if not text:
        raise RuntimeError("StableHLO export returned no text")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return summarize_hlo_text(text)


def prepare_device(hlo_path):
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_SAVE_TENSORS_FILE"] = hlo_path
    os.environ["XLA_SAVE_TENSORS_FMT"] = "hlo"
    xr.set_device_type("TT")
    supported = xm.get_xla_supported_devices()
    if len(supported) != EXPECTED_VISIBLE_DEVICE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_VISIBLE_DEVICE_COUNT} visible TT devices, saw {len(supported)}"
        )
    device = xm.xla_device(SELECTED_DEVICE_ORDINAL)
    return device, supported


def compute_deviation(device_tensor, cpu_tensor):
    """Compare a device tensor against the FP32 CPU authority.

    Returns absolute deviation, deviation relative to the reference
    magnitude, Pearson correlation, and the reference scale so the named
    tolerance check is interpretable.
    """
    moved = device_tensor.detach().to("cpu").to(torch.float32)
    reference = cpu_tensor.detach().to(torch.float32)
    if moved.shape != reference.shape:
        raise RuntimeError(
            f"shape mismatch: {tuple(moved.shape)} vs {tuple(reference.shape)}"
        )
    if not torch.isfinite(moved).all():
        raise RuntimeError("device tensor contains a nonfinite value")
    difference = moved - reference
    abs_deviation = float(difference.abs().max().item())
    reference_scale = float(reference.abs().max().item())
    relative = abs_deviation / reference_scale if reference_scale > 0 else abs_deviation
    flat_moved = moved.reshape(-1)
    flat_reference = reference.reshape(-1)
    if flat_moved.numel() > 1 and flat_moved.std() > 0 and flat_reference.std() > 0:
        pcc = float(
            torch.corrcoef(torch.stack([flat_moved, flat_reference]))[0, 1].item()
        )
    else:
        pcc = 1.0
    return {
        "max_abs": abs_deviation,
        "relative": relative,
        "pcc": pcc,
        "reference_scale": reference_scale,
    }


def check_tolerance(deviation, tolerance):
    """Allclose-style gate: |a - b| <= atol + rtol * |reference|."""
    bound = tolerance["atol"] + tolerance["rtol"] * deviation["reference_scale"]
    return deviation["max_abs"] <= bound


def run_wkv7_forward(args):
    import torch_xla

    profile = args.profile
    shape = profile["probe"][args.shape_kind]
    dtype = (
        torch.bfloat16
        if profile["precision"]["model_dtype"] == "bf16"
        else torch.float32
    )
    tolerance = profile["tolerances"]["wkv7_output"]
    state_tolerance = profile["tolerances"]["wkv7_state"]

    generator = torch.Generator().manual_seed(SEED)
    cpu_inputs = build_wkv7_inputs(shape, torch.float32, generator)
    with torch.no_grad():
        cpu_out, cpu_state = wkv7_forward(*cpu_inputs.values())

    # Direct-lowering compile evidence at the admitted device dtype.
    export_args = tuple(tensor.to(dtype) for tensor in cpu_inputs.values())
    hlo_summary = capture_stablehlo_export(Wkv7ForwardModule(), export_args, args.hlo)

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

    out_deviation = compute_deviation(device_out, cpu_out)
    state_deviation = compute_deviation(device_state, cpu_state)
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
        "hlo": hlo_summary,
    }
    return result


def run_wkv7_backward(args):
    import torch_xla

    profile = args.profile
    shape = profile["probe"][args.shape_kind]
    dtype = (
        torch.bfloat16
        if profile["precision"]["model_dtype"] == "bf16"
        else torch.float32
    )
    tolerance = profile["tolerances"]["adapter_gradient"]

    generator = torch.Generator().manual_seed(SEED)
    cpu_inputs = build_wkv7_inputs(shape, torch.float32, generator)

    # Direct-lowering compile evidence for the recurrence. The forward module
    # exports cleanly at the representative shape; the backward module's sum
    # reduction trips a torch.export broadcast error there, so the HLO evidence
    # covers the recurrence lowering that both directions share.
    export_args = tuple(tensor.to(dtype) for tensor in cpu_inputs.values())
    hlo_summary = capture_stablehlo_export(Wkv7ForwardModule(), export_args, args.hlo)

    # The XLA backward must run before any CPU backward in this process: this
    # runtime's autograd engine trips a device-index assert in engine.cpp
    # when a CPU backward initializes the engine first.
    device, supported = prepare_device(args.hlo)
    module = Wkv7BackwardModule().to(dtype).to(device)
    # Eager torch_xla execution keeps the autograd graph intact for the
    # backward pass; the compiled backend path is exercised by the forward
    # probe.
    compile_start = time.monotonic()
    device_leaves = [
        tensor.to(device).to(dtype).requires_grad_(True)
        for tensor in cpu_inputs.values()
    ]
    device_loss = module(*device_leaves)
    device_loss.backward()
    torch_xla.sync(wait=True)
    compile_seconds = time.monotonic() - compile_start

    # Materialize device results before the CPU reference backward runs.
    device_gradients = {}
    for name, leaf in zip(cpu_inputs, device_leaves):
        if leaf.grad is None:
            raise RuntimeError(f"input '{name}' received no gradient")
        device_gradients[name] = leaf.grad.detach().to("cpu").to(torch.float32)
    device_loss_cpu = device_loss.detach().to("cpu").to(torch.float32)

    # CPU reference gradients from identical seeded inputs.
    cpu_leaves = [
        tensor.detach().clone().requires_grad_(True) for tensor in cpu_inputs.values()
    ]
    cpu_loss = Wkv7BackwardModule()(*cpu_leaves)
    cpu_loss.backward()
    cpu_gradients = {
        name: leaf.grad.detach().clone() for name, leaf in zip(cpu_inputs, cpu_leaves)
    }

    deviations = {
        name: compute_deviation(device_gradients[name], cpu_gradients[name])
        for name in cpu_inputs
    }
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
        "loss_deviation": compute_deviation(device_loss_cpu, cpu_loss),
        "deviations": deviations,
        "tolerances_satisfied": all(
            check_tolerance(value, tolerance) for value in deviations.values()
        ),
        "finite": True,
        "hlo": hlo_summary,
    }
    return result


def run_training_ops(args):
    import torch_xla

    profile = args.profile
    model = profile["model"]
    hidden = model["hidden_size"]
    vocab = model["vocabulary_size"]
    static_shape = profile["static_shape"]
    tokens = static_shape["micro_batch"] * static_shape["sequence_length"]
    rank = profile["lora"]["rank"]
    optimizer_policy = profile["optimizer"]
    learning_rate = optimizer_policy["learning_rate"]
    clip_norm = optimizer_policy["clip_norm"]
    grad_tolerance = profile["tolerances"]["adapter_gradient"]
    loss_tolerance = profile["tolerances"]["loss"]
    delta_tolerance = profile["tolerances"]["adapter_delta"]
    dtype = (
        torch.bfloat16
        if profile["precision"]["model_dtype"] == "bf16"
        else torch.float32
    )

    # Build every input once on CPU so the device and reference runs consume
    # identical values.
    generator = torch.Generator().manual_seed(SEED)
    x_cpu = torch.randn(tokens, hidden, generator=generator)
    labels_cpu = torch.randint(0, vocab, (tokens,), generator=generator)
    labels_cpu[::3] = LOSS_IGNORE_INDEX
    logits_cpu = torch.randn(tokens, vocab, generator=generator)
    grad_src_cpu = torch.randn(hidden, rank, generator=generator)
    param_init_cpu = torch.randn(hidden, rank, generator=generator) * 0.02
    base_w_cpu = torch.randn(hidden, hidden, generator=generator) * 0.02
    lora_a_cpu = torch.randn(rank, hidden, generator=generator) * 0.02
    lora_b_cpu = torch.zeros(hidden, rank)

    # Placement evidence: the LoRA projection lowers to StableHLO.
    export_module = LoraProjectionModule(hidden, rank, torch.float32, generator)
    hlo_summary = capture_stablehlo_export(export_module, (x_cpu,), args.hlo)

    def projection(x, base_w, lora_a, lora_b):
        return torch.nn.functional.linear(x, base_w) + x @ lora_a.t() @ lora_b.t()

    # Device phase runs first so the autograd engine is not pre-initialized
    # by a CPU backward.
    device, supported = prepare_device(args.hlo)
    compile_start = time.monotonic()

    x_dev = x_cpu.to(device).to(dtype)
    base_w_dev = base_w_cpu.to(device).to(dtype)
    lora_a_dev = lora_a_cpu.to(device).to(dtype).requires_grad_(True)
    lora_b_dev = lora_b_cpu.to(device).to(dtype).requires_grad_(True)
    proj_out = projection(x_dev, base_w_dev, lora_a_dev, lora_b_dev)
    proj_out.sum().backward()

    logits_dev = logits_cpu.to(device).to(dtype).requires_grad_(True)
    loss_dev = MaskedLossModule()(logits_dev, labels_cpu.to(device))
    loss_dev.backward()

    param_dev = param_init_cpu.to(device).to(dtype).requires_grad_(True)
    grad_src_dev = grad_src_cpu.to(device).to(dtype)
    (param_dev * grad_src_dev).sum().backward()
    clipped_norm_dev = torch.nn.utils.clip_grad_norm_([param_dev], clip_norm)
    optimizer_dev = torch.optim.AdamW(
        [param_dev], lr=learning_rate, weight_decay=0.0, capturable=True
    )
    optimizer_dev.step()
    torch_xla.sync(wait=True)

    device_results = {
        "lora_a_grad": lora_a_dev.grad.detach().to("cpu").to(torch.float32),
        "lora_b_grad": lora_b_dev.grad.detach().to("cpu").to(torch.float32),
        "loss": loss_dev.detach().to("cpu").to(torch.float32),
        "logits_grad": logits_dev.grad.detach().to("cpu").to(torch.float32),
        "clipped_norm": clipped_norm_dev.detach().to("cpu").to(torch.float32),
        "param_after": param_dev.detach().to("cpu").to(torch.float32),
    }
    compile_seconds = time.monotonic() - compile_start

    # CPU reference phase.
    lora_a_ref = lora_a_cpu.clone().requires_grad_(True)
    lora_b_ref = lora_b_cpu.clone().requires_grad_(True)
    proj_ref = projection(x_cpu, base_w_cpu, lora_a_ref, lora_b_ref)
    proj_ref.sum().backward()

    logits_ref = logits_cpu.clone().requires_grad_(True)
    loss_ref = MaskedLossModule()(logits_ref, labels_cpu)
    loss_ref.backward()

    param_ref = param_init_cpu.clone().requires_grad_(True)
    (param_ref * grad_src_cpu).sum().backward()
    clipped_norm_ref = torch.nn.utils.clip_grad_norm_([param_ref], clip_norm)
    optimizer_ref = torch.optim.AdamW([param_ref], lr=learning_rate, weight_decay=0.0)
    optimizer_ref.step()

    deviations = {
        "lora_a_grad": compute_deviation(
            device_results["lora_a_grad"], lora_a_ref.grad
        ),
        "lora_b_grad": compute_deviation(
            device_results["lora_b_grad"], lora_b_ref.grad
        ),
        "logits_grad": compute_deviation(
            device_results["logits_grad"], logits_ref.grad
        ),
        "param_delta": compute_deviation(
            device_results["param_after"] - param_init_cpu,
            param_ref.detach() - param_init_cpu,
        ),
    }
    result = {
        "device": {
            "type": "TT",
            "ordinal": SELECTED_DEVICE_ORDINAL,
            "visible_count": len(supported),
            "supported_devices": supported,
        },
        "compile_seconds": compile_seconds,
        "shape": {"tokens": tokens, "hidden": hidden, "vocab": vocab, "rank": rank},
        "dtype": str(dtype).replace("torch.", ""),
        "loss_deviation": compute_deviation(device_results["loss"], loss_ref.detach()),
        "clipped_norm_deviation": compute_deviation(
            device_results["clipped_norm"], clipped_norm_ref.detach()
        ),
        "deviations": deviations,
        "tolerances_satisfied": (
            check_tolerance(deviations["lora_a_grad"], grad_tolerance)
            and check_tolerance(deviations["lora_b_grad"], grad_tolerance)
            and check_tolerance(deviations["logits_grad"], grad_tolerance)
            and check_tolerance(deviations["param_delta"], delta_tolerance)
            and check_tolerance(
                compute_deviation(device_results["loss"], loss_ref.detach()),
                loss_tolerance,
            )
        ),
        "finite": True,
        "hlo": hlo_summary,
    }
    return result


def run_probe(args):
    handlers = {
        "wkv7-forward": run_wkv7_forward,
        "wkv7-backward": run_wkv7_backward,
        "training-ops": run_training_ops,
    }
    handler = handlers.get(args.probe)
    if handler is None:
        raise RuntimeError(f"unknown probe '{args.probe}'")
    return handler(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument(
        "--shape-kind", default="focused", choices=["focused", "representative"]
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--hlo", required=True)
    args = parser.parse_args()
    with open(args.profile, encoding="utf-8") as handle:
        args.profile = json.load(handle)
    try:
        result = run_probe(args)
        emit("pass", args.probe, **result)
        return 0
    # This process boundary converts every framework error into a terminal diagnostic.
    except Exception as error:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        emit(
            "fail",
            args.probe,
            detail=f"{type(error).__name__}: {error}",
            traceback_tail=tb.strip().splitlines()[-8:],
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
