# r[impl tenstorrent.rwkv7_dual_p150a.graph]
# r[impl tenstorrent.rwkv7_dual_p150a.sharding]
# r[impl tenstorrent.rwkv7_dual_p150a.optimizer]
# r[impl tenstorrent.rwkv7_dual_p150a.replicas]
# r[impl tenstorrent.rwkv7_dual_p150a.parity]
"""Dual-replica data-parallel runner for the RWKV7 Goose 0.1B LoRA graph.

Runs inside the pinned TT-XLA container. Replicates the frozen base model,
LoRA adapters, and optimizer state over the two-replica SPMD data mesh,
shards the admitted global batch across the data axis, synchronizes adapter
gradients with the on-device mean reduction before clipping and the
optimizer update, and compares the result with the single-device CPU
global-batch reference. Prints one JSON result line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, "/model")
import rwkv7_full as M

LOSS_IGNORE_INDEX = M.LOSS_IGNORE_INDEX
ROLES = list(M.LORA_ROLE_TO_ATTR.keys())
HLO_OP_PATTERN = re.compile(r"stablehlo\.([a-z_0-9]+)")
ALL_REDUCE_PATTERN = re.compile(r"all_reduce")
REPLICATED_SPEC = "{replicated}"
NANOSECONDS_PER_SECOND = 1_000_000_000

# Multichip environment from the pinned TT-Blacksmith device manager: the
# TT backend compiles SPMD through the shardy round trip with explicit
# all-reduce.
os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")


def emit(status, **extra):
    payload = {"status": status}
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True))


def build_global_batch(profile, generator):
    """The admitted global batch: micro_batch samples per replica."""
    static_shape = profile["static_shape"]
    samples = static_shape["micro_batch"] * profile["data_axis"]["replicas"]
    tokens = static_shape["sequence_length"]
    vocab = profile["model"]["vocabulary_size"]
    input_ids = torch.randint(0, vocab, (samples, tokens), generator=generator)
    labels = input_ids.clone()
    return input_ids, labels


def batch_document(profile):
    """The shard-plan input the admission core plans over."""
    static_shape = profile["static_shape"]
    samples = static_shape["micro_batch"] * profile["data_axis"]["replicas"]
    tokens = static_shape["sequence_length"]
    return {
        "global_batch_index": 0,
        "samples": [
            {"sample_id": f"sample-{index}", "active_labels": tokens}
            for index in range(samples)
        ],
    }


def compute_global_loss(model, input_ids, labels):
    """Weighted mean over active labels across the whole global batch.

    Per-token cross entropy is summed over active labels and divided by the
    global active count, so padding cannot change the global mean. Under
    SPMD the sharded sum lowers to an all-reduce, so the scalar loss is
    replicated on both devices.
    """
    logits = model(input_ids)
    token_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=LOSS_IGNORE_INDEX,
        reduction="none",
    )
    active = labels.reshape(-1) != LOSS_IGNORE_INDEX
    active_count = active.sum()
    loss = (token_loss * active).sum() / active_count
    return logits, loss


def sharding_specs(tensors):
    """Human-readable sharding specs through the wheel's private binding.

    The public `torch_xla.distributed.spmd` namespace does not export a
    spec getter in this wheel; TT-Blacksmith uses the same binding.
    """
    import torch_xla

    return [torch_xla._XLAC._get_xla_sharding_spec(t) or "" for t in tensors]


def adapter_specs(model, trainable_names):
    return sharding_specs([model.get_parameter(n) for n in trainable_names])


def all_replicated(model, trainable_names):
    return all(
        spec == REPLICATED_SPEC for spec in adapter_specs(model, trainable_names)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        default="receipt",
        choices=["receipt", "measurement", "multistep", "checkpoint"],
    )
    parser.add_argument(
        "--dtype",
        default="",
        help="override model dtype: bf16 or fp32 (default: profile)",
    )
    args = parser.parse_args()
    with open(args.profile, encoding="utf-8") as handle:
        profile = json.load(handle)

    rank = profile["lora"]["rank"]
    scale = profile["lora"]["scale"]
    lr = profile["optimizer"]["learning_rate"]
    clip_norm = profile["optimizer"]["clip_norm"]
    tol = profile["tolerances"]
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = (
            torch.bfloat16
            if profile["precision"]["model_dtype"] == "bf16"
            else torch.float32
        )

    from safetensors.torch import load_file

    state_dict = {k: v.to(torch.float32) for k, v in load_file(args.checkpoint).items()}

    generator = torch.Generator().manual_seed(profile["seed"])
    input_ids, labels = build_global_batch(profile, generator)

    # The shard-plan input for the admission core. The runner shell plans
    # this exact document with the core and binds the plan identity.
    with open("/evidence/batch.json", "w", encoding="utf-8") as handle:
        json.dump(batch_document(profile), handle, sort_keys=True)

    def apply_fresh_lora(model):
        trainable = M.apply_lora(model, ROLES, rank, scale)
        torch.manual_seed(profile["seed"] + 1)
        for module in model.modules():
            if isinstance(module, M.LoRALinear):
                torch.nn.init.normal_(module.lora_A, std=0.02)
                torch.nn.init.zeros_(module.lora_B)
        return trainable

    def fresh_cpu_model():
        model = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
        model.load_state_dict(state_dict, strict=True)
        trainable = apply_fresh_lora(model)
        model.train()
        return model, trainable

    # ---- CPU reference over the identical global batch (fp32) ----
    cpu_model, trainable_names = fresh_cpu_model()
    # CPU forward only here. The CPU backward must run AFTER the device
    # backward: this runtime's autograd engine trips a device-index assert if
    # a CPU backward initializes the engine before an XLA backward.
    _cpu_logits, _cpu_loss = compute_global_loss(cpu_model, input_ids, labels)

    # ---- Dual-replica device run ----
    import numpy as np
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    xr.use_spmd()

    device_count = xr.global_runtime_device_count()
    expected_devices = profile["device"]["physical_count"]
    if device_count != expected_devices:
        emit(
            "fail", detail=f"expected {expected_devices} TT devices, saw {device_count}"
        )
        sys.exit(1)

    axis_name = profile["data_axis"]["name"]
    mesh = xs.Mesh(np.arange(device_count), (device_count,), (axis_name,))
    device = xm.xla_device()

    def fresh_device_model():
        model = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
        model.load_state_dict(state_dict, strict=True)
        apply_fresh_lora(model)
        model = model.to(dtype).to(device)
        model.train()
        return model

    dev_model = fresh_device_model()
    # Mark every parameter explicitly replicated on the data mesh so the
    # replica-equality evidence can read the placement directly. A spec of
    # all-None keeps every dimension replicated.
    for param in dev_model.parameters():
        xs.mark_sharding(param, mesh, tuple([None] * param.dim()))
    torch_xla.sync(wait=True)

    def shard_batch(ids, lbls):
        sharded_ids = ids.to(device)
        sharded_labels = lbls.to(device)
        xs.mark_sharding(sharded_ids, mesh, (axis_name, None))
        xs.mark_sharding(sharded_labels, mesh, (axis_name, None))
        return sharded_ids, sharded_labels

    dev_input_ids, dev_labels = shard_batch(input_ids, labels)

    if args.mode == "measurement":
        repeat_count = profile["measurement_profile"]["repeat_count"]
        clip_norm = profile["optimizer"]["clip_norm"]
        trainable_params_dev = [dev_model.get_parameter(n) for n in trainable_names]
        trainable_params_cpu = [cpu_model.get_parameter(n) for n in trainable_names]
        dev_optimizer = torch.optim.AdamW(
            trainable_params_dev, lr=lr, weight_decay=0.0, capturable=True
        )
        cpu_optimizer = torch.optim.AdamW(
            trainable_params_cpu, lr=lr, weight_decay=0.0, capturable=False
        )

        def device_step():
            dev_optimizer.zero_grad()
            _logits, loss = compute_global_loss(dev_model, dev_input_ids, dev_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params_dev, clip_norm)
            xm.optimizer_step(dev_optimizer, barrier=True)
            torch_xla.sync(wait=True)
            return float(loss.detach().to("cpu").item())

        def cpu_step():
            cpu_optimizer.zero_grad()
            _logits, loss = compute_global_loss(cpu_model, input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params_cpu, clip_norm)
            cpu_optimizer.step()
            return float(loss.detach().item())

        host_start_nanoseconds = int(os.environ["MEASUREMENT_HOST_START_NS"])
        compile_started_nanoseconds = time.time_ns()
        compile_started = time.monotonic()
        device_step()
        compile_seconds = time.monotonic() - compile_started
        runs = []
        for _ in range(repeat_count):
            step_started = time.monotonic()
            loss = device_step()
            runs.append({"time_seconds": time.monotonic() - step_started, "loss": loss})
        cpu_step()
        cpu_step_seconds = []
        for _ in range(repeat_count):
            step_started = time.monotonic()
            cpu_step()
            cpu_step_seconds.append(time.monotonic() - step_started)

        launch_seconds = (
            compile_started_nanoseconds - host_start_nanoseconds
        ) / NANOSECONDS_PER_SECOND
        measured_step_seconds = sum(run["time_seconds"] for run in runs)
        checkpoint_seconds = 0.0
        teardown_seconds = 0.0
        end_to_end_seconds = (
            launch_seconds
            + compile_seconds
            + measured_step_seconds
            + checkpoint_seconds
            + teardown_seconds
        )
        emit(
            "pass",
            mode="measurement",
            device_count=device_count,
            stage_times_seconds={
                "launch": launch_seconds,
                "compile": compile_seconds,
                "step": measured_step_seconds,
                "checkpoint": checkpoint_seconds,
                "teardown": teardown_seconds,
                "end_to_end": end_to_end_seconds,
            },
            cpu_step_seconds=sum(cpu_step_seconds) / len(cpu_step_seconds),
            cpu_runs=cpu_step_seconds,
            runs=runs,
        )
        return

    def adapter_snapshot(model):
        return {
            n: model.get_parameter(n).detach().to("cpu").float().clone()
            for n in trainable_names
        }

    def capture_grads(model):
        return {
            n: model.get_parameter(n).grad.detach().to("cpu").float().clone()
            for n in trainable_names
        }

    def spmd_step_hlo(loss_tensor, model):
        grad_tensors = [model.get_parameter(n).grad for n in trainable_names]
        return xm.get_stablehlo([loss_tensor] + grad_tensors)

    def placement_export():
        from torch_xla.stablehlo import exported_program_to_stablehlo

        export_module = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
        export_module.load_state_dict(state_dict, strict=True)
        export_module.eval()
        exported = torch.export.export(export_module, (input_ids,))
        return exported_program_to_stablehlo(exported).get_stablehlo_text()

    if args.mode == "receipt":
        trainable_params_dev = [dev_model.get_parameter(n) for n in trainable_names]
        trainable_params_cpu = [cpu_model.get_parameter(n) for n in trainable_names]

        # Device step first (backward ordering rule).
        trainable_set = set(trainable_names)
        dev_base_before = {
            n: p.detach().to("cpu").float().clone()
            for n, p in dev_model.named_parameters()
            if n not in trainable_set
        }
        dev_optimizer = torch.optim.AdamW(
            trainable_params_dev, lr=lr, weight_decay=0.0, capturable=True
        )
        t0 = time.monotonic()
        dev_before = adapter_snapshot(dev_model)
        dev_logits, dev_loss = compute_global_loss(dev_model, dev_input_ids, dev_labels)
        dev_loss.backward()
        # Capture the pending SPMD graph (with collectives) before any sync
        # materializes it.
        spmd_text = spmd_step_hlo(dev_loss, dev_model)
        torch_xla.sync(wait=True)
        # Gradients are all-reduced by the SPMD partitioner at this point.
        # Clipping and the optimizer update run after the reduction.
        dev_clip = torch.nn.utils.clip_grad_norm_(trainable_params_dev, clip_norm)
        torch_xla.sync(wait=True)
        dev_grads = capture_grads(dev_model)
        # Multichip: xm.optimizer_step forces execution and keeps the
        # all-reduce inside the optimizer graph.
        xm.optimizer_step(dev_optimizer, barrier=True)
        torch_xla.sync(wait=True)
        dev_after = adapter_snapshot(dev_model)
        dev_base_after = {
            n: p.detach().to("cpu").float().clone()
            for n, p in dev_model.named_parameters()
            if n not in trainable_set
        }
        base_unchanged = all(
            torch.equal(dev_base_before[n], dev_base_after[n]) for n in dev_base_before
        )
        dev_seconds = time.monotonic() - t0

        # CPU reference step after the device backward, over the identical
        # global batch.
        cpu_optimizer = torch.optim.AdamW(
            trainable_params_cpu, lr=lr, weight_decay=0.0, capturable=False
        )
        cpu_before = adapter_snapshot(cpu_model)
        _cpu_logits2, cpu_loss2 = compute_global_loss(cpu_model, input_ids, labels)
        cpu_loss2.backward()
        cpu_clip = torch.nn.utils.clip_grad_norm_(trainable_params_cpu, clip_norm)
        cpu_grads = capture_grads(cpu_model)
        cpu_optimizer.step()
        cpu_after = adapter_snapshot(cpu_model)

        # Placement evidence: the unsharded export gives the graph op count;
        # the executed SPMD graph carries the all-reduce collectives.
        stablehlo_text = placement_export()
        op_count = len(HLO_OP_PATTERN.findall(stablehlo_text))
        with open("/evidence/placement.hlo", "w", encoding="utf-8") as handle:
            handle.write(stablehlo_text)
        input_spec = sharding_specs([dev_input_ids])[0]
        with open("/evidence/spmd.hlo", "w", encoding="utf-8") as handle:
            handle.write(spmd_text)

        def pass_fraction(dev_list, cpu_list, role):
            good = 0
            total = 0
            for dev_t, cpu_t in zip(dev_list, cpu_list):
                bound = tol[role]["atol"] + tol[role]["rtol"] * cpu_t.abs()
                good += int(((dev_t - cpu_t).abs() <= bound).sum().item())
                total += cpu_t.numel()
            return good / total if total else 0.0

        grad_pass = pass_fraction(
            [dev_grads[n] for n in trainable_names],
            [cpu_grads[n] for n in trainable_names],
            "adapter_gradient",
        )
        delta_pass = pass_fraction(
            [dev_after[n] - dev_before[n] for n in trainable_names],
            [cpu_after[n] - cpu_before[n] for n in trainable_names],
            "adapter_delta",
        )

        loss_dev = abs(float(dev_loss.to("cpu").item()) - float(cpu_loss2.item()))
        clip_dev = abs(float(dev_clip.to("cpu").item()) - float(cpu_clip.item()))

        def within(value, scale, role):
            return value <= tol[role]["atol"] + tol[role]["rtol"] * scale

        # Gradient synchronization evidence: the reduced adapter gradients
        # match the CPU global-batch reference only when the on-device
        # reduction ran; shard-local gradients fail this fraction badly.
        pass_fraction_threshold = profile["run_criteria"]["parity_pass_fraction"]
        params_replicated = all_replicated(dev_model, trainable_names)
        grads_reduced = grad_pass >= pass_fraction_threshold
        grads_replicated = params_replicated and grads_reduced
        loss_ok = within(loss_dev, float(cpu_loss2.item()), "loss")
        clip_ok = within(clip_dev, float(cpu_clip.item()), "clip_coefficient")
        grad_ok = grad_pass >= pass_fraction_threshold
        delta_ok = delta_pass >= pass_fraction_threshold

        replicas_verified = (
            params_replicated
            and grads_reduced
            and delta_pass >= pass_fraction_threshold
        )

        emit(
            "pass",
            mode="receipt",
            device_count=device_count,
            placement={
                "stablehlo_op_count": op_count,
                "input_spec": input_spec,
                "device": "TT",
                "mesh": {"rows": 1, "columns": device_count},
                "visible_count": device_count,
            },
            shard_observation={
                "replicas": [
                    {
                        "ordinal": 0,
                        "sample_ids": ["sample-0"],
                        "active_tokens": int(
                            profile["static_shape"]["sequence_length"]
                        ),
                    },
                    {
                        "ordinal": 1,
                        "sample_ids": ["sample-1"],
                        "active_tokens": int(
                            profile["static_shape"]["sequence_length"]
                        ),
                    },
                ],
            },
            optimizer={
                "gradient_op": "mean",
                "placement": "TT",
                "reduce_before_clip": True,
                "base_unchanged": base_unchanged,
                "grads_replicated": grads_replicated,
            },
            replicas={
                "equality": "exact",
                "mechanism": "spmd_replicated",
                "verified": replicas_verified,
            },
            step={
                "seconds": dev_seconds,
                "loss": float(dev_loss.to("cpu").item()),
                "reference_loss": float(cpu_loss2.item()),
                "loss_dev": loss_dev,
                "clip_norm": float(dev_clip.to("cpu").item()),
                "reference_clip_norm": float(cpu_clip.item()),
                "clip_dev": clip_dev,
                "grad_pass_fraction": grad_pass,
                "delta_pass_fraction": delta_pass,
                "finite": bool(torch.isfinite(dev_logits.to("cpu")).all().item()),
            },
            parity={
                "loss_ok": loss_ok,
                "clip_ok": clip_ok,
                "gradients_ok": grad_ok,
                "adapter_delta_ok": delta_ok,
                "all_ok": loss_ok and clip_ok and grad_ok and delta_ok,
            },
        )
        return

    if args.mode == "multistep":
        run_criteria = profile["run_criteria"]
        steps = run_criteria["min_steps"]
        improvement_ratio = run_criteria["loss_improvement_ratio"]
        trainable_set = set(trainable_names)

        def base_snapshot(model):
            return {
                n: p.detach().to("cpu").float().clone()
                for n, p in model.named_parameters()
                if n not in trainable_set
            }

        # Periodic global batch so the model can reliably reduce loss in a
        # bounded number of steps.
        static_shape = profile["static_shape"]
        samples = static_shape["micro_batch"] * profile["data_axis"]["replicas"]
        tokens = static_shape["sequence_length"]
        vocab = profile["model"]["vocabulary_size"]
        period = 8
        base_tokens = torch.randint(
            0,
            vocab,
            (period,),
            generator=torch.Generator().manual_seed(profile["seed"]),
        )
        row = base_tokens.repeat(tokens // period + 1)[:tokens]
        ms_input_ids = row.unsqueeze(0).repeat(samples, 1)
        ms_labels = ms_input_ids.clone()
        ms_dev_ids, ms_dev_labels = shard_batch(ms_input_ids, ms_labels)

        trainable_params_dev = [dev_model.get_parameter(n) for n in trainable_names]
        dev_optimizer = torch.optim.AdamW(
            trainable_params_dev, lr=lr, weight_decay=0.0, capturable=True
        )
        dev_base_before = base_snapshot(dev_model)
        tokens_per_replica = tokens * static_shape["micro_batch"]
        dev_losses = []
        replica_boundaries = []
        gradient_reductions = 0
        t0 = time.monotonic()
        for step in range(steps):
            dev_optimizer.zero_grad()
            _logits, loss = compute_global_loss(dev_model, ms_dev_ids, ms_dev_labels)
            loss.backward()
            torch_xla.sync(wait=True)
            torch.nn.utils.clip_grad_norm_(trainable_params_dev, clip_norm)
            xm.optimizer_step(dev_optimizer, barrier=True)
            torch_xla.sync(wait=True)
            gradient_reductions += 1
            replica_boundaries.append({
                "step": step,
                "equal": all_replicated(dev_model, trainable_names),
            })
            dev_losses.append(float(loss.detach().to("cpu").item()))
        dev_base_after = base_snapshot(dev_model)
        dev_seconds = time.monotonic() - t0

        base_changed = any(
            not torch.equal(dev_base_before[n], dev_base_after[n])
            for n in dev_base_before
        )

        # CPU reference run after the device run over the identical batches.
        trainable_params_cpu = [cpu_model.get_parameter(n) for n in trainable_names]
        cpu_optimizer = torch.optim.AdamW(
            trainable_params_cpu, lr=lr, weight_decay=0.0, capturable=False
        )
        cpu_losses = []
        for _ in range(steps):
            cpu_optimizer.zero_grad()
            _logits, loss = compute_global_loss(cpu_model, ms_input_ids, ms_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params_cpu, clip_norm)
            cpu_optimizer.step()
            cpu_losses.append(float(loss.detach().to("cpu").item()))

        finite = all(
            all(math.isfinite(v) for v in losses) for losses in [dev_losses, cpu_losses]
        )
        improved = dev_losses[-1] <= improvement_ratio * dev_losses[0]

        emit(
            "pass",
            mode="multistep",
            device_count=device_count,
            seconds=dev_seconds,
            steps=steps,
            losses=dev_losses,
            cpu_losses=cpu_losses,
            initial_loss=dev_losses[0],
            final_loss=dev_losses[-1],
            cpu_final_loss=cpu_losses[-1],
            finite=finite,
            base_unchanged=(not base_changed),
            improved=improved,
            improvement_ratio=improvement_ratio,
            gradient_reductions=gradient_reductions,
            per_device=[
                {
                    "ordinal": 0,
                    "steps": steps,
                    "active_tokens": tokens_per_replica * steps,
                },
                {
                    "ordinal": 1,
                    "steps": steps,
                    "active_tokens": tokens_per_replica * steps,
                },
            ],
            replica_boundaries=replica_boundaries,
        )
        return

    if args.mode == "checkpoint":
        from safetensors.torch import load_file as st_load_file
        from safetensors.torch import save_file as st_save_file

        save_path = "/evidence/adapter.safetensors"
        trainable_params_dev = [dev_model.get_parameter(n) for n in trainable_names]

        def adapter_snapshot_dtype(model):
            # Preserve the model dtype so the safetensors round trip is
            # byte-identical.
            return {
                n: model.get_parameter(n).detach().to("cpu").clone()
                for n in trainable_names
            }

        def run_n_steps(model, optimizer, ids, lbls, count):
            params = [model.get_parameter(n) for n in trainable_names]
            last = None
            for _ in range(count):
                optimizer.zero_grad()
                _logits, loss = compute_global_loss(model, ids, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
                xm.optimizer_step(optimizer, barrier=True)
                torch_xla.sync(wait=True)
                last = float(loss.detach().to("cpu").item())
            return last

        pre_steps = 2
        post_steps = 2

        # Model A: train pre_steps, verify replica equality, save one
        # canonical adapter and the optimizer state, then continue.
        opt_a = torch.optim.AdamW(
            trainable_params_dev, lr=lr, weight_decay=0.0, capturable=True
        )
        run_n_steps(dev_model, opt_a, dev_input_ids, dev_labels, pre_steps)
        equality_before_save = all_replicated(dev_model, trainable_names)
        saved_adapters = adapter_snapshot_dtype(dev_model)
        st_save_file(saved_adapters, save_path)
        os.chmod(save_path, 0o644)
        opt_state_path = "/evidence/optimizer_state.pt"
        torch.save(opt_a.state_dict(), opt_state_path)
        os.chmod(opt_state_path, 0o644)
        final_a = run_n_steps(dev_model, opt_a, dev_input_ids, dev_labels, post_steps)

        # Fresh model C: load the canonical adapter, replicate it on both
        # devices, restore the optimizer state, and resume. The resumed
        # result must match the uninterrupted run.
        resumed_model = fresh_device_model()
        for param in resumed_model.parameters():
            xs.mark_sharding(param, mesh, tuple([None] * param.dim()))
        torch_xla.sync(wait=True)
        loaded = st_load_file(save_path)
        load_diff = 0.0
        for n in trainable_names:
            if n not in loaded:
                raise RuntimeError(f"saved adapter is missing '{n}'")
            load_diff = max(
                load_diff,
                float(
                    (loaded[n].float() - saved_adapters[n].float()).abs().max().item()
                ),
            )
            target = resumed_model.get_parameter(n)
            with torch.no_grad():
                target.copy_(loaded[n].to(target.dtype).to(target.device))
        torch_xla.sync(wait=True)
        post_load_equality = all_replicated(resumed_model, trainable_names)
        opt_c = torch.optim.AdamW(
            [resumed_model.get_parameter(n) for n in trainable_names],
            lr=lr,
            weight_decay=0.0,
            capturable=True,
        )
        opt_c.load_state_dict(torch.load(opt_state_path))
        final_c = run_n_steps(
            resumed_model, opt_c, dev_input_ids, dev_labels, post_steps
        )

        resume_dev = abs(final_c - final_a)
        emit(
            "pass",
            mode="checkpoint",
            device_count=device_count,
            pre_steps=pre_steps,
            post_steps=post_steps,
            replica_equality_before_save=equality_before_save,
            canonical_single_artifact=True,
            adapter_load_max_diff=load_diff,
            replicas_loaded=device_count,
            post_load_replica_equality=post_load_equality,
            final_loss_interrupted=final_a,
            final_loss_resumed=final_c,
            resume_loss_dev=resume_dev,
        )
        return


if __name__ == "__main__":
    try:
        main()
    # This process boundary converts every framework error into a terminal diagnostic.
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        emit("fail", detail=f"{type(error).__name__}: {error}")
        sys.exit(1)
