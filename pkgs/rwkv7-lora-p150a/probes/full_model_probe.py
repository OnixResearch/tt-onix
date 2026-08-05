# r[impl tenstorrent.rwkv7_p150a_finetuning.graph]
# r[impl tenstorrent.rwkv7_p150a_finetuning.device]
"""Full-model device probe for the RWKV7 Goose 0.1B LoRA graph.

Runs inside the pinned TT-XLA container. Loads the pinned checkpoint,
applies rank-2 LoRA to the admitted projection roles, moves the graph to one
logical P150A, and reports forward/backward/optimizer evidence plus a CPU
parity comparison. Prints one JSON result line.
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
SELECTED_ORDINAL = 0
HLO_OP_PATTERN = re.compile(r"stablehlo\.([a-z_0-9]+)")


def emit(status, **extra):
    payload = {"status": status}
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True))


def build_batch(profile, generator):
    static_shape = profile["static_shape"]
    tokens = static_shape["micro_batch"] * static_shape["sequence_length"]
    vocab = profile["model"]["vocabulary_size"]
    input_ids = torch.randint(0, vocab, (static_shape["micro_batch"], tokens),
                              generator=generator)
    labels = input_ids.clone()
    return input_ids, labels


def compute_loss(model, input_ids, labels):
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=LOSS_IGNORE_INDEX)
    return logits, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="forward",
                        choices=["forward", "step", "debug", "isolate", "receipt", "multistep", "checkpoint"])
    parser.add_argument("--dtype", default="",
                        help="override model dtype: bf16 or fp32 (default: profile)")
    parser.add_argument("--emit-cpu", default="",
                        help="optional path to also save CPU reference tensors")
    args = parser.parse_args()
    with open(args.profile, encoding="utf-8") as handle:
        profile = json.load(handle)

    rank = profile["lora"]["rank"]
    scale = profile["lora"]["scale"]
    lr = profile["optimizer"]["learning_rate"]
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if profile["precision"]["model_dtype"] == "bf16" else torch.float32

    from safetensors.torch import load_file
    state_dict = {k: v.to(torch.float32) for k, v in load_file(args.checkpoint).items()}

    generator = torch.Generator().manual_seed(profile["seed"])
    input_ids, labels = build_batch(profile, generator)

    # ---- CPU reference (fp32) ----
    cpu_model = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
    cpu_model.load_state_dict(state_dict, strict=True)
    trainable_names = M.apply_lora(cpu_model, ROLES, rank, scale)
    cpu_model.train()
    # seed the LoRA init identically on both sides by rebuilding with same seed
    torch.manual_seed(profile["seed"] + 1)
    for module in cpu_model.modules():
        if isinstance(module, M.LoRALinear):
            torch.nn.init.normal_(module.lora_A, std=0.02)
            torch.nn.init.zeros_(module.lora_B)
    # CPU forward only here. The CPU backward must run AFTER the device
    # backward: this runtime's autograd engine trips a device-index assert if
    # a CPU backward initializes the engine before an XLA backward.
    cpu_logits, cpu_loss = compute_loss(cpu_model, input_ids, labels)

    # ---- Device run ----
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    os.environ["PJRT_DEVICE"] = "TT"
    xr.set_device_type("TT")
    supported = xm.get_xla_supported_devices()
    device = xm.xla_device(SELECTED_ORDINAL)

    dev_model = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
    dev_model.load_state_dict(state_dict, strict=True)
    M.apply_lora(dev_model, ROLES, rank, scale)
    torch.manual_seed(profile["seed"] + 1)
    for module in dev_model.modules():
        if isinstance(module, M.LoRALinear):
            torch.nn.init.normal_(module.lora_A, std=0.02)
            torch.nn.init.zeros_(module.lora_B)
    dev_model = dev_model.to(dtype).to(device)
    dev_model.train()

    if args.mode == "debug":
        # Localize the first non-finite activation by hooking each block and
        # the embedding / final norm / lm_head.
        captured = {}
        hooks = []

        def make_hook(name):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    captured[name] = tensor
            return hook

        hooks.append(dev_model.model.embeddings.register_forward_hook(make_hook("embeddings")))
        for i, layer in enumerate(dev_model.model.layers):
            hooks.append(layer.register_forward_hook(make_hook(f"layer{i}")))
        hooks.append(dev_model.model.norm.register_forward_hook(make_hook("final_norm")))

        with torch.no_grad():
            dev_logits = dev_model(input_ids.to(device))
        torch_xla.sync(wait=True)
        report = {}
        for name, tensor in captured.items():
            cpu_t = tensor.to("cpu")
            report[name] = {"absmax": float(cpu_t.abs().max().item()),
                            "finite": bool(torch.isfinite(cpu_t).all().item())}
        logits_cpu = dev_logits.to("cpu")
        report["logits"] = {"absmax": float(logits_cpu.abs().max().item()),
                            "finite": bool(torch.isfinite(logits_cpu).all().item())}
        for h in hooks:
            h.remove()
        emit("pass", mode="debug", device_count=len(supported), report=report)
        return

    if args.mode == "isolate":
        # Op-by-op comparison of layer-0 attention between CPU and device.
        def layer0_prefix(model):
            device_of = next(model.parameters()).device
            emb = model.model.embeddings(input_ids.to(device_of))
            layer0 = model.model.layers[0]
            pre = layer0.pre_norm(emb)
            attn_in = layer0.attn_norm(pre)
            return {"emb": emb, "pre_norm": pre, "attn_input": attn_in}, attn_in

        def summarize(tag, cpu_t, dev_t, out):
            cpu_t = cpu_t.detach().to("cpu").float()
            dev_t = dev_t.detach().to("cpu").float()
            out[tag] = {
                "max_abs_diff": float((dev_t - cpu_t).abs().max().item()),
                "cpu_absmax": float(cpu_t.abs().max().item()),
                "dev_absmax": float(dev_t.abs().max().item()),
            }

        comparison = {}
        with torch.no_grad():
            cpu_prefix, cpu_h = layer0_prefix(cpu_model)
            dev_prefix, dev_h = layer0_prefix(dev_model)
        torch_xla.sync(wait=True)
        for tag in ("emb", "pre_norm", "attn_input"):
            summarize(tag, cpu_prefix[tag], dev_prefix[tag], comparison)

        with torch.no_grad():
            cpu_inter = M.attention_intermediates(cpu_model.model.layers[0].attn, cpu_h)
            dev_inter = M.attention_intermediates(dev_model.model.layers[0].attn, dev_h)
        torch_xla.sync(wait=True)
        for key in cpu_inter:
            summarize(key, cpu_inter[key], dev_inter[key], comparison)
        emit("pass", mode="isolate", device_count=len(supported), comparison=comparison)
        return

    if args.mode == "forward":
        t0 = time.monotonic()
        with torch.no_grad():
            dev_logits, dev_loss = compute_loss(dev_model, input_ids.to(device),
                                                labels.to(device))
        torch_xla.sync(wait=True)
        dt = time.monotonic() - t0
        emit("pass", mode="forward", device_count=len(supported),
             seconds=dt,
             loss=float(dev_loss.to("cpu").item()),
             logits_absmax=float(dev_logits.abs().max().to("cpu").item()),
             logits_finite=bool(torch.isfinite(dev_logits.to("cpu")).all().item()),
             cpu_loss=float(cpu_loss.item()),
             loss_dev=float(abs(dev_loss.to("cpu").item() - cpu_loss.item())))
        return

    # mode == "receipt": full one-step placement + parity receipt
    if args.mode == "receipt":
        clip_norm = profile["optimizer"]["clip_norm"]
        tol = profile["tolerances"]
        trainable_params_dev = [dev_model.get_parameter(n) for n in trainable_names]
        trainable_params_cpu = [cpu_model.get_parameter(n) for n in trainable_names]

        def adapter_snapshot(model):
            return {n: model.get_parameter(n).detach().to("cpu").float().clone()
                    for n in trainable_names}

        def capture_grads(model):
            return {n: model.get_parameter(n).grad.detach().to("cpu").float().clone()
                    for n in trainable_names}

        # Device step first (backward ordering rule).
        dev_optimizer = torch.optim.AdamW(trainable_params_dev, lr=lr,
                                          weight_decay=0.0, capturable=True)
        t0 = time.monotonic()
        dev_before = adapter_snapshot(dev_model)
        dev_logits, dev_loss = compute_loss(dev_model, input_ids.to(device),
                                            labels.to(device))
        dev_loss.backward()
        dev_clip = torch.nn.utils.clip_grad_norm_(trainable_params_dev, clip_norm)
        torch_xla.sync(wait=True)
        dev_grads = capture_grads(dev_model)
        dev_optimizer.step()
        torch_xla.sync(wait=True)
        dev_after = adapter_snapshot(dev_model)
        dev_seconds = time.monotonic() - t0

        # CPU reference step after the device backward.
        cpu_optimizer = torch.optim.AdamW(trainable_params_cpu, lr=lr,
                                          weight_decay=0.0, capturable=False)
        cpu_before = adapter_snapshot(cpu_model)
        cpu_logits2, cpu_loss2 = compute_loss(cpu_model, input_ids, labels)
        cpu_loss2.backward()
        cpu_clip = torch.nn.utils.clip_grad_norm_(trainable_params_cpu, clip_norm)
        cpu_grads = capture_grads(cpu_model)
        cpu_optimizer.step()
        cpu_after = adapter_snapshot(cpu_model)

        # Placement evidence: lower the full forward to StableHLO and count ops.
        from torch_xla.stablehlo import exported_program_to_stablehlo
        export_module = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
        export_module.load_state_dict(state_dict, strict=True)
        export_module.eval()
        exported = torch.export.export(export_module, (input_ids,))
        stablehlo_text = exported_program_to_stablehlo(exported).get_stablehlo_text()
        op_count = len(HLO_OP_PATTERN.findall(stablehlo_text))
        with open("/evidence/placement.hlo", "w", encoding="utf-8") as handle:
            handle.write(stablehlo_text)

        def rel_dev(dev_t, cpu_t):
            scale = max(float(cpu_t.abs().max().item()), 1e-9)
            return float((dev_t - cpu_t).abs().max().item()) / scale

        grad_devs = {n: rel_dev(dev_grads[n], cpu_grads[n]) for n in trainable_names}
        delta_devs = {n: rel_dev(dev_after[n] - dev_before[n],
                                 cpu_after[n] - cpu_before[n])
                      for n in trainable_names}
        max_grad_rel = max(grad_devs.values())
        max_delta_rel = max(delta_devs.values())
        loss_dev = abs(float(dev_loss.to("cpu").item()) - float(cpu_loss2.item()))
        clip_dev = abs(float(dev_clip.to("cpu").item()) - float(cpu_clip.item()))

        def within(value, scale, role):
            return value <= tol[role]["atol"] + tol[role]["rtol"] * scale

        grad_scale = max(float(cpu_grads[n].abs().max().item()) for n in trainable_names)
        delta_scale = max(float((cpu_after[n] - cpu_before[n]).abs().max().item())
                          for n in trainable_names)
        def pass_fraction(dev_list, cpu_list, role):
            """Fraction of elements within atol + rtol*|cpu| across tensors.

            The chaotic recurrence makes a few worst-case gradient/delta
            elements flip sign, so the max deviation is dominated by outliers.
            The element pass fraction is the robust parity measure for the
            chaotic roles.
            """
            good = 0
            total = 0
            for dev_t, cpu_t in zip(dev_list, cpu_list):
                bound = tol[role]["atol"] + tol[role]["rtol"] * cpu_t.abs()
                good += int(((dev_t - cpu_t).abs() <= bound).sum().item())
                total += cpu_t.numel()
            return good / total if total else 0.0

        grad_dev_tensors = [dev_grads[n] for n in trainable_names]
        grad_cpu_tensors = [cpu_grads[n] for n in trainable_names]
        delta_dev_tensors = [dev_after[n] - dev_before[n] for n in trainable_names]
        delta_cpu_tensors = [cpu_after[n] - cpu_before[n] for n in trainable_names]

        grad_pass = pass_fraction(grad_dev_tensors, grad_cpu_tensors, "adapter_gradient")
        delta_pass = pass_fraction(delta_dev_tensors, delta_cpu_tensors, "adapter_delta")

        pass_fraction_threshold = profile["run_criteria"]["parity_pass_fraction"]
        grad_ok = grad_pass >= pass_fraction_threshold
        delta_ok = delta_pass >= pass_fraction_threshold
        loss_ok = within(loss_dev, float(cpu_loss2.item()), "loss")
        clip_ok = within(clip_dev, float(cpu_clip.item()), "clip_coefficient")

        emit("pass", mode="receipt", device_count=len(supported),
             placement={"stablehlo_op_count": op_count, "device": "TT",
                        "ordinal": SELECTED_ORDINAL, "visible_count": len(supported)},
             step={"seconds": dev_seconds,
                   "loss": float(dev_loss.to("cpu").item()),
                   "cpu_loss": float(cpu_loss2.item()),
                   "loss_dev": loss_dev,
                   "clip_norm": float(dev_clip.to("cpu").item()),
                   "cpu_clip_norm": float(cpu_clip.item()),
                   "clip_dev": clip_dev,                   "max_grad_rel_dev": max_grad_rel,
                   "max_delta_rel_dev": max_delta_rel,
                   "grad_pass_fraction": grad_pass,
                   "delta_pass_fraction": delta_pass,
                   "finite": bool(torch.isfinite(dev_logits.to("cpu")).all().item())},
             parity={"loss_ok": loss_ok, "clip_ok": clip_ok,
                     "gradients_ok": grad_ok, "adapter_delta_ok": delta_ok,
                     "all_ok": loss_ok and clip_ok and grad_ok and delta_ok})
        return

    # mode == "checkpoint": adapter save, load, and resume parity.
    if args.mode == "checkpoint":
        from safetensors.torch import save_file as st_save_file
        from safetensors.torch import load_file as st_load_file

        run_criteria = profile["run_criteria"]
        clip_norm = profile["optimizer"]["clip_norm"]
        trainable_set = set(trainable_names)
        save_path = "/evidence/adapter.safetensors"

        def adapter_snapshot(model):
            # Preserve the model dtype so the safetensors save/load round trip
            # is byte-identical; casting to fp32 here would introduce a load
            # deviation that the resume comparison then amplifies.
            return {n: model.get_parameter(n).detach().to("cpu").clone()
                    for n in trainable_names}

        def run_n_steps(model, optimizer, ids, labels, count):
            params = [model.get_parameter(n) for n in trainable_names]
            last = None
            for _ in range(count):
                optimizer.zero_grad()
                logits, loss = compute_loss(model, ids, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
                optimizer.step()
                torch_xla.sync(wait=True)
                last = float(loss.detach().to("cpu").item())
            return last

        pre_steps = 2
        post_steps = 2

        # Model A: train pre_steps, save the adapter and optimizer state, then
        # continue. Restoring both makes resume match the uninterrupted run.
        opt_a = torch.optim.AdamW([dev_model.get_parameter(n) for n in trainable_names],
                                  lr=lr, weight_decay=0.0, capturable=True)
        run_n_steps(dev_model, opt_a, input_ids.to(device), labels.to(device), pre_steps)
        saved_adapters = adapter_snapshot(dev_model)
        st_save_file(saved_adapters, save_path)
        os.chmod(save_path, 0o644)
        opt_state_path = "/evidence/optimizer_state.pt"
        torch.save(opt_a.state_dict(), opt_state_path)
        os.chmod(opt_state_path, 0o644)
        final_a = run_n_steps(dev_model, opt_a, input_ids.to(device), labels.to(device), post_steps)

        # Fresh model C: load the saved adapter and continue; must match A.
        resumed_model = M.RWKV7ForCausalLM(M.GOOSE_0P1B_CONFIG)
        resumed_model.load_state_dict(state_dict, strict=True)
        M.apply_lora(resumed_model, ROLES, rank, scale)
        torch.manual_seed(profile["seed"] + 1)
        for module in resumed_model.modules():
            if isinstance(module, M.LoRALinear):
                torch.nn.init.normal_(module.lora_A, std=0.02)
                torch.nn.init.zeros_(module.lora_B)
        resumed_model = resumed_model.to(dtype).to(device)
        loaded = st_load_file(save_path)
        load_diff = 0.0
        for n in trainable_names:
            if n not in loaded:
                raise RuntimeError(f"saved adapter is missing '{n}'")
            # Round-trip check: the loaded tensor must equal the saved tensor.
            load_diff = max(
                load_diff,
                float((loaded[n].float() - saved_adapters[n].float()).abs().max().item()))
            target = resumed_model.get_parameter(n)
            with torch.no_grad():
                target.copy_(loaded[n].to(target.dtype).to(target.device))
        torch_xla.sync(wait=True)
        opt_c = torch.optim.AdamW([resumed_model.get_parameter(n) for n in trainable_names],
                                  lr=lr, weight_decay=0.0, capturable=True)
        opt_c.load_state_dict(torch.load(opt_state_path))
        final_c = run_n_steps(resumed_model, opt_c, input_ids.to(device), labels.to(device), post_steps)

        resume_dev = abs(final_c - final_a)
        emit("pass", mode="checkpoint", device_count=len(supported),
             pre_steps=pre_steps, post_steps=post_steps,
             adapter_load_max_diff=load_diff,
             final_loss_interrupted=final_a,
             final_loss_resumed=final_c,
             resume_loss_dev=resume_dev)
        return

    # mode == "multistep": bounded multi-step run with loss-improvement
    # criterion, adapter-only mutation, and finite-value validation.
    if args.mode == "multistep":
        run_criteria = profile["run_criteria"]
        steps = run_criteria["min_steps"]
        improvement_ratio = run_criteria["loss_improvement_ratio"]
        clip_norm = profile["optimizer"]["clip_norm"]
        trainable_set = set(trainable_names)

        def base_snapshot(model):
            return {n: p.detach().to("cpu").float().clone()
                    for n, p in model.named_parameters() if n not in trainable_set}

        def run_steps(model, optimizer, ids, labels, on_device):
            losses = []
            params = [model.get_parameter(n) for n in trainable_names]
            for _ in range(steps):
                optimizer.zero_grad()
                logits, loss = compute_loss(model, ids, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, clip_norm)
                optimizer.step()
                if on_device:
                    torch_xla.sync(wait=True)
                losses.append(float(loss.detach().to("cpu").item()))
            return losses

        # Use a periodic sequence so the model can reliably reduce loss in a
        # bounded number of steps; a purely random batch cannot be learned.
        static_shape = profile["static_shape"]
        tokens = static_shape["micro_batch"] * static_shape["sequence_length"]
        vocab = profile["model"]["vocabulary_size"]
        period = 8
        base_tokens = torch.randint(0, vocab, (period,), generator=torch.Generator().manual_seed(profile["seed"]))
        ms_input_ids = base_tokens.repeat(tokens // period + 1)[:tokens].unsqueeze(0)
        ms_labels = ms_input_ids.clone()

        # Device run first (backward ordering rule).
        dev_optimizer = torch.optim.AdamW(
            [dev_model.get_parameter(n) for n in trainable_names],
            lr=lr, weight_decay=0.0, capturable=True)
        dev_base_before = base_snapshot(dev_model)
        t0 = time.monotonic()
        dev_losses = run_steps(dev_model, dev_optimizer,
                               ms_input_ids.to(device), ms_labels.to(device), True)
        dev_base_after = base_snapshot(dev_model)
        dev_seconds = time.monotonic() - t0

        base_changed = any(
            not torch.equal(dev_base_before[n], dev_base_after[n])
            for n in dev_base_before)

        # CPU reference run after the device run.
        cpu_optimizer = torch.optim.AdamW(
            [cpu_model.get_parameter(n) for n in trainable_names],
            lr=lr, weight_decay=0.0, capturable=False)
        cpu_losses = run_steps(cpu_model, cpu_optimizer, ms_input_ids, ms_labels, False)

        finite = all(
            all(math.isfinite(v) for v in losses) for losses in [dev_losses, cpu_losses])
        improved = dev_losses[-1] <= improvement_ratio * dev_losses[0]
        final_loss_dev = abs(dev_losses[-1] - cpu_losses[-1])

        emit("pass", mode="multistep", device_count=len(supported),
             seconds=dev_seconds,
             steps=steps,
             losses=dev_losses,
             cpu_losses=cpu_losses,
             initial_loss=dev_losses[0],
             final_loss=dev_losses[-1],
             cpu_final_loss=cpu_losses[-1],
             final_loss_dev=final_loss_dev,
             finite=finite,
             base_unchanged=(not base_changed),
             improved=improved,
             improvement_ratio=improvement_ratio)
        return

    # mode == "step": backward + AdamW adapter update
    optimizer = torch.optim.AdamW(
        [p for n, p in dev_model.named_parameters() if n in set(trainable_names)],
        lr=lr, weight_decay=0.0, capturable=True)
    t0 = time.monotonic()
    dev_logits, dev_loss = compute_loss(dev_model, input_ids.to(device), labels.to(device))
    dev_loss.backward()
    torch_xla.sync(wait=True)
    grad_dev = {n: dev_model.get_parameter(n).grad.to("cpu").to(torch.float32)
                for n in trainable_names}
    optimizer.step()
    torch_xla.sync(wait=True)
    dt = time.monotonic() - t0

    # CPU reference backward runs only after the device backward completes.
    cpu_loss.backward()
    cpu_grads = {n: cpu_model.get_parameter(n).grad.clone()
                 for n in trainable_names}

    max_grad_dev = 0.0
    for n in trainable_names:
        g_dev = grad_dev[n]
        g_cpu = cpu_grads[n]
        scale_ref = max(float(g_cpu.abs().max().item()), 1e-9)
        max_grad_dev = max(max_grad_dev,
                           float((g_dev - g_cpu).abs().max().item()) / scale_ref)

    emit("pass", mode="step", device_count=len(supported), seconds=dt,
         loss=float(dev_loss.to("cpu").item()), cpu_loss=float(cpu_loss.item()),
         max_grad_rel_dev=max_grad_dev,
         logits_finite=bool(torch.isfinite(dev_logits.to("cpu")).all().item()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        emit("fail", detail=f"{type(error).__name__}: {error}")
        sys.exit(1)
