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
import os
import sys
import time

import torch

sys.path.insert(0, "/model")
import rwkv7_full as M

LOSS_IGNORE_INDEX = M.LOSS_IGNORE_INDEX
ROLES = list(M.LORA_ROLE_TO_ATTR.keys())
SELECTED_ORDINAL = 0


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
                        choices=["forward", "step", "debug", "isolate"])
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
    cpu_logits, cpu_loss = compute_loss(cpu_model, input_ids, labels)
    cpu_loss.backward()
    cpu_grads = {n: cpu_model.get_parameter(n).grad.clone()
                 for n in trainable_names}

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
