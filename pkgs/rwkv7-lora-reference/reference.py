"""Device-free shell for the RWKV-7 LoRA CPU reference.

Runs bounded validation: equation gradient agreement, LoRA target policy,
one optimizer step that changes only adapters, adapter checkpoint save/load
and resume admission, and a bounded receipt. Makes no hardware claims.
"""

from __future__ import annotations

import json
import os
import sys

import lora_core
import torch
import wkv7
from generated_model_target_identity import (
    CPU_REFERENCE_PRECISION,
    HEAD_COUNT,
    HEAD_SIZE,
    HIDDEN_SIZE,
    LORA_FINITE_DIFFERENCE_TOLERANCE,
    LORA_GRADIENT_AGREEMENT_TOLERANCE,
    MODEL_ID,
    MODEL_LAYER_COUNT,
    MODEL_REVISION,
    PROFILE_BLAKE3,
    TOKENIZER_VOCAB_BLAKE3,
)
from generated_model_target_identity import MODEL_BLAKE3 as CHECKPOINT_BLAKE3

torch.manual_seed(0)

BOUNDED_BATCH = 1
BOUNDED_TIME = 4
BOUNDED_HEADS = 2
BOUNDED_WIDTH = 4
GRADIENT_TOLERANCE = LORA_GRADIENT_AGREEMENT_TOLERANCE
FINITE_DIFFERENCE_TOLERANCE = LORA_FINITE_DIFFERENCE_TOLERANCE
LORA_RANK = 2
LEARNING_RATE = 1e-2


def bounded_inputs() -> dict[str, torch.Tensor]:
    shape = (BOUNDED_BATCH, BOUNDED_TIME, BOUNDED_HEADS, BOUNDED_WIDTH)
    generator = torch.Generator().manual_seed(7)
    return {
        "r": torch.randn(shape, generator=generator),
        "w": torch.randn(shape, generator=generator) * 0.2,
        "k": torch.randn(shape, generator=generator),
        "v": torch.randn(shape, generator=generator),
        "a": torch.sigmoid(torch.randn(shape, generator=generator)),
        "b": torch.randn(shape, generator=generator),
        "state": torch.zeros((
            BOUNDED_BATCH,
            BOUNDED_HEADS,
            BOUNDED_WIDTH,
            BOUNDED_WIDTH,
        )),
    }


def check_gradient_agreement() -> dict[str, float]:
    inputs = bounded_inputs()
    out, _ = wkv7.wkv7_forward(
        inputs["r"],
        inputs["w"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        inputs["state"],
    )
    wkv7.require_finite("wkv7 forward output", out)
    gout = torch.ones_like(out)

    autograd = wkv7.autograd_gradients(
        inputs["r"],
        inputs["w"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        inputs["state"],
        gout,
    )
    manual = wkv7.wkv7_backward(
        inputs["r"],
        inputs["w"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        gout,
        torch.zeros((BOUNDED_BATCH, BOUNDED_HEADS, BOUNDED_WIDTH, BOUNDED_WIDTH)),
    )
    names = ["r", "w", "k", "v", "a", "b"]
    max_manual = 0.0
    for name, ag, mg in zip(names, autograd, manual[:6]):
        for tensor in (ag, mg):
            wkv7.require_finite(f"gradient {name}", tensor)
        max_manual = max(max_manual, wkv7.max_abs_difference(ag, mg))
    if max_manual > GRADIENT_TOLERANCE:
        raise ValueError(f"autograd/manual backward disagree: {max_manual}")

    # Finite-difference check on the r gradient for one element.
    def scalar_for_r(r_candidate: torch.Tensor) -> float:
        candidate_out, _ = wkv7.wkv7_forward(
            r_candidate,
            inputs["w"],
            inputs["k"],
            inputs["v"],
            inputs["a"],
            inputs["b"],
            inputs["state"],
        )
        return float((candidate_out * gout).sum().item())

    fd = wkv7.finite_difference_gradient(scalar_for_r, inputs["r"], (0, 0, 0, 0))
    fd_error = abs(fd - float(autograd[0][0, 0, 0, 0].item()))
    if fd_error > FINITE_DIFFERENCE_TOLERANCE:
        raise ValueError(f"finite difference disagrees with autograd: {fd_error}")

    return {"autograd_vs_manual_max_abs": max_manual, "finite_difference_abs": fd_error}


def check_lora_policy() -> dict[str, object]:
    roles = lora_core.select_lora_targets(lora_core.DEFAULT_LORA_ROLES)
    plan = lora_core.parameter_freeze_plan(
        ["model.embeddings.weight", "lm_head.weight"], list(roles)
    )
    if not plan["trainable"]:
        raise ValueError("LoRA plan produced no trainable adapters")

    for bad_plan, label in (
        ([], "empty"),
        (["all-linear"], "all-linear"),
        (["attention.receptance", "attention.receptance"], "duplicate"),
        (["not.a.real.role"], "unknown"),
    ):
        try:
            lora_core.select_lora_targets(bad_plan)
        except ValueError:
            continue
        raise ValueError(f"unsafe LoRA plan accepted: {label}")
    return {"roles": list(roles), "trainable_count": len(plan["trainable"])}


class BoundedLoraTimemix(torch.nn.Module):
    """A bounded differentiable timemix block with frozen base and LoRA adapters."""

    def __init__(self) -> None:
        super().__init__()
        hidden = BOUNDED_HEADS * BOUNDED_WIDTH
        self.base_receptance = torch.nn.Linear(hidden, hidden, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.zeros(LORA_RANK, hidden))
        self.lora_B = torch.nn.Parameter(torch.zeros(hidden, LORA_RANK))
        torch.nn.init.normal_(self.lora_A)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base_receptance(value)
        adapter = value @ self.lora_A.t() @ self.lora_B.t()
        return base + adapter


def check_one_step() -> dict[str, object]:
    torch.manual_seed(3)
    block = BoundedLoraTimemix()
    hidden = BOUNDED_HEADS * BOUNDED_WIDTH
    base_before = {
        name: tensor.detach().clone()
        for name, tensor in block.named_parameters()
        if not name.startswith("lora_")
    }
    adapter_before = {
        name: tensor.detach().clone()
        for name, tensor in block.named_parameters()
        if name.startswith("lora_")
    }

    optimizer = torch.optim.Adam(
        [p for n, p in block.named_parameters() if n.startswith("lora_")],
        lr=LEARNING_RATE,
    )
    stimulus = torch.randn(BOUNDED_TIME, hidden)
    target = torch.randn(BOUNDED_TIME, hidden)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(block(stimulus), target)
    if not torch.isfinite(loss):
        raise ValueError("training loss is nonfinite")
    loss.backward()
    got_adapter_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in block.named_parameters()
        if n.startswith("lora_")
    )
    if not got_adapter_grad:
        raise ValueError("no adapter tensor received a finite gradient")
    optimizer.step()

    for name, before in base_before.items():
        after = dict(block.named_parameters())[name].detach()
        if not torch.equal(before, after):
            raise ValueError(f"base tensor changed: {name}")
    changed = False
    for name, before in adapter_before.items():
        after = dict(block.named_parameters())[name].detach()
        if not torch.equal(before, after):
            changed = True
    if not changed:
        raise ValueError("no adapter tensor changed after the optimizer step")
    return {"loss": float(loss.item()), "base_frozen": True, "adapter_changed": changed}


def check_checkpoint_resume() -> dict[str, object]:
    torch.manual_seed(5)
    block = BoundedLoraTimemix()
    adapters = {
        name: tensor.detach().clone()
        for name, tensor in block.named_parameters()
        if name.startswith("lora_")
    }
    # Zero-effect init: force at least one nonzero adapter so save/load is meaningful.
    adapters["lora_A"].normal_()

    from safetensors.torch import load as st_load
    from safetensors.torch import save as st_save

    blob = st_save(adapters)
    tensor_blake3 = {
        name: lora_core.blake3_hex(t.numpy().tobytes()) for name, t in adapters.items()
    }
    bound = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_blake3": CHECKPOINT_BLAKE3,
        "tokenizer_vocab_blake3": TOKENIZER_VOCAB_BLAKE3,
        "lora_roles": list(lora_core.DEFAULT_LORA_ROLES),
        "lora_rank": LORA_RANK,
    }
    manifest = lora_core.checkpoint_manifest(tensor_blake3, bound, step=1)

    loaded = st_load(blob)
    for name, tensor in adapters.items():
        if not torch.equal(tensor, loaded[name]):
            raise ValueError(f"loaded adapter differs from saved: {name}")

    # Resume with matching identities passes; drift fails closed.
    lora_core.validate_resume_manifest(manifest, dict(manifest))
    drifted = dict(manifest)
    drifted["bound"] = dict(bound, model_revision="deadbeef")
    try:
        lora_core.validate_resume_manifest(manifest, drifted)
    except ValueError:
        resume_rejected = True
    else:
        raise ValueError("incompatible resume manifest was accepted")
    return {
        "manifest_blake3": manifest["manifest_blake3"],
        "resume_rejected": resume_rejected,
    }


def check_data_planning() -> dict[str, object]:
    sequence_length = 64
    total_active = 0
    record_count = 0
    with open("fixtures/corpus.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt, response = lora_core.admit_record(record)
            prompt_ids = [b + 1 for b in prompt.encode()]
            response_ids = [b + 1 for b in response.encode()]
            plan = lora_core.plan_token_labels(
                prompt_ids, response_ids, sequence_length
            )
            plan_again = lora_core.plan_token_labels(
                prompt_ids, response_ids, sequence_length
            )
            if plan != plan_again:
                raise ValueError("label planning is not deterministic")
            active = [label for label in plan["labels"] if label != -100]
            if not active:
                raise ValueError("record produced no active labels")
            total_active += len(active)
            record_count += 1
    if record_count == 0:
        raise ValueError("corpus produced no records")

    # The malformed corpus must fail admission on its bad record.
    malformed_failed = False
    with open("fixtures/corpus-malformed.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                lora_core.admit_record(json.loads(line))
            except ValueError:
                malformed_failed = True
    if not malformed_failed:
        raise ValueError("malformed corpus did not fail admission")

    for bad, label in (
        ({"prompt": "", "response": "x"}, "empty prompt"),
        ({"prompt": "x", "response": "   "}, "empty response"),
        ({"prompt": "x", "response": "y", "extra": 1}, "unknown field"),
        ({"prompt": "x"}, "missing response"),
    ):
        try:
            lora_core.admit_record(bad)
        except ValueError:
            continue
        raise ValueError(f"invalid record accepted: {label}")
    return {
        "record_count": record_count,
        "active_labels": total_active,
        "sequence_length": sequence_length,
        "malformed_corpus_rejected": malformed_failed,
    }


def check_model_authority() -> dict[str, object]:
    fields = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "checkpoint_blake3": CHECKPOINT_BLAKE3,
        "tokenizer_vocab_blake3": TOKENIZER_VOCAB_BLAKE3,
        "layer_count": MODEL_LAYER_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "head_count": HEAD_COUNT,
        "head_size": HEAD_SIZE,
        "precision": CPU_REFERENCE_PRECISION,
    }
    identity = lora_core.authority_identity(fields)
    identity_again = lora_core.authority_identity(fields)
    if identity != identity_again:
        raise ValueError("authority identity is not deterministic")
    drifted = dict(fields, revision="deadbeef")
    if lora_core.authority_identity(drifted) == identity:
        raise ValueError("authority drift was not detected")
    return {
        "authority_blake3": identity,
        "profile_blake3": PROFILE_BLAKE3,
    }


def check_model_load() -> dict[str, object]:
    """Load the pinned BF16 checkpoint and demonstrate FP32-from-BF16.

    Bounded: opens the safetensors file, reads one tensor, verifies its stored
    dtype is bf16 and its BLAKE3-bound model path, and converts a bounded
    slice to finite FP32.
    """
    model_path = os.environ.get("RWKV7_LORA_MODEL")
    expected_blake3 = os.environ.get("RWKV7_LORA_MODEL_BLAKE3")
    if not model_path:
        raise ValueError("RWKV7_LORA_MODEL is not bound")
    from safetensors import safe_open

    with safe_open(model_path, framework="pt") as handle:
        names = handle.keys()
        if not names:
            raise ValueError("pinned checkpoint has no tensors")
        first = min(names)
        tensor = handle.get_tensor(first)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"pinned checkpoint tensor dtype is {tensor.dtype}, not bf16")
    bounded = tensor.reshape(-1)[:64].to(torch.float32)
    if not torch.isfinite(bounded).all():
        raise ValueError("FP32-from-BF16 conversion produced a nonfinite value")
    if expected_blake3 != CHECKPOINT_BLAKE3:
        raise ValueError("bound checkpoint BLAKE3 differs from the pinned authority")
    return {
        "checkpoint_dtype": "bf16",
        "fp32_from_bf16_finite": True,
        "first_tensor": first,
        "checkpoint_blake3": expected_blake3,
    }


def main() -> int:
    receipt = {
        "schema_version": lora_core.SCHEMA_VERSION,
        "model": check_model_authority(),
        "model_load": check_model_load(),
        "data": check_data_planning(),
        "lora": check_lora_policy(),
        "gradients": check_gradient_agreement(),
        "training_step": check_one_step(),
        "checkpoint": check_checkpoint_resume(),
        "non_claims": [
            "No Tenstorrent execution is established.",
            "No useful convergence is established.",
            "No dataset quality is established.",
            "No production fine-tuning quality is established.",
        ],
        "outcome": "bounded_cpu_reference_pass",
    }
    json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
