# RWKV-7 LoRA bring-up on one P150A

Bounded single-device RWKV-7 LoRA fine-tuning on one logical P150A card,
inside the validated dual-card TT-XLA training platform.

This package proves the smallest complete accelerator claim: one admitted SFT
batch, full loss, adapter gradients, and one optimizer step on one logical
P150A device, plus a bounded multi-step run and adapter save/load/resume.

It does not prove two-card scaling, long-context support, useful
generalization, or speedup over the CPU reference.

## What this establishes

The admitted evidence for the named profile `rwkv7-lora-p150a-single`:

- **Compile probes** (WKV7 forward, WKV7 backward, LoRA projection, masked
  loss, gradient norm, clipping, optimizer update) compile and run on device
  with CPU parity. Checked-in receipts re-validate device-free.
- **Placement**: the full trainable graph lowers to 25594 StableHLO operations
  on the selected TT device with no host fallback.
- **One-step parity**: one forward/backward/clip/AdamW step matches the CPU
  authority. Loss deviation 0.0013, gradient element pass fraction 1.0,
  adapter-delta pass fraction 0.991, clip within the named tolerance.
- **Bounded multi-step run**: 8 steps keep every value finite, mutate only the
  LoRA adapters, and drive the loss from 13.19 to 0.0 against a periodic
  reference batch while the CPU reference tracks it.
- **Checkpoint**: the adapter artifact saves to Safetensors, loads
  byte-identical, and resumes to a result that matches the uninterrupted run
  exactly (the AdamW optimizer state is restored alongside the adapter).

Every receipt binds the exact runtime, profile, graph, inputs, and device
with BLAKE3 and is re-validated by the device-free check.

## Model authority

The full model is the pinned `RWKV/RWKV7-Goose-World2.8-0.1B-HF` checkpoint
(revision `d81965cb4e1a9f96696b4f70b84212b8f2e43216`, checkpoint BLAKE3
`905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5`). The
reimplementation in `model/rwkv7_full.py` is pure torch so it runs unmodified
on CPU (FP32 reference) and on the XLA device (BF16). All 399 checkpoint
tensors load with no missing or unexpected keys.

The WKV7 recurrence uses the paper/naive convention with fla inputs mapped in.
That mapping was verified numerically against fla's `chunk_rwkv7` kernel
(max abs diff ~4e-6 in fp32) before the model was written.

## The runtime LayerNorm defect (important)

This runtime's fused `nn.LayerNorm` (and `nn.GroupNorm`) lowering is
numerically wrong on device: the error grows with input magnitude (a standalone
probe shows ~100 percent error at unit scale and thousands-fold at larger
scale). The explicit fp32 mean/variance/rsqrt decomposition in
`model/rwkv7_full.py` (`PreciseLayerNorm`, `PreciseGroupNorm`) is exact and is
what the model uses. Do not reintroduce the fused norm ops.

Other runtime facts learned during bring-up, all encoded in the probes:

- The backward of the basic-indexing select behind `x[:, t, :]` fails to
  scatter gradients into a repeatedly-sliced tensor. The `narrow`-based slice
  backward is exact, so the recurrence reads timesteps through `narrow`.
- A CPU backward run before an XLA backward trips a device-index assert in the
  autograd engine. The device backward must run first in any combined probe.
- Unscaled random inputs drive the WKV7 recurrence unstable and amplify dtype
  rounding into chaotic divergence; probe inputs are scaled by 1/sqrt(width)
  to keep the update contractive.
- A crashed or interrupted run can leave a stale sysmem mapping. Clear it with
  `tt-smi -r 0 1` before retrying.
- Container device isolation is not supported; the runner passes all host
  devices and selects one logical `1x1` submesh through the supported device
  API.

## Prerequisites

- The dual-P150A platform smoke receipt is current
  (`checks.x86_64-linux.training-platform`).
- The CPU reference receipt is current
  (`checks.x86_64-linux.rwkv7-lora-reference`).
- The device is idle: no process owns `/dev/tenstorrent/0` or `/dev/tenstorrent/1`.
  If the host inference owner holds a card, isolate it with the sanctioned
  owner-control mechanism before running, and restore it after.
- The runtime wheels (pjrt-plugin-tt, torch, torch-xla, tt_lang) are cached
  locally with sha256 identities matching the platform manifest.

## Commands

Device-free validation (no hardware):

```sh
nix build .#checks.x86_64-linux.rwkv7-lora-p150a
```

This exports the named profile deterministically, rejects the negative
fixtures, runs the core self-test, re-validates every checked-in receipt
against the admitted profile, and re-checks the prerequisite admission.

Compile probes (hardware):

```sh
TOOLS=$(nix build .#packages.x86_64-linux.rwkv7-lora-p150a-tools --no-link --print-out-paths)
PLATFORM=$(nix build .#packages.x86_64-linux.training-platform-tools --no-link --print-out-paths)
"$TOOLS/bin/rwkv7-lora-p150a-probe" \
  --profile "$TOOLS/share/rwkv7-lora-p150a/profile.json" \
  --manifest "$PLATFORM/share/training-platform/manifest.json" \
  --config "$PLATFORM/share/training-platform/platform.json" \
  --descriptor "$PLATFORM/share/training-platform/p150_x2_mesh_graph_descriptor.textproto" \
  --wheels-dir <wheel-cache-dir> \
  --evidence-root <evidence-dir> \
  --probe wkv7-forward --shape-kind representative
```

Probes: `wkv7-forward`, `wkv7-backward`, `training-ops`. Shape kinds:
`focused`, `representative`.

Full-model step, bounded run, and checkpoint (hardware):

```sh
"$TOOLS/bin/rwkv7-lora-p150a-full-step" \
  --profile "$TOOLS/share/rwkv7-lora-p150a/profile.json" \
  --manifest "$PLATFORM/share/training-platform/manifest.json" \
  --config "$PLATFORM/share/training-platform/platform.json" \
  --descriptor "$PLATFORM/share/training-platform/p150_x2_mesh_graph_descriptor.textproto" \
  --checkpoint <pinned-checkpoint.safetensors> \
  --wheels-dir <wheel-cache-dir> \
  --evidence-root <evidence-dir> \
  --mode receipt          # or: multistep, checkpoint
```

- `receipt`: one-step placement + parity receipt.
- `multistep`: bounded multi-step run with loss-improvement criterion.
- `checkpoint`: adapter save/load/resume receipt and the Safetensors artifact.

Each run writes a BLAKE3-bound receipt to the evidence root and validates it
with the admission core before it counts as evidence.

## Failure diagnosis

- `stage ownership failed`: a process owns a device node. Find it with
  `lsof /dev/tenstorrent/0` and stop it, or use the owner-control isolate
  step. Do not proceed while a device is owned.
- `stage step_execute failed` with a sysmem/NOC address error: a stale sysmem
  mapping survived an earlier session. Run `tt-smi -r 0 1` and retry.
- `stage probe_execute failed` with `No matching distribution found`: the wheel
  cache is incomplete or the index is unreachable. Re-resolve the wheels and
  confirm each sha256 against the platform manifest.
- Non-finite logits or loss: confirm the model uses the precise norm
  implementations, not the fused `nn.LayerNorm`/`nn.GroupNorm`.

## Artifact use

The adapter artifact is `safetensors-adapter-only`, the same contract as the
CPU reference. Load it with `safetensors.torch.load_file` and copy the tensors
into a `LoRALinear` adapter whose base checkpoint matches the pinned identity.
Resume is admitted only when the saved manifest and the current inputs agree on
model, tokenizer, LoRA targets, rank, optimizer policy, data identity, and
every adapter tensor BLAKE3.

## Non-claims

A passing receipt does NOT establish:

- Dual-card scaling, model parallelism, or data parallelism.
- Useful generalization or production fine-tuning quality.
- Long-context support or other RWKV model sizes.
- Speedup over the CPU reference.
