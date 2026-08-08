# RWKV-7 LoRA bring-up on one P150A

Bounded single-device RWKV-7 LoRA fine-tuning on one logical P150A card,
inside the validated dual-card TT-XLA training platform.

This package proves the smallest complete accelerator claim: one admitted SFT
batch, full loss, adapter gradients, and one optimizer step on one logical
P150A device, plus a bounded multi-step run and adapter save/load/resume.

It does not prove two-card scaling, long-context support, or useful
generalization. The checked-in parity receipts do not prove speedup.

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

## Measurement profile

The profile `rwkv7-lora-p150a-single-measurement` keeps all parity
tolerances unchanged. It records five measured steps after one compile step.
The receipt reports launch, compile, step, checkpoint, teardown, and
end-to-end wall time.

The receipt also reports mean, p50, p99, and standard deviation for step time
and loss. The speedup ratio uses a CPU step from the same process and batch.
Each receipt binds the board class, firmware, PCIe link, power state, and mesh
identity with BLAKE3.

Create a machine-fact values file before a hardware measurement:

```json
{
  "board_identity": "p150a:device-0",
  "firmware_version": "<firmware-version>",
  "mesh_descriptor_identity": "single-device-1x1",
  "pcie_link": "<generation-and-width>",
  "power_state": "<power-state>"
}
```

Do not put a serial number in this file. The runner fails before device access
when a required fact is absent.

## Utilization and bottleneck evidence

Measurement mode also requires `--device-telemetry <profiler-telemetry.json>`.
Convert the package-matched TT-Metal operation report into the closed telemetry
schema in `fixtures/device-telemetry.json`. Do not estimate missing fields.

The telemetry names the source columns for math busy, NoC idle, and operation
latency. It records peak and capacity bytes for L1 and DRAM on each logical
device. The operation list must exactly match the latency table. Each operation
contains all observed latency samples. The core computes mean and p99 latency.

The runner writes `utilization-receipt.json` and
`utilization-admission.json`. Their BLAKE3 identities bind the measurement,
telemetry, memory table, and complete operation table. A single-device receipt
cannot make a scaling claim.

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
"$TOOLS/bin/rwkv7-lora-p150a-probe" \
  --profile "$TOOLS/share/rwkv7-lora-p150a/profile.json" \
  --evidence-root <evidence-dir> \
  --probe wkv7-forward --shape-kind representative
```

Probes: `wkv7-forward`, `wkv7-backward`, `training-ops`. Shape kinds:
`focused`, `representative`.

Full-model step, bounded run, and checkpoint (hardware):

```sh
"$TOOLS/bin/rwkv7-lora-p150a-full-step" \
  --profile "$TOOLS/share/rwkv7-lora-p150a/profile.json" \
  --checkpoint <pinned-checkpoint.safetensors> \
  --evidence-root <evidence-dir> \
  --mode receipt          # or: measurement, multistep, checkpoint
```

- `receipt`: one-step placement + parity receipt.
- `measurement`: five measured steps and a bound CPU speedup ratio. Use
  `--profile "$TOOLS/share/rwkv7-lora-p150a/measurement-profile.json"`.
  Add `--machine-facts <machine-fact-values.json>` and
  `--device-telemetry <profiler-telemetry.json>`.
- `multistep`: bounded multi-step run with loss-improvement criterion.
- `checkpoint`: adapter save/load/resume receipt and the Safetensors artifact.

Each runner uses the package-owned training-platform authority and offline
wheelhouse. It does not accept replacement platform files or wheel sources.
Each run writes a BLAKE3-bound receipt to the evidence root and validates it
with the admission core before it counts as evidence.

## Regression gates

The package includes `regression-policy.json` and `baseline-store.json`.
The policy names the mean step time, p99 step time, and mean loss bounds.
The store key binds the full measurement profile and the machine facts.
A profile or machine mismatch fails closed.

The checked baseline uses the device-free check fixture. It tests the gate but
does not claim physical P150A performance. Create a physical baseline from an
admitted measurement before you make a hardware regression claim.

Run one independent measurement first. Then add all three options to a later
measurement:

```sh
--regression-baseline <physical-baseline-store.json> \
--regression-policy "$TOOLS/share/rwkv7-lora-p150a/regression-policy.json" \
--determinism-receipt <independent-measurement-receipt.json>
```

The options are one group. The runner rejects a partial group. A passing run
writes `regression-receipt.json` and `determinism-receipt.json`. The
determinism receipt records the maximum absolute time and loss differences.
It is not a bitwise tensor-equivalence claim.

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
- Speedup over the CPU reference from a parity, run, or checkpoint receipt.
  Only a passing physical `measurement` receipt can report the named ratio.
- Sustained utilization, thermal endurance, or power efficiency. The
  utilization receipt covers only its bound measured run.
