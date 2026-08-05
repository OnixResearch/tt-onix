# RWKV-7 LoRA data-parallel fine-tuning on two P150A cards

Bounded two-replica data-parallel RWKV-7 LoRA fine-tuning over the physical
`p150_x2` (1x2) mesh, inside the validated dual-card TT-XLA training
platform.

This package proves the smallest complete dual-card claim: one admitted
global batch shards exactly once across two logical replicas, both devices
execute the complete trainable graph, adapter gradients synchronize through
the on-device mean reduction before clipping and the optimizer update, the
result matches the single-device CPU global-batch reference, a bounded
multi-step run improves loss with equal replicas, and one canonical adapter
checkpoint saves, loads, and resumes both replicas.

It does not prove speedup, scaling efficiency, long-run stability, useful
generalization, or support for other models.

## What this establishes

The admitted evidence for the named profile `rwkv7-lora-dual-p150a`:

- **Authority chain**: the dual runner binds the current platform smoke
  receipt, the CPU reference receipt, and the single-P150A step, run, and
  checkpoint receipts. A stale or failing prerequisite blocks launch.
- **Collective probe**: the two-replica logical mesh runs an exact mean
  all-reduce (3.0 and 5.0 reduce to exactly 4.0) and a barrier over both
  devices with TT sharding placement. A single-device or host fallback
  cannot produce the exact reduced value.
- **Shard plan**: the pure planner assigns every global-batch sample to
  exactly one replica under the even contiguous policy, with BLAKE3
  identity. Odd batches and duplicate samples are rejected before device
  access.
- **Parity step**: one dual-replica forward/backward/clip/AdamW step
  matches the CPU global-batch reference within the named tolerances and
  pass fractions, with replicated gradients and unchanged base tensors.
- **Bounded run**: the named multi-step profile keeps every value finite,
  mutates only the LoRA adapters, runs one gradient reduction per step,
  verifies replica equality at every optimizer boundary, and satisfies the
  loss-improvement criterion.
- **Canonical checkpoint**: after verified replica equality, one adapter
  artifact saves to Safetensors, loads byte-identical, and resumes both
  replicas to a result that matches the uninterrupted run.

Every receipt binds the exact runtime, profile, shard plan, and device
identities with BLAKE3 and is re-validated by the device-free check.

## Cable and mesh prerequisites

- Two Blackhole P150A cards connected as one 1x2 mesh (QSFP-DD fabric
  cables), visible as `/dev/tenstorrent/0` and `/dev/tenstorrent/1`.
- The `p150_x2` mesh graph descriptor from the packaged TT-Metal.
- The platform smoke receipt is current
  (`checks.x86_64-linux.training-platform`).
- The CPU reference receipt is current
  (`checks.x86_64-linux.rwkv7-lora-reference`).
- The single-P150A receipts are current
  (`checks.x86_64-linux.rwkv7-lora-p150a`).
- Both devices are idle: no process owns `/dev/tenstorrent/0` or
  `/dev/tenstorrent/1`.
- The runtime wheels (pjrt-plugin-tt, torch, torch-xla, tt_lang) are cached
  locally with sha256 identities matching the platform manifest.

## Commands

Device-free validation (no hardware):

```sh
nix build .#checks.x86_64-linux.rwkv7-lora-dual-p150a
```

This exports the named profile deterministically, rejects the negative
fixtures, runs the core self-test, plans the admitted global batch, binds
the prerequisite chain, and re-validates every checked-in dual receipt
against the admitted profile.

Collective probe (hardware):

```sh
TOOLS=$(nix build .#packages.x86_64-linux.rwkv7-lora-dual-p150a-tools --no-link --print-out-paths)
PLATFORM=$(nix build .#packages.x86_64-linux.training-platform-tools --no-link --print-out-paths)
"$TOOLS/bin/rwkv7-lora-dual-p150a-collective" \
  --profile "$TOOLS/share/rwkv7-lora-dual-p150a/profile.json" \
  --manifest "$PLATFORM/share/training-platform/manifest.json" \
  --config "$PLATFORM/share/training-platform/platform.json" \
  --descriptor "$PLATFORM/share/training-platform/p150_x2_mesh_graph_descriptor.textproto" \
  --wheels-dir <wheel-cache-dir> \
  --evidence-root <evidence-dir>
```

Dual step, bounded run, and checkpoint (hardware):

```sh
"$TOOLS/bin/rwkv7-lora-dual-p150a-full-step" \
  --profile "$TOOLS/share/rwkv7-lora-dual-p150a/profile.json" \
  --manifest "$PLATFORM/share/training-platform/manifest.json" \
  --config "$PLATFORM/share/training-platform/platform.json" \
  --descriptor "$PLATFORM/share/training-platform/p150_x2_mesh_graph_descriptor.textproto" \
  --checkpoint <pinned-checkpoint.safetensors> \
  --wheels-dir <wheel-cache-dir> \
  --evidence-root <evidence-dir> \
  --mode receipt           # or: multistep, checkpoint
```

- `receipt`: one dual-replica placement + parity step receipt.
- `multistep`: bounded multi-step run. Requires `--parity-receipt` pointing
  at a passing `receipt`-mode receipt; the runner admits it before launch
  and binds its identity.
- `checkpoint`: canonical adapter save/load/resume receipt and the
  Safetensors artifact.

Run the modes in order: collective probe first, then `receipt`, then
`multistep --parity-receipt <receipt evidence>/receipt.json`, then
`checkpoint`. Each run writes a BLAKE3-bound receipt to the evidence root
and validates it with the admission core before it counts as evidence.

## Data contract

The admitted global batch has `micro_batch * 2` samples of
`sequence_length` tokens. Sample order is the global order; replica `r`
owns samples `[r * micro_batch, (r + 1) * micro_batch)`. Loss is the
weighted mean over active labels across the whole global batch, so padding
cannot change the mean. Local SFT data must map into this shape before
admission; the profile rejects batches that cannot shard evenly.

## Failure diagnosis

- `stage ownership failed`: a process owns a device node. Find it with
  `fuser /dev/tenstorrent/0` and stop it before you retry.
- `stage probe_execute failed` with a sysmem/NOC address error: a stale
  sysmem mapping survived an earlier session. Run `tt-smi -r 0 1` and
  retry.
- `stage step_execute failed` with `mesh graph descriptor file not found`:
  the descriptor mount or the `TT_MESH_GRAPH_DESC_PATH` value is wrong.
  Check that the runner passes the launch-plan mount path.
- `GSPMD presharded argument` or shardy errors: the multichip environment
  is incomplete. The runner sets `XLA_STABLEHLO_COMPILE`,
  `XLA_ALWAYS_ALLREDUCE`, `CONVERT_SHLO_TO_SHARDY`, and
  `DISABLE_NUMERIC_CC_TOKEN` per the pinned TT-Blacksmith device manager.
- Non-finite logits or loss: the model uses the precise norm
  implementations from the single-card package. Do not reintroduce the
  fused `nn.LayerNorm`/`nn.GroupNorm`.

## Recovery

If a run fails before `probe_execute`/`step_execute`, no device state
changed. Read the admission diagnostic in the evidence root.

If a run fails inside the container, the container is removed by `--rm`.
Check that no process owns the device nodes before you retry. The runner
runs the same check in its `cleanup` and `restore` stages.

If a fabric or device error persists, run `tt-smi -r 0 1` once, wait for
re-initialization, and retry. Do not flash firmware from this package.

## Artifact use

The canonical adapter artifact is `safetensors-adapter-only`, the same
contract as the single-card package and the CPU reference. It is saved
only after verified replica equality. Load it with
`safetensors.torch.load_file` and copy the tensors into a `LoRALinear`
adapter whose base checkpoint matches the pinned identity. Resume is
admitted only when the saved manifest and the current inputs agree on
model, LoRA targets, rank, optimizer policy, data identity, and every
adapter tensor BLAKE3.

## Non-claims

A passing receipt does NOT establish:

- Speedup over one P150A or over the CPU reference.
- Scaling efficiency to more devices.
- Long-run stability or thermal endurance.
- Useful generalization or production fine-tuning quality.
- Support for other RWKV models or sizes.
