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

The checked-in receipts do not prove speedup or scaling efficiency. They also
do not prove long-run stability, useful generalization, or other models.

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

## Measurement profile

The profile `rwkv7-lora-dual-p150a-measurement` keeps all parity tolerances
unchanged. It records five dual-replica steps after one compile step. The
receipt reports launch, compile, step, checkpoint, teardown, and end-to-end
wall time.

The receipt also reports mean, p50, p99, and standard deviation for step time
and loss. The speedup ratio uses the CPU global-batch step from the same
process. Each receipt binds both board classes, firmware, PCIe links, power
state, and mesh identity with BLAKE3.

Create a machine-fact values file before a hardware measurement:

```json
{
  "board_identity": "p150a:device-0,p150a:device-1",
  "firmware_version": "<firmware-version>",
  "mesh_descriptor_identity": "p150_x2:1x2",
  "pcie_link": "<device-links>",
  "power_state": "<power-state>"
}
```

Do not put serial numbers in this file. The runner fails before device access
when a required fact is absent.

## Utilization and bottleneck evidence

Measurement mode also requires `--device-telemetry <profiler-telemetry.json>`.
Convert the package-matched TT-Metal operation report into the closed telemetry
schema in `fixtures/device-telemetry.json`. Do not estimate missing fields.

The telemetry names the source columns for math busy, NoC idle, and operation
latency. It records peak and capacity bytes for L1 and DRAM on each logical
device. The operation list must exactly match the latency table. Each operation
contains all observed latency samples. The core computes mean and p99 latency.

A dual measurement also requires the passing one-device measurement profile
and receipt. Use `--single-measurement-profile` and
`--single-measurement-receipt`. The core verifies both receipts and checks that
repeat policy, metric policy, micro-batch, and sequence length match. It then
reports one-device and two-device step time, the ideal two-device line, and the
scaling-efficiency ratio.

The runner writes `utilization-receipt.json` and
`utilization-admission.json`. Their BLAKE3 identities bind both measurements,
the telemetry, memory table, complete operation table, and scaling result.

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
"$TOOLS/bin/rwkv7-lora-dual-p150a-collective" \
  --profile "$TOOLS/share/rwkv7-lora-dual-p150a/profile.json" \
  --evidence-root <evidence-dir>
```

Dual step, bounded run, and checkpoint (hardware):

```sh
"$TOOLS/bin/rwkv7-lora-dual-p150a-full-step" \
  --profile "$TOOLS/share/rwkv7-lora-dual-p150a/profile.json" \
  --checkpoint <pinned-checkpoint.safetensors> \
  --evidence-root <evidence-dir> \
  --mode receipt           # or: measurement, multistep, checkpoint
```

- `receipt`: one dual-replica placement + parity step receipt.
- `measurement`: five measured dual-replica steps and a bound CPU ratio. Use
  `--profile "$TOOLS/share/rwkv7-lora-dual-p150a/measurement-profile.json"`.
  Add `--machine-facts <machine-fact-values.json>`,
  `--device-telemetry <profiler-telemetry.json>`,
  `--single-measurement-profile <single-measurement-profile.json>`, and
  `--single-measurement-receipt <single-measurement-receipt.json>`.
- `multistep`: bounded multi-step run. Requires `--parity-receipt` pointing
  at a passing `receipt`-mode receipt; the runner admits it before launch
  and binds its identity.
- `checkpoint`: canonical adapter save/load/resume receipt and the
  Safetensors artifact.

Run the modes in order: collective probe first, then `receipt`, then
`measurement`, then `multistep --parity-receipt <receipt evidence>/receipt.json`,
then `checkpoint`. Each runner uses the package-owned training-platform authority
and offline wheelhouse. It does not accept replacement platform files or
wheel sources. Each run writes a BLAKE3-bound receipt to the evidence root and
validates it with the admission core before it counts as evidence.

## Data contract

The admitted global batch has `micro_batch * 2` samples of
`sequence_length` tokens. Sample order is the global order; replica `r`
owns samples `[r * micro_batch, (r + 1) * micro_batch)`. Loss is the
weighted mean over active labels across the whole global batch, so padding
cannot change the mean. Local SFT data must map into this shape before
admission; the profile rejects batches that cannot shard evenly.

## Runtime facts learned during dual bring-up

All of these are encoded in the probes and runners:

- Multichip torch-xla on this bundle needs the pinned TT-Blacksmith device
  manager environment: `XLA_STABLEHLO_COMPILE=1`,
  `XLA_ALWAYS_ALLREDUCE=1`, `CONVERT_SHLO_TO_SHARDY=1`, and
  `DISABLE_NUMERIC_CC_TOKEN=1`, plus `xr.use_spmd()`. Without it the
  backend fails with `GSPMD presharded argument missing @Sharding custom
  call`.
- The multichip optimizer step must go through
  `xm.optimizer_step(optimizer, barrier=True)` so the all-reduce stays
  inside the optimizer graph.
- The gradient all-reduce lowers inside the TT backend after SPMD
  partitioning. No pre-partition StableHLO dump can show the collective,
  so collective evidence is behavioral: exact reduced values from distinct
  per-device contributions, the data-axis sharding annotation, and
  numerical parity with the global-batch reference.
- `xs.get_xla_sharding_specs` is not exported by this wheel's public
  `torch_xla.distributed.spmd` namespace. The probes read placement through
  `torch_xla._XLAC._get_xla_sharding_spec`, the same binding TT-Blacksmith
  uses.
- Gradient and other intermediate tensors return an empty sharding spec.
  Replica-equality evidence comes from parameters that the runner marks
  explicitly replicated with a rank-matched all-`None` partition spec. An
  empty tuple spec fails with `Partition spec length (0)`.
- The training-platform `launcher.sh` once extracted the mesh descriptor
  path with `.plan.environment[keys[0]]`, which evaluates `keys` against
  the root object and yields `null`. It now reads the admitted
  `TT_MESH_GRAPH_DESC_PATH` key by name and fails closed when it is absent.
- A crashed or interrupted run can leave a stale sysmem mapping. Clear it
  with `tt-smi -r 0 1` before retrying.

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

- `ambient TT_METAL_HOME is set`: the runners fail closed on ambient
  `PYTHONPATH`, `PYTHONHOME`, `TT_METAL_HOME`, and `TT_METAL_RUNTIME_ROOT`
  because they can silently replace the admitted runtime. Host shells that
  load the NixOS Tenstorrent module export these; unset them for the run
  (`env -u TT_METAL_HOME -u TT_METAL_RUNTIME_ROOT ...`).

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

- Speedup from a parity, run, or checkpoint receipt. Only a passing physical
  `measurement` receipt can report the named CPU ratio.
- Scaling efficiency to more than the bound two-device comparison.
- Long-run stability, thermal endurance, or power efficiency. The utilization
  receipt covers only its bound measured run.
- Useful generalization or production fine-tuning quality.
- Support for other RWKV models or sizes.
