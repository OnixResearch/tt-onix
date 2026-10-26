## Why

The user asked to run `Nanbeige/Nanbeige4.3-33B` on the two Blackhole P150
cards. The pinned revision `c1093c501cc6460427c0ac634db7223d0450c0ee` is a
plain `LlamaForCausalLM` (no custom code): 52 layers, hidden 7168, 56 query
heads, 8 KV heads, head size 128, MLP 22016, vocabulary 166912, untied LM head,
`rope_theta` 1.2e8 with `rope_scaling: null`, 33,118,977,024 BF16 parameters
(66.24 GB). The verified checkpoint is at
`/home/brittonr/models/nanbeige4.3-33b-c1093c501cc6`.

The packaged TT-Metal 0.77.0 source already carries a generic HF Llama path,
`models/tt_transformers`. With TP2 on a `1x2` mesh, every shape divides. The
weights do not fit in BF16: every parameter in BF16 plus an 8k KV pool needs
33.99 GB per device, against 34.23 GB of addressable GDDR per P150. They fit with the upstream `accuracy` preset (BF16
attention, BFP8 MLP): about 21.9 GB per device at an 8k KV pool. The repository
has no package that runs `tt_transformers` for a dense Llama model on the P150x2
mesh, so no model can use this path yet.

## What Changes

- Add `pkgs/nanbeige43`. It installs the runtime-matched
  `${tt-metal.src}/models` tree, with no upstream patches at first. It uses
  the same Metalium overlay, Python environment, and wrapper convention as
  `pkgs/qwen36`, and adds a small repo-authored (LGPL-2.1-only) runner. The runner calls
  `models.tt_transformers.tt.common.create_tt_model` and `Generator`.
- Add a pinned checkpoint authority for the revision. It records the shard and
  metadata SHA-256 values from the download receipt, so a drifted checkpoint is
  rejected before any device work.
- Add commands for the validation ladder: a device-free check, a CPU reference
  generator, a reduced-layer checkpoint tool, a hardware-neutral `1x1` smoke
  test for TTsim, and the fail-closed production `nanbeige43-p150x2-generate`
  command on an exact `1x2` mesh with physical devices 0 and 1.
- Carry an Apache-2.0 TT-Metal patch only if a later run observes a failure.
  Record the failure in this change.

## Impact

- **Files**: new `pkgs/nanbeige43/`, an overlay/flake export entry, and this
  change directory. The `qwen36` package and its services stay unchanged.
- **Testing**: the device-free check (package install check and command), the
  CPU reference, the simulated `1x1` reduced-layer smoke, then one bounded
  physical `1x2` attempt with teacher-forced comparison against the CPU
  reference.
- **Production**: no service is added or restarted. The masked
  `qwen38-p150x2.service` and the cards are touched only in the physical phase,
  under the standing maintenance authority.
