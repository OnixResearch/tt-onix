# TT-Metal 0.77 port candidate

## Status

A later [approved hardware attempt](evidence/tt-metal-0.77/hardware-attempt.md) stopped at an apparent stale device lock. It produced no numerical result. The sections below record the earlier simulator-only port.

The 0.77 package builds. The Qwen device-free checks and the basic Blackhole simulator smoke pass. The fused GDN numerical smoke remains blocked by a TTsim limitation. This branch is not a production runtime release.

No production service restart, hardware probe, kernel-module activation, firmware update, or push ran during this port.

## Package changes

The package now pins TT-Metal `v0.77.0`, including its submodules:

```text
source SHA-256: sha256-5NY5orbs7w9BTOUsylg8eAXIxJK+smthhkjd6HFePGA=
UMD submodule: 67b3c0da2bd1e6d960b62dfdd5be98c0772811e0
nanobind: 86e5626728fc282637a6c338597120913dded1bb (2.14.0)
nanobind SHA-256: sha256-jAg/CotM361gc+5Ymsz8jP0ddLBzrFTxCkssJKFe0NY=
```

The port makes these changes:

- Moves GDN Python registration into the transformer-owned `sources.cmake` list. The central TT-NN source list no longer owns this registration.
- Installs the GDN kernel files through the current transformer CMake rule.
- Removes local cadical and header-only patches that upstream already includes.
- Removes obsolete Python-discovery and libdwarf hunks from the package-discovery patch.
- Keeps patch application at zero fuzz.
- Preserves Tracy's generated debug-category header while the separate CLI and WASM builds remain disabled in TT-Metal.
- Updates the Blackhole harvesting patch for the `MetalEnvImpl` member method.
- Derives the separate Tracy tools version from TT-Metal.

The GDN compute, reader, and writer source files remain byte-identical to the installed 0.74 baseline. A direct directory comparison verified this property. A second patch application failed as expected.

SFPI stays at the locally carried 7.70.0 release. The port does not change kernel equations, precision, or synchronization.

## Preserve Qwen defaults

TT-Metal 0.77 exposes both `chunk_gated_delta_rule` and the `fuse_swiglu` keyword. Their presence alone does not establish model correctness.

The added `prefill-runtime-admission.patch` keeps both previously unused paths disabled:

- `FUSED_CHUNK_PREFILL_ADMITTED = False`
- `FUSED_MLP_GATEUP_ADMITTED = False`

Device-free tests cover the disabled defaults, admitted capability fixtures, missing capability fixtures, and the single-device rejection. The existing P150x2 restriction on fused prefill output reduction remains intact.

These flags do not change the existing diagnostic GDN decode gate. The model source stays at `d102e7f4512060e09e580e40cf26a870bfa88c55`.

## Validation

The initial baseline used the prior `b4b4556` package state:

- The existing Qwen package check resolved successfully.
- The new NumPy reference passed its positive and negative host checks.
- The new GDN smoke on TT-Metal 0.74 reached device-kernel execution, then stopped at the TTsim limitation shown below.

The port passed:

- TT-Metal host compilation and package installation.
- Package-matched Tracy CLI compilation through the Qwen dependency.
- Qwen device-free checks, including the revised prefill admission controls.
- The Blackhole 1x1 smoke: device open, firmware compilation, `ttnn.ones`, `ttnn.add`, and clean close.

The new GDN check attempts numerical comparison with a NumPy recurrence, out-of-place state preservation, in-place state carry, program-cache reuse, and invalid-input rejection. The simulator stops before the numerical comparisons complete. The device rejection cases and the final retry therefore remain unverified.

The new Python test passes Ruff. The existing Qwen checker has the same six pre-existing Ruff findings before and after this port. This is not a claim that repository-wide lint passes.

## Exact GDN blocker

Both the 0.74 baseline and the 0.77 candidate stop with this diagnostic:

```text
ERROR: UnsupportedFunctionality: tensix_matmul_op: ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled
```

TTsim 1.10.6 does not model this FP32 matmul control flag. The test uses slow dispatch and disables `SFPLOADMACRO`. Neither setting resolves this separate limit.

The `ttsim-gdn-reference-smoke` check remains failing. It does not convert an unsupported operation into a pass. A basic device-open receipt does not replace recurrent-state evidence.

The 0.77 device compiler also reports deprecated GDN init names. The pinned release retains those aliases. The port deliberately keeps the device source unchanged rather than mixing an API cleanup with the runtime migration.

## Commands and evidence

The saved [basic smoke receipt](evidence/tt-metal-0.77/blackhole-smoke.json) binds TTsim 1.10.6 to TT-Metal 0.77.0. The [GDN diagnostic](evidence/tt-metal-0.77/gdn-blocker.log) records the separate failed gate.

Verified build outputs:

```text
tt-metal: /nix/store/pj7h4adxvyd96gx5l9gsq50wf608m41g-tt-metal-0.77.0
qwen36: /nix/store/5r6npipz2ic1k40jhdb4cjhn8sxvjn3x-qwen36-0.1.0
runtime bundle check: /nix/store/6rs46fxys02pgfnn1g2w668llgcy54j0-ttsim-runtime-bundle-check
```

Run the passing package and device-free checks with:

```console
nix build .#tt-metal \
  .#checks.x86_64-linux.ttsim-runtime-bundle \
  .#checks.x86_64-linux.ttsim-single-chip-smoke \
  .#checks.x86_64-linux.qwen36 --no-link -L
```

Run the GDN check with:

```console
nix build .#checks.x86_64-linux.ttsim-gdn-reference-smoke --no-link -L
```

The second command currently fails with the recorded simulator diagnostic. Full `nix flake check` success is not claimed.

Local logs are under `~/.cache/tt-upgrade-review/evidence/`:

- `metal077-source-prefetch.log` records the source and submodule fetch.
- `metal077-build-r1.log` records the missing Tracy header.
- `metal077-build-r2.log` records the moved descriptor-method error.
- `metal077-build-r3.log` records the first successful host build.
- `gdn074-baseline.log` records the baseline simulator blocker.
- `gdn077-probe.log` records the same blocker with 0.77.
- `metal077-final-checks-r1.log` records the complete candidate checks and the initial Quasar-message assertion mismatch.
- `metal077-accepted-subset.log` records the retry after that assertion repair.

## Remaining acceptance boundary

Numerical GDN acceptance requires a simulator that models the control flag, or a bounded run on an approved idle Blackhole device. The same host and device binaries must run on the simulator and silicon. No simulator-specific kernel branch is permitted.

Before a hardware attempt, verify the installed firmware against the [0.77 installation requirements](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/INSTALLING.md). That document lists firmware 19.8.1 and TT-KMD 2.8.0 or later for Blackhole. These are upstream requirements, not observations of the deployed machine.

Production P150x2 devices require explicit stop approval before a probe. Until that approval or a suitable idle device exists, hardware acceptance remains blocked. The vLLM serving trial remains separate and unstarted.
