# Reliability correction: checks and remaining gates

Recorded 2026-09-13 on `gdn-reliability-20260913`, based on `origin/main` at `865d3f3`.
The change remains open. It does not admit or deploy B=8.

## Reproductions and host checks

| Check | Before correction | After correction |
| --- | --- | --- |
| Unsupported admission width | Fake adapter recorded device effects before rejection | 60 positive and negative cases pass |
| Dst capacity | Original installed kernel rejects on an invalid Dst index | Actual replacement body passes bounds and symbolic tile-order checks |
| Undersized Dst fixture | Not an acceptance case | Rejects with the required capacity diagnostic |
| Startup-tail regression | Pass | Pass |
| Receipt builder self-test | Pass | Pass |
| NumPy GDN reference checks | Existing oracle | Positive and negative checks pass |

The service checks cover B4, B8, and B16 profiles, recovery after rejection, disabled admission, execution errors, reset errors, and expired requests.
The installed package runs these checks, not only the source checkout.

The new host fixture compares one, two, and four instances against the per-tile control.
It does not establish tensor-state parity on a device.
The independent source/API review returned no defect in that narrow scope.
Its process reached the timeout after it returned the report, so it is not a clean independent gate exit.

## Build and package evidence

The corrected runtime built successfully:

`/nix/store/qj3jz2dmn22vfzs4hbz1i1ada7xi6jky-tt-metal-0.77.0`

The variant package built successfully:

`/nix/store/hgqgxm6gmy9hvqwb7zb304ffhw1m5ym0-qwen38-0.1.0`

The following Nix targets pass:

- `qwen38-gdn-grid`
- `checks.x86_64-linux.repository-quality-cairn`
- `checks.x86_64-linux.qwen38`
- `checks.x86_64-linux.qwen38-authority-negative`
- `checks.x86_64-linux.qwen38-nixos-service`
- `checks.x86_64-linux.migration-boundary`
- `checks.x86_64-linux.model-target-profile-factory`
- `checks.x86_64-linux.rwkv-layer-framework-parity`

The `qwen36` check aliases also pass.
The shared TT-Metal output remains separate from the candidate runtime.

## Fresh simulated execution

The packaged Blackhole single-chip simulator ran `pkgs/ttsim/gdn-smoke.py` against the corrected runtime.
The run used slow dispatch and `TT_METAL_DISABLE_SFPLOADMACRO=1`, with a named 720-second limit.
It opened simulated chip 0 and compiled the GDN compute kernel with compile arguments `4,4`.
It then stopped with exit code 1 after approximately twelve seconds:

```text
UnsupportedFunctionality: tensix_matmul_op: ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled
```

This is a simulator feature gap, not a passing recurrence test.
The run supplies device compilation evidence but no numerical, state-carry, placement, or performance acceptance.
No hardware attempt followed.
The user approved source access, but Confluence authentication remains unavailable.
See [hazard-source-preflight.md](hazard-source-preflight.md).

## Existing repository blockers

The quality checks also ran on the unchanged primary checkout at `865d3f3`.
They reproduce these existing failures:

- Artifact manifest: expected 333 entries, found 526 on the baseline and 527 in the first changed snapshot.
- Nix format gate: four baseline files fail. This change formats its two touched package files. `flake-modules/onix-packages.nix` and `nixos/module.nix` remain unchanged and fail.
- Training-platform check: the failing derivation is identical before and after this change: `/nix/store/sl6im95x6fzx0k8z161zjlhnzqz8hc9l-training-platform-check.drv`.

No artifact manifest regeneration, unrelated format changes, or training-platform repair occurred.
D3 remains open.

## Production and retained evidence

The production service remains active on its existing package.
Its configured endpoint, `http://127.0.0.1:8000/healthz`, returned HTTP success.
No service isolation, restart, firmware change, or model request occurred during this correction.

Detailed logs and original receipts remain under `.pi/evidence/qwen38/gdn-reliability-20260913/` in the primary checkout.
The four original receipts remain visible, including both rejections.
The old candidate's timing results do not transfer to the corrected runtime.
The task list reopens I3, V6, and V7 where earlier evidence did not satisfy the stated work.
