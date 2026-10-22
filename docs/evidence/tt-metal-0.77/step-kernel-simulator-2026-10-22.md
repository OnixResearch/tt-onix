# Staged step-kernel simulator attempt — 2026-10-22

## Verdict

Corrected isolated launch: exit **0**, `SIMULATED STEP KERNEL: PASS`. The [receipt](step-kernel-simulator-2026-10-22.json) records all three recurrent steps and the package identities. Simulated mesh opening completed, both program families executed, all convolution states were exact, state relative errors were `0.001534386355145001`, `0.001454896039034123`, and `0.0015221651721497192`; gated-output relative errors were `0.00192821252340541`, `0.0017164542368592064`, and `0.0017277577908767804`. The fresh JIT reported `0/51 hits (0.0%)`; the device closed cleanly. No unsupported-feature diagnostic occurred on this bounded step-kernel path. This does not supersede the separately recorded fused-op simulator blocker.

## Artifact and scope

- Qwen package built from the tree: `/nix/store/a7nlx35sz0pvklx96vbcks55qbaqam18-qwen38-0.1.0`.
- Its wrapper embeds package identity `/nix/store/rymc1wl22dwr1db16948sxlqk1p6wrm3-qwen36-execution-identity-0.1.0` and runtime `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- Simulator: `/nix/store/0qvnw9q02agnjbnm5pwvfpxhmrkqczii-ttsim-1.10.6/lib/ttsim/runtimes/blackhole-x1/libttsim.so` (packaged `libttsim_bh.so`, with the packaged descriptor).
- The packaged `step_reader.cpp` matches the reader extracted from the HEAD patch byte-for-byte: SHA-256 `df4c46da400cdb36a8f7628bd7f6878be9d67aceb432236b5dadf90adb589d12`. HEAD patch SHA-256: `ba72ddbab51e543499c35dddceb5133b73965b04d8693b0fd6f0354be51692c1`.
- Geometry: batch 1, capacity 1, one key head, one value head, key/value dimension 128. Intended execution: three recurrent `GdnDecodeStep.conv_prep` + `recurrence` calls, carrying both convolution and recurrent state, compared against the package's float64 `reference_step`.
- Acceptance bounds taken from the preserved physical harness `.pi/tmp/gdn-step/test_step.py:59-72`: exact convolution state; relative L2 state error `< 2e-3`; relative L2 gated-output error `< 2e-2`. No tolerance is relaxed for simulation.

## Command and confinement

Preparation: `nix build .#ttsim .#qwen38 --no-link --print-out-paths` (exit 0); `nix build nixpkgs#bubblewrap --no-link --print-out-paths` (exit 0).

The throwaway launcher and numerical harness use the existing `pkgs/ttsim/gdn-smoke-check.nix` shape: a writable runtime overlay, packaged simulator wrapper, slow dispatch, and `TT_METAL_DISABLE_SFPLOADMACRO=1`. The existing GDN smoke exercises a different fused op, so the throwaway harness calls the package's unchanged step program builders directly; no production branch or kernel is modified.

```text
python3 /tmp/qwen38-step-simulator-2026-10-22/launch.py
```

The launcher invokes:

```text
/nix/store/v24sx9l1jmvvia3z8iznrd7jwhvc8kcf-bubblewrap-0.11.2/bin/bwrap --ro-bind / / --dev /dev --proc /proc --unshare-pid --tmpfs /tmp --bind /tmp/qwen38-step-simulator-2026-10-22 /tmp/qwen38-step-simulator-2026-10-22 --chdir /tmp/qwen38-step-simulator-2026-10-22/runtime -- /nix/store/0qvnw9q02agnjbnm5pwvfpxhmrkqczii-ttsim-1.10.6/bin/ttsim-blackhole-sfpu timeout --signal=TERM --kill-after=10s 3580 /nix/store/qwm7vyvj1jbr0q65jdmshfqk6k9l4isc-python3-3.13.13-env/bin/python /tmp/qwen38-step-simulator-2026-10-22/smoke.py
```

The named `SIMULATOR_TIMEOUT_SECONDS` is 3580 for the corrected launch, reserving time for the initial approximately ten-second setup failure and kill grace within the aggregate 60-minute bound. The first launch lacked writable sandbox `/tmp`; MPI reported a read-only session directory and the child exited 139. That was an environment failure, not a numerical result. The correction supplies private writable `/tmp` without changing any runtime or kernel code.

The mount namespace replaces host `/dev` with synthetic devices; `/dev/tenstorrent` is absent and checked before mesh opening. Physical accelerator nodes are not available to the child. The filesystem is read-only except for the throwaway workspace and private temporary filesystem. No host service is stopped or restarted.

## Non-claims

This is a simulated attempt, not physical-device execution. It cannot establish physical placement, performance, hardware profiling, production readiness, multi-device correctness, or an admission. A device-open or firmware-compile result alone is not recurrent-state acceptance. Unsupported operations and timeouts are blockers, not passes.
