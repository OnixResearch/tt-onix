## Why

The matched physical prefill attribution receipt
(`pkgs/qwen36/evidence/2026-08-21/qwen38-27b-prefill-profile-attribution-receipt.json`)
found that eager and candidate device programs are nearly identical: the summed
program-median difference explains only `0.225%` of the measured
`1.024 s` wall-prefill gap. The exact recorded blocker is missing host-side
attribution: enqueue gaps, command-queue idle time, and non-program barrier
spans around candidate prefill. No optimization can be selected responsibly
until those host spans are observable.

The packaged TT-Metal Python `tracy` module already instruments every host C++
zone (`EnqueueProgram`, command-queue finish, dispatch), but the Nix tt-metal
package intentionally omits Tracy's CLI tools, so no capture tool exists and
the packaged module cannot record anything.

## What Changes

- Add a `tt-metal-tracy-tools` package that builds `tracy-capture` and
  `tracy-csvexport` from the package-matched tenstorrent/tracy fork vendored in
  the tt-metal source tree, with system capstone/zstd instead of the upstream
  GUI/web dependency stack.
- Ship the two binaries inside the qwen36 runtime overlay at
  `$TT_METAL_HOME/build/tools/profiler/bin`, where the packaged
  `python -m tracy` resolves them.
- Provide a deterministic host-timeline comparator that converts validated
  captures into ranked per-phase host spans and inter-enqueue gaps for matched
  balanced roots.
- Keep all host-timeline evidence diagnostic-only: production behavior,
  admission gates, and TTFT semantics remain unchanged.

## Impact

- **Files**: `pkgs/tt-metal-tracy-tools/{default.nix,CMakeLists.txt}`,
  `flake-modules/onix-packages.nix`, `pkgs/qwen36/default.nix`,
  `scripts/qwen36-host-timeline-receipt.rs`,
  `cairn/specs/qwen38-p150x2-runtime/spec.md`
- **Testing**: `checks.x86_64-linux.tt-metal-tracy-tools-package`,
  `checks.x86_64-linux.qwen36`, receipt self-tests, Cairn validation gates,
  one bounded physical pilot under authorized owner-control isolation.
