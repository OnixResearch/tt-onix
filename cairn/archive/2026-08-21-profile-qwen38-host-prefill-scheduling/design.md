## Context

The prefill attribution change proved that device programs do not explain the
TTFT wall gap. The next attribution surface is host scheduling: the time
between program enqueues, command-queue idle spans, and non-program barrier
work. The packaged tt-metal runtime already carries Tracy client instrumentation
in every host zone, and it ships the Python `tracy` module. Only the capture
and export CLI tools are missing from the Nix package because upstream builds
them through a GUI/web CPM stack (ImGui, GLFW, emsdk) that the Nix build
disables on purpose.

## Goals

- Make `python -m tracy` fully functional against the packaged runtime without
  changing the tt-metal package or its closure.
- Expose per-program host enqueue timestamps and inter-enqueue gaps for eager
  and candidate prefill under one bounded physical pilot.
- Keep every new observation diagnostic-only.

## Non-Goals

- No optimization selection, admission change, or TTFT semantics change.
- No modification of the packaged tt-metal derivation.
- No device-side profiler configuration; host zones only.

## Decisions

### Build tools from the package-matched fork source

`tracy-capture` must read captures produced by the exact Tracy client version
compiled into the packaged runtime (`libtracy.so.0.13.3`, fork revision
`0aaefbb689b4c60694edc905545fc4709fd13f6a`). Building from upstream wolfpld
releases risks protocol drift, so the package compiles
`tt_metal/third_party/tracy/{csvexport,capture}` from the vendored fork inside
the tt-metal source tree. The version banner therefore reads `0.13.3 /
package-matched`.

A minimal top-level CMake project replaces the fork's own subproject files so
`vendor.cmake` never runs. Dependencies map to:

- `TracyGetOpt`, `TracyDtl`: vendored in the fork itself.
- `TracyCapstone`: system capstone via pkg-config. The fork uses capstone 6
  alpha names (`CS_ARCH_AARCH64`, `detail.aarch64`, `AARCH64_OP_IMM`); the
  package rewrites those three identifiers to their stable capstone 5 names in
  a writable copy before configuring. Only stable disassembly APIs are used.
- `libzstd`: system zstd 1.5.7 via pkg-config — the same version upstream pins.
- `PPQSort`: header-only at the fork-pinned `v1.0.6`, patched with the fork's
  own nodebug/semaphore patches.
- GUI-only vendors (ImGui, freetype, GLFW, NFD, md4c, base64, tidy, usearch,
  pugixml, libcurl): not needed by either CLI binary; omitted entirely.

Both tools link `TRACY_NO_STATISTICS` off: csvexport requires the statistics
Worker members that define removes.

### Overlay placement matches the packaged module's expectations

`tracy/common.py` resolves `PROFILER_BIN_DIR` to
`$TT_METAL_HOME/build/tools/profiler/bin`. The qwen36 overlay already exists as
a writable-directory symlink tree, so the install phase adds that directory with
symlinks to the two tool binaries. No wrapper changes.

### Diagnostic-only evidence contract

Host-timeline captures extend the existing prefill profiling non-admission:
profiled responses stay ineligible for TTFT admission, and the pilot runs only
under `ttwkv7-owner-control isolate` with an authorized outage. Production
defaults register no capture tool usage.

### Deterministic comparator

`scripts/qwen36-host-timeline-receipt.rs` parses `tracy-csvexport -u` zone
output, validates identities (runtime, package, model, mesh, root order), and
ranks inter-enqueue gaps and phase totals per decode mode. Positive and
negative self-tests cover valid captures, malformed zones, identity drift, and
wrong execution order. Receipt content hashes use BLAKE3.

## Risks / Trade-offs

- The capstone rename rewrite is narrow (three identifiers in one file); if the
  fork later uses more renamed APIs the build fails visibly rather than silently.
- Tracy capture adds host overhead during diagnostics only; production never
  enables it.
