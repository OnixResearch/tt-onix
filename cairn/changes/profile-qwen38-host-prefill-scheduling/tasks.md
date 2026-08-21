## Phase 1: Package-matched host tooling

- [x] [serial] A1 Add the `tt-metal-tracy-tools` package with the minimal fork-source CMake build, capstone compatibility rewrite, and positive/negative build checks. r[tenstorrent.qwen38_p150x2.host_timeline_profile]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile] A2 Export and check the package, wire it into `flake-modules/onix-packages.nix`, and ship both binaries in the qwen36 overlay at `$TT_METAL_HOME/build/tools/profiler/bin`. r[tenstorrent.qwen38_p150x2.host_timeline_profile]

## Phase 2: Deterministic host comparator

- [x] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile] B1 Add `scripts/qwen36-host-timeline-receipt.rs` with identity validation, ranked gap attribution, BLAKE3 receipt hashing, and positive plus negative self-tests. r[tenstorrent.qwen38_p150x2.host_timeline_profile.valid]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile.invalid] B2 Reject malformed zones, missing captures, identity drift, and wrong execution order in the comparator self-tests. r[tenstorrent.qwen38_p150x2.host_timeline_profile.invalid]

## Phase 3: Physical pilot

- [x] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile] P1 Build one immutable package and verify `python -m tracy` discovers the capture tool inside the qwen36 overlay without a device. r[tenstorrent.qwen38_p150x2.host_timeline_profile.disabled]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile.disabled] P2 Run one bounded `eager,candidate,candidate,eager` host-profiled pilot under `ttwkv7-owner-control isolate`, then restore production and verify health. r[tenstorrent.qwen38_p150x2.host_timeline_profile.valid]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile.valid] P3 Generate the deterministic host-timeline receipt, name the strongest checked host-side difference or the exact remaining blocker, and record simulator-free non-claims. r[tenstorrent.qwen38_p150x2.host_timeline_profile]

## Phase 4: Verification and lifecycle

- [x] [serial] V1 Run Nix formatting, Ruff, focused Qwen checks, both receipt self-tests, and repository quality gates. r[tenstorrent.qwen38_p150x2.host_timeline_profile]
- [ ] [serial] [depends:tenstorrent.qwen38_p150x2.host_timeline_profile] V2 Run Cairn validation, proposal/design/tasks gates, sync, archive dry run, archive execution, and post-archive validation. r[tenstorrent.qwen38_p150x2.host_timeline_profile]
