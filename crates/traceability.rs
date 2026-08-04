#![forbid(unsafe_code)]

//! Maps the repository's `pkgs` layout into Cairn's canonical `crates` scan profile.
//!
//! Each marker maps to a constant that names its executable implementation or check.
//! This index does not replace the matching marker in that source file.

pub const FULL_DEVICE_RUNTIME_SOURCE: &str = "pkgs/rwkv7-p150x2/src/main.cpp";
// r[impl tenstorrent.rwkv7_p150x2.full_device_runtime]
// r[verify tenstorrent.rwkv7_p150x2.full_device_runtime.complete_path]

pub const EVIDENCE_CLASSIFIER_SOURCE: &str = "pkgs/rwkv7-p150x2-evidence/src/lib.rs";
// r[verify tenstorrent.rwkv7_p150x2.full_device_runtime.invalid_mesh]
// r[impl tenstorrent.rwkv7_p150x2.evidence_classifier]
// r[verify tenstorrent.rwkv7_p150x2.evidence_classifier.complete]
// r[verify tenstorrent.rwkv7_p150x2.evidence_classifier.missing]

pub const PROFILER_POLICY_SOURCE: &str =
    "pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-command.rs";
// r[impl tenstorrent.rwkv7_p150x2.profiler_isolation]
// r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.clean]
// r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.watcher_zero]
// r[verify tenstorrent.rwkv7_p150x2.profiler_isolation.dprint_present]

pub const DEVICE_FREE_CHECK_SOURCE: &str = "flake-modules/onix-packages.nix";
// r[impl tenstorrent.rwkv7_p150x2.device_free_validation]
// r[verify tenstorrent.rwkv7_p150x2.device_free_validation.focused]
// r[verify tenstorrent.rwkv7_p150x2.device_free_validation.cpu_fallback]

pub const ATTEMPT_THREE_SOURCE: &str =
    "pkgs/rwkv7-p150x2-evidence/src/bin/rwkv7-p150x2-attempt3.rs";
// r[impl tenstorrent.rwkv7_p150x2.attempt3_retention]
// r[verify tenstorrent.rwkv7_p150x2.attempt3_retention.recorded]
// r[verify tenstorrent.rwkv7_p150x2.attempt3_retention.false_success]

pub const NO_HARDWARE_CLAIM_SOURCE: &str =
    "cairn/changes/enable-rwkv7-p150x2-full-device/design.md";
// r[verify tenstorrent.rwkv7_p150x2.no_hardware_claim]
// r[verify tenstorrent.rwkv7_p150x2.no_hardware_claim.complete]
