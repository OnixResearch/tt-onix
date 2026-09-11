## Phase 1: Hygiene validation core

- [x] [serial] Add the pure device-free trace-capture-hygiene validation core that rejects a resident second trace, zeroed priming, or a reallocated trace-bound buffer for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [x] [serial] Add positive and negative device-free tests for the hygiene core, including boundary and malformed-evidence cases, for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].

## Phase 2: Trace control wiring

- [x] [serial] Wire the hygiene gate into the native MTP verifier capture path behind its bounded execution contract for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [x] [serial] Keep eager production behavior unchanged and the gates diagnostic-only for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].

## Phase 3: Evidence

- [x] [serial] Pass Qwen, package, service, Cairn, and repository-quality checks for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [x] [serial] Sync the requirement, regenerate the manifest, and archive the change for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
