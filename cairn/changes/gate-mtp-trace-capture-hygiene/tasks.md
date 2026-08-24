## Phase 1: Hygiene validation core

- [ ] [serial] Add the pure device-free trace-capture-hygiene validation core that rejects a resident second trace, zeroed priming, or a reallocated trace-bound buffer for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [ ] [serial] Add positive and negative device-free tests for the hygiene core, including boundary and malformed-evidence cases, for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].

## Phase 2: Trace control wiring

- [ ] [serial] Wire the hygiene gate into the MTP trace controls behind the existing diagnostic-only gating for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [ ] [serial] Keep eager production behavior unchanged and the gates diagnostic-only for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].

## Phase 3: Evidence

- [ ] [serial] Pass Qwen, package, service, Cairn, and repository-quality checks for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [ ] [serial] Run one authorized bounded P150x2 trace control, restore eager health, and preserve the result for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
- [ ] [serial] Sync the requirement, regenerate the manifest, and archive the change for r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene].
