## Phase 1: Lifecycle and baseline

- [x] [serial] Create the native Cairn change and record a passing pre-change RWKV baseline. r[tenstorrent.rwkv7_p150x2.device_free_validation]

## Phase 2: Reusable packages

- [x] [serial] Port the full-model replicated P150x2 runtime into the standalone package authority. r[tenstorrent.rwkv7_p150x2.full_device_runtime]
- [x] [serial] Port the bounded command and evidence classifier without machine-specific owner orchestration. r[tenstorrent.rwkv7_p150x2.evidence_classifier]
- [x] [serial] Expose the runtime, classifier, source boundary, and CPU-fallback closure checks through the flake. r[tenstorrent.rwkv7_p150x2.device_free_validation]

## Phase 3: Profiler isolation

- [x] [serial] Add the pure profiler environment policy and make the process shell remove Watcher and DPrint variables. r[tenstorrent.rwkv7_p150x2.profiler_isolation]
- [x] [serial] Make the C++ runtime reject every present Watcher or DPrint value during profiling. r[tenstorrent.rwkv7_p150x2.profiler_isolation]
- [x] [serial] Add positive and negative tests for absent, zero-valued, empty, and enabled debug variables. r[tenstorrent.rwkv7_p150x2.profiler_isolation]

## Phase 4: Historical evidence

- [x] [serial] Add and validate a bounded immutable classifier fixture for attempt 3. r[tenstorrent.rwkv7_p150x2.attempt3_retention]

## Phase 5: Validation and integration

- [ ] [serial] Run focused package, source, closure, retained-evidence, and historical RWKV checks. r[tenstorrent.rwkv7_p150x2.device_free_validation]
- [ ] [serial] Run Cairn validation and gates, sync the accepted specification, archive the change, and integrate the verified branch. r[tenstorrent.rwkv7_p150x2.no_hardware_claim]
