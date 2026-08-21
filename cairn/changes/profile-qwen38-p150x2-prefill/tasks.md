## Phase 1: Evidence frame and baseline

- [x] [serial] A1 Record the exact goal, false completions, approach registry, authority sources, and bounded physical budget. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
- [x] [serial] A2 Run focused Qwen and receipt baselines before runtime edits. r[tenstorrent.qwen38_p150x2.prefill_program_profile]

## Phase 2: Deterministic profiling capability

- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_program_profile] B1 Add the pure record validator and deterministic per-signature, per-chip summary. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_program_profile] B2 Add the opt-in TTNN callback adapter and service-shell profile region. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_program_profile] B3 Mark profile responses as diagnostic and reject them from TTFT admission. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_program_profile] B4 Add positive and negative core, adapter, and response tests. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_profile_evidence] B5 Add a deterministic Rust comparator with positive and negative self-tests. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]

## Phase 3: Physical attribution

- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_profile_evidence] P1 Build one immutable candidate and verify the profiler is active on physical P150x2. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_profile_evidence] P2 Run `eager,candidate,candidate,eager` profiled roots with the bound corpus. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_profile_evidence] P3 Generate the deterministic attribution receipt and identify the strongest checked program difference. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]
- [x] [serial] [depends:tenstorrent.qwen38_p150x2.prefill_profile_evidence] P4 Select one immutable optimization candidate, or return an exact bounded blocker. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]

## Phase 4: Verification and lifecycle

- [x] [serial] V1 Run Ruff, Python compilation, focused Qwen Nix checks, receipt self-tests, and formatting checks. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
- [ ] [serial] V2 Restore eager production and verify the service health endpoint. r[tenstorrent.qwen38_p150x2.prefill_profile_evidence]
- [ ] [serial] V3 Run Cairn validation and lifecycle gates. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
- [ ] [serial] V4 Sync the accepted specification, archive the change, and verify zero active changes. r[tenstorrent.qwen38_p150x2.prefill_program_profile]
