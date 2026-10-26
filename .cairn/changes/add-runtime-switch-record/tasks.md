# Tasks: The switch record, its GC roots, and `tt-model check`

## Phase 1: Registry fields

- [x] [serial] R1 Add a package-relative runtime root and execution identity to the profile contract, and record `libexec/tt-metalium` for qwen38 and nanbeige43. r[tenstorrent.runtime_profile.check.runtime]
- [x] [serial] R2 Turn gate entries into records with an optional artifact and digest, and name the model target beside its identity. r[tenstorrent.runtime_profile.check.gate_artifact] r[tenstorrent.runtime_profile.check.model_target]
- [x] [serial] R3 Regenerate the export and extend the registry check: resolve every `model_target_name` through the target registry, and refuse a gate artifact that is not package-relative. r[tenstorrent.runtime_profile.check.valid]

## Phase 2: Check

- [x] [serial] C1 Add `check` with one named item per refusal: profile, package shape/existence/registration/store name, entry binary, runtime root, execution identity, receipts, authority, gates, checkpoint, model target. r[tenstorrent.runtime_profile.check]
- [x] [serial] C2 Keep registration a read-only query, report the method, and let `TT_MODEL_NIX` pin or disable it. r[tenstorrent.runtime_profile.check.package_unregistered]
- [x] [serial] C3 Bound `--deep` at 1 GiB, measure before hashing, print progress, and never hash implicitly. r[tenstorrent.runtime_profile.check.deep.bound] r[tenstorrent.runtime_profile.check.deep.valid]

## Phase 3: Record

- [x] [serial] D1 Write `active.json` atomically, an exclusive history entry, and one root per store path. r[tenstorrent.runtime_profile.record] r[tenstorrent.runtime_profile.record.valid] r[tenstorrent.runtime_profile.record.roots]
- [x] [serial] D2 Refuse a job profile, a structural check failure, an unattended force, and a held lock; bypass only admission failures with `--force`. r[tenstorrent.runtime_profile.record.job] r[tenstorrent.runtime_profile.record.failing_check] r[tenstorrent.runtime_profile.record.force] r[tenstorrent.runtime_profile.record.unattended] r[tenstorrent.runtime_profile.record.lock]
- [x] [serial] D3 Chain `previous` to the prior record digest and refuse to overwrite an unreadable record. r[tenstorrent.runtime_profile.record]

## Phase 4: Status

- [x] [serial] S1 Make `managed` require the package root, the execution-identity root, and any store runtime root to resolve. r[tenstorrent.runtime_profile.status.record.managed] r[tenstorrent.runtime_profile.status.record.unrooted]
- [x] [serial] S2 Keep the legacy contradiction rules and report the record's execution identity. r[tenstorrent.runtime_profile.status.record.contradiction]

## Phase 5: Verification

- [x] [serial] V1 Run every refusal and the record lifecycle in the package install check, unprivileged in `$TMPDIR`. r[tenstorrent.runtime_profile.check] r[tenstorrent.runtime_profile.record]  (evidence/check-verdicts.md)
- [x] [serial] V2 Check the real qwen38 and nanbeige43 profiles on this host and record the true verdicts. r[tenstorrent.runtime_profile.check.valid] r[tenstorrent.runtime_profile.check.receipt_package]  (evidence/check-verdicts.md)
- [x] [serial] V3 Prove the deep bound refuses the real 62 GB checkpoint without hashing it. r[tenstorrent.runtime_profile.check.deep.bound]  (evidence/check-verdicts.md)
- [x] [serial] V4 Write a real record on this host against a temporary state root and read it back. r[tenstorrent.runtime_profile.record.valid] r[tenstorrent.runtime_profile.status.record.managed]  (evidence/record-lifecycle.md)
- [x] [serial] V5 Build the registry check and run the aggregate quality gate. r[tenstorrent.runtime_profile.check.valid]  (evidence/quality.md)

## Verification Coverage

- `Scenario: A candidate satisfies the profile` -> R3, C1, V2, V5
- `Scenario: The profile is unknown` -> C1
- `Scenario: The candidate is not a store path` -> C1
- `Scenario: The candidate is missing` -> C1
- `Scenario: The candidate is not registered` -> C2
- `Scenario: The candidate has the wrong store name` -> C1
- `Scenario: The entry binary is absent` -> C1
- `Scenario: The runtime root or execution identity is absent` -> R1, C1
- `Scenario: An admission receipt is absent` -> C1
- `Scenario: An admission receipt has changed` -> C1
- `Scenario: An admission receipt is no longer admitted` -> C1
- `Scenario: An admission receipt names another package` -> C1, V2
- `Scenario: An authority document has changed` -> C1
- `Scenario: A gate artifact has changed or is absent` -> R2, C1
- `Scenario: The checkpoint is absent or misnamed` -> C1
- `Scenario: The model-target identity moved` -> R2, C1
- `Scenario: A deep re-hash exceeds the bound` -> C3, V3
- `Scenario: A deep re-hash is bounded and bounded-progress` -> C3
- `Scenario: A serve profile is recorded` -> D1, V4
- `Scenario: The profile is a job` -> D2
- `Scenario: The check fails structurally` -> D2
- `Scenario: Only admission failed and the caller forced it` -> D2, V4
- `Scenario: A forced record is unattended` -> D2
- `Scenario: Another writer holds the lock` -> D2
- `Scenario: The roots cannot be written` -> D1
- `Scenario: A rooted record is managed` -> S1, V4
- `Scenario: A root is missing or a package is gone` -> S1
- `Scenario: A legacy source disagrees` -> S2
