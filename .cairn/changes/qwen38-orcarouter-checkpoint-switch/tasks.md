## Implementation tasks

All tasks remain open. This package records a plan, not download or outage authority.

- [ ] Record approved source selection, authorized HF access, immutable revisions, and complete manifests without exposing credentials. [task:o1] [serial] r[tenstorrent.qwen38_p150x2.orca_source]
- [ ] Add loader inventory fixtures for valid native weights, GGUF rejection, missing shards, shape drift, and unsupported dtypes. [task:o2] [after:o1] r[tenstorrent.qwen38_p150x2.orca_format]
- [ ] Add model-bound staging and cache isolation with positive reuse and negative cross-checkpoint, recipe-drift, and interrupted-publication tests. [task:o3] [after:o2] r[tenstorrent.qwen38_p150x2.orca_cache]
- [ ] Verify tokenizer, template, model alias, thinking-token accounting, and supported requests with positive and unsupported-feature fixtures. [task:o4] [after:o3] r[tenstorrent.qwen38_p150x2.orca_service]
- [ ] Preserve disabled MTP and matching-head metadata with negative mixed-cohort and unexpected-load tests. [task:o5] [after:o4] r[tenstorrent.qwen38_p150x2.orca_mtp]
- [ ] Freeze typed quality, memory, timing, sampling, and stability gates before any comparative measurement. [task:o6] [after:o5] r[tenstorrent.qwen38_p150x2.orca_admission]
- [ ] Run device-free baselines and fixtures for malformed manifests, remote-code loading, missing timing, denied effects, and failed rollback. [task:o7] [after:o6] r[tenstorrent.qwen38_p150x2.orca_promotion]
- [ ] Run a representative simulator smoke with same-source numerical checks and state carry, or preserve the exact capability blocker. [task:o8] [after:o7] r[tenstorrent.qwen38_p150x2.orca_admission]
- [ ] Run an authorized matched P150x2 screen with repeated requests, quality results, memory peaks, and complete control/candidate timings. [task:o9] [after:o8] r[tenstorrent.qwen38_p150x2.orca_admission]
- [ ] Record owner admission or rejection, reversible service activation, verified model identity, and the new model-bound optimization baseline. [task:o10] [after:o9] r[tenstorrent.qwen38_p150x2.orca_promotion]

## Evidence coverage

| Requirement | Tasks | Required evidence |
| --- | --- | --- |
| Source | `o1`, `o7` | Native-source approval, complete manifest, restricted-access and substitution rejection |
| Format | `o2`, `o7` | Valid inventory plus GGUF, missing-shard, shape, and dtype rejection |
| Cache | `o3`, `o7` | New-root identity, preserved old root, stale-cache and interrupted-publication rejection |
| Service | `o4`, `o9` | Text contract, actual template, complete token accounting, unsupported-feature rejection |
| MTP | `o5`, `o7` | Disabled route makes no MTP allocations, mixed official/OrcaRouter cohort rejected |
| Admission | `o6`, `o8`, `o9` | Frozen gates, same-source oracle, simulated label or blocker, matched physical results |
| Promotion | `o7`, `o10` | Denied activation, failed-health rollback, stalled-worker rejection, approved final identity |

An admission rejection can close the bounded evaluation only with an explicit scope decision and preserved evidence. It cannot count as a successful switch.
