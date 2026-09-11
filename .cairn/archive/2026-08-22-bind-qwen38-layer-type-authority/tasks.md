## Phase 1: Authority boundary

- [x] [serial] A1 Add a pure layer-type resolver, retain only the exact legacy fallback, and remove the unused mapper set for r[tenstorrent.qwen38_p150x2.layer_type_authority].
- [x] [serial] A2 Add positive and negative device-free tests for complete Qwen3.8 dispatch and invalid authority inputs for r[tenstorrent.qwen38_p150x2.layer_type_authority].

## Phase 2: Evidence and documentation

- [x] [serial] D1 Record the adopted-source dispatch dataflow and correct the reversed repeated-sample labels for r[tenstorrent.qwen38_p150x2.layer_type_authority].

## Phase 3: Validation and lifecycle

- [x] [serial] V1 Run focused Qwen, authority, Cairn, Python, artifact, and repository-quality gates for r[tenstorrent.qwen38_p150x2.layer_type_authority].
- [x] [serial] V2 Sync the accepted requirement, archive this change, commit, push, integrate, and preserve eager production for r[tenstorrent.qwen38_p150x2.layer_type_authority].
