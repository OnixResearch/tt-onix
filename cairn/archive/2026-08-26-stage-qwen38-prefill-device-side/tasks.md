## Phase 1: Implementation and device-free gates

- [x] [serial] Add the device-staging and RoPE-table patch with the no-trace gate and fallback for r[tenstorrent.qwen38_p150x2.prefill_device_staging].
- [x] [serial] Add device-free source checks for the staging gate, slot-write ownership, and the cached RoPE path for r[tenstorrent.qwen38_p150x2.prefill_device_staging].

## Phase 2: Package validation

- [x] [serial] Build the package with the new patch set and pass the full device-free check for r[tenstorrent.qwen38_p150x2.prefill_device_staging].

## Phase 3: Physical decision

- [x] [serial] Run matched B=1 versus B=4 roots in `single,batch,batch,single` order on devices `0` and `1` for r[tenstorrent.qwen38_p150x2.prefill_device_staging].
- [x] [serial] Bind a passing B=4 receipt on the new execution identity or preserve the exact rejection for r[tenstorrent.qwen38_p150x2.prefill_device_staging].

## Phase 4: Lifecycle completion

- [x] [serial] Restore production health and validate package, Cairn, formatting, and repository gates for r[tenstorrent.qwen38_p150x2.prefill_device_staging].
- [x] [serial] Sync, archive, and commit the validated decision on the isolated branch for r[tenstorrent.qwen38_p150x2.prefill_device_staging].
