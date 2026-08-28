## Why

The fused GDN decode kernel met its speed gates (GDN ratio 0.886, full-step 1.065x) and its state and reset gates, but the previous change rejected it because three of four 64-token prompts flipped one late token each. Field comparison shows every other tuned Tenstorrent hybrid model ships this kernel class: upstream tt-metal 0.77 ships a fused chunked GDN op, the Qwen3.6 bring-up and the Adartras bundle gate on PCC and invariants, and llama.cpp upstream took the same route for RWKV-7.

The exact-token-parity bar stays correct for greedy and deterministic serving. But the production service samples with host-side temperature, and a sampled endpoint has no exact-token contract between runs of the same path. For sampled serving, the correct contract is bounded closeness of the token distribution. The repository today has no gate that can express that, so the fused kernel can never be admitted even for sampled traffic.

## What Changes

- Add a pure, device-free distributional admission core for decode execution-identity candidates.
- Add a closed typed Nickel policy, contracts, and validated twin that own every bound.
- Add positive and negative device-free probes to the qwen36 check.
- Bind the new policy files into the package execution identity.
- Define the spec requirement: a candidate admitted under this gate serves sampled requests only. Greedy and deterministic requests keep the exact-parity path.

This change admits nothing by itself. The fused GDN candidate stays default-off until the probe harness records matched physical evidence and the owner reviews the threshold constants.

## Impact

- **Files**: `pkgs/qwen36/service/qwen36_service/kernel_distribution_admission_core.py`, `pkgs/qwen36/kernel-distribution-admission-{contracts,policy,policy-validated}.ncl`, `pkgs/qwen36/check.py`, `pkgs/qwen36/default.nix`, this change directory.
- **Testing**: qwen36 device-free check (new probe), pinned Cairn strict validation, Nickel export of the validated policy.
- **Production**: unchanged. No routing, admission, or deployment decision changes in this change.
