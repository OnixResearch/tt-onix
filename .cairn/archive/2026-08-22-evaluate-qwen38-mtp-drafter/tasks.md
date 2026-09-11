## Phase 1: Policy and mapping

- [x] [serial] P1 Bind checkpoint tensor shapes, shifted-input semantics, target-hidden authority, cache ownership, source revisions, false completions, and the physical budget for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] P2 Add typed diagnostic policy and a pure evaluator for three drafts, five prompts, acceptance alignment, timing, rollback, stability, and fail-closed non-admission for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].

## Phase 2: Drafter implementation

- [x] [serial] I1 Implement a bounded MTP safetensor loader, shared target embedding and LM head, BF16 FC and norms, and one package-matched full-attention decoder layer for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] I2 Expose post-final-norm target prompt hidden rows and implement shifted MTP prompt-cache initialization plus three aligned draft steps for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] I3 Add the explicit probe shell, packaging, cleanup, and immutable receipt output without changing production wrappers for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].

## Phase 3: Evidence gates

- [x] [serial] T1 Add positive and negative tests for profile shape, tensor mapping, shifted alignment, malformed samples, acceptance, timing, stability, rollback, and non-admission for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] P3 Run one bounded exact-P150x2 loader and alignment smoke, then restore eager production health for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] P4 If the smoke passes, run the five-prompt acceptance and economics pilot for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] I4 If local acceptance and cost pass, implement and verify target GDN prefix rollback plus MTP-cache re-prefill for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] D1 Record the selected route or exact blocker, non-claims, receipts, and operator disposition for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
- [x] [serial] V1 Pass formatting, Python, package, focused Nix quality, Cairn validation, proposal, design, tasks, policy, authority, and artifact gates for r[tenstorrent.qwen38_p150x2.mtp_drafter_probe].
