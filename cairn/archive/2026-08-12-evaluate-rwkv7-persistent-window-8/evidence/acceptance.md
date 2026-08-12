# RWKV-7 persistent decode window-8 acceptance

## Decision

Admit production persistent decode windows `[2, 4, 8]`.

The admitted profile is `rwkv7-p150x2-persistent-decode-v3`. Its BLAKE3 identity is `c5bfa37d83c026bfdb9255da4810bc90af71c7a0e949f05c9c5bf3e709654568`.

The evaluation profile remains `rwkv7-p150x2-persistent-decode-window8-candidate-v1`. Its BLAKE3 identity is `24c07fd0fee91e5e1f290c052f52492680e7edcde30230f3ed91bd5a711102f7`.

## Acceptance evidence

- CPU authority passed with continuation `[36786, 34, 308, 459, 39743, 4811, 31063, 32227]`.
- FLA and official RWKV-LM produced the same tokens.
- The maximum recurrent-state difference was `1.239776611328125e-05`.
- The maximum final-logit difference was `3.0517578125e-05`.
- The CPU receipt BLAKE3 is `eb5624294919b43d487e30d6bbca1555bbebbcf2ec71a8504c60b57909fc9da8`.
- Candidate and admitted-profile simulator receipts passed. Each receipt BLAKE3 is `9391d0552f8ebbf03d8cab0aef2b9e0a1d336e146034b4d25f97a4691f049ea4`.
- Completion, cancellation, timeout, and failure probes acknowledged, zeroed, and released every slot.
- The trace profiler produced 520 rows with no dropped markers.
- The profiler CSV BLAKE3 is `888a3fc456056411a77b8be8bed320e3b9b948be0733eec012367205d667822b`.
- Five-sample A/B evidence passed with minimum speedups of `2.972756752992956` one-card and `3.141978481337331` two-card.
- The five-sample A/B receipt BLAKE3 is `b3db7be0520831d385d6b7f343cc0cbe9e1840ee6d1baa66c7d541f215b9404d`.
- The bounded canary completed 72 planned executions in `594.710505009` seconds.
- Canary monitoring accepted 12 candidate events, rejected none, and reported no alerts.
- Canary minimum speedups were `3.0872335969083236` one-card and `3.399025914825305` two-card.
- The final candidate interlock state was `retain` with reason `clean_canary`.
- The canary receipt BLAKE3 is `66eb56df92331a9e8e08e94f3062c1be4bafab8ebe965f65dfb7c1a0b115a218`.

## Final deterministic gates

The following outputs passed:

- Runtime: `/nix/store/vvysqwb12l68ayk0rhcc391b5i1sfdlz-rwkv7-p150x2-runtime-0.1.0`
- Evidence: `/nix/store/z35j7z0rh681s5bvrvcvfjhqrrk3lhw4-rwkv7-p150x2-evidence-0.1.0`
- Candidate runtime: `/nix/store/jcx1dpgrr99hz3kdnmj6wn0q9bb9j40m-rwkv7-p150x2-window8-candidate-runtime-0.1.0`
- Candidate evidence: `/nix/store/hgr3i3xcb1qy7s6im9cnwmnl4kpczq6a-rwkv7-p150x2-window8-candidate-evidence-0.1.0`
- Source boundary: `/nix/store/gwy40glq0k8cl576kb5bhlgaslmszna1-rwkv7-p150x2-source-check`
- Production closure: `/nix/store/xf7dkhamyww2wywcnw2c9z6vld5lw2wf-rwkv7-p150x2-production-closure-check`
- Repository quality: `/nix/store/5bp6g2423d45xbg8ymr5mm4ppkh6f5jc-tenstorrent-repository-quality`

Rust formatting passed. All 142 library tests and all binary tests passed.

## Evidence limits

Simulator receipts do not establish physical placement, determinism, or performance. Candidate evidence does not mutate production admission. The production profile change is the only admission action.
