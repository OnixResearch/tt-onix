# S2 physical screen result (2026-08-27)

Status: **screen executed; receipt rejected under the current thresholds on
one bound**. Owner decision S3 required before any admission.

## Window

- Package: `/nix/store/lk1agynj00jfa5sskdx7jb4x2wm4d6dk-qwen36-0.1.0`
  (identity `dfdgl49i...`)
- Runtime: packaged tt-metal 0.74.0
- Devices: 0 and 1, P150x2, firmware 19.11.0
- Production unit `qwen38-p150x2.service`: inactive before, inactive after
  (pre-screen state preserved; NOPASSWD stop used only if it had been running)
- Root order: `control,candidate,candidate,control` for both the
  distributional probe and the decode-component screen
- Wall time: 24m32s for 8 physical roots

## Component speed gates: PASS

- GDN ratio `0.871081` (bound at most 0.95)
- Full-step ratio `0.93958` (bound at most 0.97), speedup `1.0643x`
- Duplicate-root spread `1.065139` (bound at most 1.25)

## Distributional receipt: 8 of 9 bounds PASS, one bound FAILS

- mean logit PCC `0.996948` (floor 0.995) PASS
- **min logit PCC `0.802772` (floor 0.99) FAIL**
- mean top-K TV `0.018480` (cap 0.10) PASS
- max top-K TV `0.227952` (cap 0.25) PASS
- mean residual mass `0.022562` (cap 0.50) PASS
- stable-position argmax agreement `1.0` over 220 positions (floor 0.95) PASS
- near-tie positions: 20, agreement 0.70 (reported, not gated)
- duplicate stability: control roots 1.0, candidate roots 1.0 (240 positions)

**The fused kernel never flips a confident argmax in 220 stable positions.**
Flips happen only at near-tie margins, exactly where the archived root cause
predicted drift crossing small top1-top2 margins.

## The failing bound, characterized

9 of 240 positions fall below the per-position PCC floor of 0.99. Worst 8
PCC values: 0.7981, 0.9650, 0.9679, 0.9781, 0.9824, 0.9878, 0.9881, 0.9887.

Failing positions (prompt, position, PCC, shared top-K tokens, control
margin):

| prompt | position | PCC | shared | margin |
|---|---|---|---|---|
| 0 | 24 | 0.9887 | 19 | 7.250 |
| 0 | 36 | 0.9650 | 15 | 4.125 |
| 1 | 32 | 0.9878 | 19 | 0.250 |
| 1 | 44 | 0.9824 | 20 | 1.250 |
| 1 | 47 | 0.9781 | 20 | 0.250 |
| 2 | 1 | 0.9888 | 19 | 0.188 |
| 2 | 25 | 0.7981 | 16 | 0.375 |
| 2 | 28 | 0.9881 | 19 | 0.812 |
| 2 | 29 | 0.9679 | 15 | 2.938 |

Five of the nine sit at near-tie margins (below or near 0.5 logits), where
tail-token ordering inside the shared top-20 makes Pearson correlation
noisy. Four sit at confident margins (0.98-0.99 PCC at margins 1.25 to 7.25)
with mild tail reordering. Every failing position stays inside the TV cap of
0.25, so the bounded distributional distance remains small everywhere.

## The S3 decision (owner)

The current policy rejects only on the per-position minimum PCC floor. The
options, in increasing permissiveness:

1. **Keep the floor.** The candidate stays rejected; the follow-up is the
   staged precision-preserving kernel design from the archived root cause.
2. **Stratify the PCC floor by margin** (stable positions only). Four
   confident-margin positions still fail 0.99, so this alone does not admit.
3. **Rely on the TV cap plus stable agreement** (drop the min-PCC floor,
   keep the mean-PCC floor). Every measured bound passes. The TV cap of 0.25
   is the direct distributional bound; the min-PCC floor is a correlation
   heuristic that near-tie tails destabilize.

Any change to the bound constants is a policy revision that changes the
execution identity and needs a new receipt under the updated policy.

## Not claimed

- No admission, no routing change, no deployment. The candidate stays
  default-off and greedy/deterministic serving keeps exact parity.
- No accuracy-benchmark claim.
- Raw 238 MB per-root logit captures stay in
  `.pi/evidence/qwen38/gdn-sampled-dist-screen-20260827/` (gitignored); the
  receipt, per-root summaries, and analyses are copied into this evidence
  directory.
