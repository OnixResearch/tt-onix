## Physical result

The matched control ran on physical PCIe devices `0` and `1`, mesh `(1, 2)`, and CPUs `8-31`. It used package `/nix/store/8wv523yzyrg2qvwlhgs5ag9jrl7asfcd-qwen36-0.1.0` and runtime `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`.

The run used one loaded process, five fixed prompts, and route order `reprefill,snapshot,snapshot,reprefill` repeated three times. It produced six complete samples per route and prompt. Eager production was restored after the run and passed `/healthz` on port `8000`.

## Correctness

All 60 measured roots preserved draft, target, continuation, and committed token IDs. Every snapshot root matched re-prefill exactly for the next draft token, feedback tensor, and committed MTP K/V prefix. Target output and target rollback also passed in every root.

Natural accepted prefix lengths remained `3,0,3,3,3`. The control therefore covered full acceptance and the observed zero-accept rejection.

## Timing

- Re-prefill recovery median: `0.046100162508082576 s`.
- Snapshot recovery median: `0.02023772850225214 s`.
- Snapshot/re-prefill ratio: `0.43887204011198166`.
- Observed recovery reduction: `56.11279598880183%`.

## Rejection

The deterministic evaluator rejected snapshot rollback because six prompt-route groups exceeded the `1.25` duplicate-root recovery bound:

- prompt 1 snapshot;
- prompt 2 re-prefill;
- prompt 3 re-prefill;
- prompt 4 re-prefill;
- prompt 4 snapshot; and
- prompt 5 snapshot.

The exact state match and median reduction are diagnostic opportunity evidence. Unstable roots prevent mechanism selection. The result does not admit MTP, traced target decode, or production behavior.

## Receipt

- Path: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-kv-rollback-receipt.json`
- BLAKE3: `ea0b1d77b8c6be61a566778862a12e1faa620a597ec7cad1101a01efee929921`
