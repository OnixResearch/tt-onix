## Physical result

The matched P150x2 control ran on physical PCIe devices `0` and `1`, mesh `(1, 2)`, and CPUs `8-31`. It used package `/nix/store/yqi1a8jm0zbfsbmp8yjq3dn5zanfpksf-qwen36-0.1.0` and runtime `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`.

The run used one loaded process, five fixed prompts, and route order `eager,traced,traced,eager` repeated three times. It produced six complete samples per route and prompt. Eager production was restored after the run and passed `/healthz` on port `8000`.

## Correctness

All 60 measured roots preserved route token parity and target output. Every traced root passed address-binding, capture-idempotence, and bounded MTP K/V-prefix checks. Prompt 2 rejected all drafts and exactly restored target recurrent state, convolution state, paged K/V state, continuation, and coherent MTP state.

Accepted draft lengths remained `3,0,3,3,3`, with mean acceptance `2.4`.

## Timing

- Eager draft-decode median: `0.020045544501044787 s`.
- Traced draft-decode median: `0.01038352950126864 s`.
- Traced draft-decode ratio: `0.507276627509049`.
- Total draft ratio: `0.7213822465863816`.
- Trace-capture median: `0.02070958149852231 s`, excluded from replay economics.
- Emitted-token-weighted aggregate speedup: `1.2921427224106086`.

## Rejection

The control rejected traced drafting because all ten prompt-route duplicate-root groups exceeded the `1.25` spread gate in at least one required timing group. Representative failures include:

- Prompt 3 traced draft-decode spread: `6.760791039884577`.
- Prompt 4 eager draft-decode spread: `2.844267015475734`.
- Prompt 4 traced target-readback spread: `5.6169391378076625`.
- Prompt 5 traced target-readback spread: `5.027343805239999`.

The observed median and aggregate gains are diagnostic opportunity bounds. They do not validate a stable trace, admit MTP, or change production.

## Receipt

- Path: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-mtp-draft-trace-receipt.json`
- BLAKE3: `c35ed4dacc2557086d811ee90f14c423f2a89b051bc91f716229dfb9320daddc`
