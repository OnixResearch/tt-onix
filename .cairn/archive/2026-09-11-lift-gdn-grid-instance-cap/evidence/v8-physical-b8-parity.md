# V8: matched physical B=8 decode parity on devices 0 and 1

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V8.

## Result

Two matched runs, both on physical devices 0 and 1 with the GDN grid variant
(`/nix/store/6ccynlvv…-tt-metal-0.77.0`, fused decode forced by
`QWEN35_GDN_DECODE_FUSION=full`, `TT_METAL_DISABLE_SFPLOADMACRO=1`).

Both physical dies report **11 x 10 = 110 compute cores**, so 192
head-instances (B=8 at TP=2) take the chunked path with `ceil(192/110) = 2`
instances per core and 96 active cores.

### Direct op parity at 192 instances (bitwise)

`/tmp/gdn-grid-hw/v8-decode-parity.py`: one `ttnn.transformer.gated_delta_decode`
call with 192 head-instances against eight serialized 24-instance calls (the
in-range mapping) over identical fp32 inputs and a nonzero initial state.

```
{"event": "verdict", "all_heads_exact": true}
v8op_exit=0
```

Output and returned recurrent state are `torch.equal` (bitwise) for every user
on both devices: 16 exact comparisons per device, 32 total.

### Served parity at B=8

The unchanged B=8 batch-admission harness in the profile's matched root order
`single,batch,batch,single`, eight-prompt corpus, one warmup plus three
measured runs per prompt, server held at port 8000:

| Root | Widths served | Responses | Text-exact vs single-1 |
| --- | --- | ---: | --- |
| single-1 | 1 | 32 | reference |
| batch-1 | 8 | 32 | 24/24 measured |
| batch-2 | 8 (and one width-1 group) | 33 | 24/24 measured |
| single-2 | 1 | 32 | 24/24 measured |

Every batch receipt records `decode_path: batched-traced`,
`batch_width: 8`, `fallback_reason: null`, and the distinct
`state_slice` values 0..7 four times each. Every request returned 200.

### Discrimination: the guard still binds on the shared runtime

The same 8-wide group on the SHARED pre-change runtime with the same
`full` gate fails:

```
{"batch_width": 7, "error_type": "BatchDecodeAdapterError", "stage": "batch-failed"}
{"error_type": "BatchDecodeAdapterError", "stage": "request-failed"}   (all eight requests)
```

All eight response files were empty. The variant's width-8 traces could
therefore only have been captured through the chunked op: the guard is
exactly what the shared runtime hits and the variant does not.

## Diagnostic timings (no claim)

Computed from the server receipts, outside the profile's receipt builder, so
they are diagnostics and not a receipt. No admission binding moves:
`physically_admitted` for B=8 stays false and the readmission change owns any
verdict.

| Root | Median aggregate t/s | Median TTFT | Median tail |
| --- | ---: | ---: | ---: |
| single-1 | 17.05 | 1.388 s | 64.4 ms |
| batch-1 | 101.67 | 10.466 s | 83.3 ms |
| batch-2 | 101.66 | 10.252 s | 83.0 ms |
| single-2 | 17.09 | 1.336 s | 63.6 ms |

The rejected 2026-08-26 receipt recorded 99.29 aggregate t/s at B=8 on the
0.74 runtime with the same every-correctness-gate-passed outcome. These
diagnostics are in the same band; whether the near miss on the predeclared
bounds is now a pass is the readmission change's question, not this one's.

## Non-claims

- No admission. This change creates mechanisms and parity evidence only.
- Timings are diagnostics from raw roots, not the profile receipt.
- The B=8 serving path still prefills per user; the batched scan proved here
  is an op-level mechanism result, not a served-prefill claim.
