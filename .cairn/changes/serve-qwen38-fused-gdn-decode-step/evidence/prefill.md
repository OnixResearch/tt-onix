# Prefill latency and the fused chunk prefill op

Recorded 2026-09-24 on P150x2 with the production batched adapter at batch
capacity 16. This is a diagnostic run. The throwaway probes are
`.pi/tmp/gdn-step/prefill_eval.py`, `prefill_fused_eval.py`, and
`prefill_quality_eval.py`.

## Where time to first token goes

Prefilling one 10-token prompt takes 2.3 to 3.7 s steady state. A Python
profile of one call shows 23,231 TTNN operation calls taking 2.46 s of the
2.73 s total. The GDN chunk prefill (`chunk_gated_delta_rule_seq_adapter` and
its preprocessing) accounts for 1.46 s. Short prompts use the eager
masked-bucket path (bucket 128), so time to first token is almost independent
of prompt length below 512 tokens. Request grouping prefills users serially, so
width-16 groups waited 49 to 100 s for their first token.

## Fused chunk prefill (`ttnn.transformer.chunk_gated_delta_rule`)

The packaged 0.77 runtime already ships this op. Until now the model tree
disabled it with a constant. It now sits behind `QWEN36_GDN_FUSED_PREFILL=1`
(NixOS `gdnFusedPrefill`, default off).

| Prompt tokens | seq prefill (two runs) | fused prefill (two runs) |
| ---: | --- | --- |
| 10 | 2.54, 2.20 s | 1.18, 1.08 s |
| 120 | 2.16, 2.17 s | 1.13, 1.11 s |
| 500 | 2.42, 2.26 s | 1.13, 1.03 s |

The quality check prefilled 12 prefixes of 60 to 900 tokens from three
documents. It then teacher-forced the next 64 real tokens through the fused
decode step.

| Measure | seq | fused |
| --- | ---: | ---: |
| Mean NLL of the forced tokens | 1.3497 | 1.3566 |
| Top-1 agreement with seq | 1 | 0.958 (0.892 to 1.0 per prefix) |

Fused prefill meets the NLL bound used for BF4 (+0.010) but misses its top-1
agreement bound (0.97). It therefore stays opt-in.

## Long prompts

Untraced prompts of 1,025 to 2,048 tokens use the 2,048-row masked bucket. That
bucket failed beside the resident decode buffers. The GDN FIR conv ran out of L1
(a 20 MB intermediate with a largest free block of 127 KB per bank). The MLP
gate matmul's static circular buffers clashed with L1 buffers. The package
before this change failed the same way, and so did the composite decode route
with the fused step disabled. Staging these intermediates in DRAM only moved
the clash, because at 2,048 rows the matmul's circular buffers alone need
nearly all of L1.

`eager-prefill-chunk.patch` sets the untraced chunk to 1,024 tokens
(`EAGER_PREFILL_CHUNK_TOKENS`). Longer prompts now run as full 1,024-token
chunks plus a masked tail. The adapter harness completed prompts of 1,000,
1,450, and 2,000 tokens on both the seq and fused prefill.

| Prompt tokens | seq prefill | fused prefill |
| ---: | ---: | ---: |
| 1,000 | 3.22 s | 2.52 s |
| 1,450 | 9.14 s | 4.07 s |
| 2,000 | 7.23 s | 3.10 s |

The packaged service completed a 1,450-token prompt plus 64 generated tokens
with both settings. The previous package returned `inference failed` for the
same request. The final package, `/nix/store/br4r72ll917z5jb8b04hc0h1v6fy729r-qwen38-0.1.0`,
ran with defaults (step decode, seq prefill). Results:

- Width 1: 48.9 and 49.1 ms per token.
- Width 8: 56.1 and 51.6 ms per token.
- The long prompt completed in both rounds. Its steady time to first token was 8.5 s. The first round included kernel JIT compilation.

The fused step's shared prep buffer moved from L1 to DRAM. The unit test put
the conv-prep program at 24.7 µs and the step at 72.9 µs, a cost of about
0.5 µs per GDN layer.

## Measurement conditions

The host was shared. Its load average ran from about 40 to 95 during these
runs. Host-dispatch-bound prefill timings varied by up to 3x for the same tree,
so small prefill differences are not resolved. The table above comes from one
session. The earlier short-prompt table comes from a quieter session.
