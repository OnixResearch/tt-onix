# Physical result

## Verdict

The packaged 128-token masked fallback is not a valid or useful MTP target verifier on P150x2. MTP remains unselected and unadmitted.

## Bound identity

- Package: `/nix/store/r8sqn9dj0r5c32329zw3jpza96yrf81m-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Checkpoint and tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model source: `d102e7f4512060e09e580e40cf26a870bfa88c55`
- Physical PCIe device IDs: `0`, `1`
- Mesh: `(1, 2)`
- CPU affinity: `8-31`
- Samples: five for each width `1`, `2`, and `4`
- Authority BLAKE3: `c6146a5b5ea290afad2f5abbd99e73e987ed0d85e7d8df9ffb3b4ec0c56ee444`
- Receipt BLAKE3: `1750300f560a9f11bd479a057cf85d1f40a3dc51a73769de6600176f041dfe3a`

## Result

The runtime reports `runtime_supports_native_short_gdn_verify = false`. The diagnostic therefore used the existing 128-token masked prefill fallback. It released the parked trace before verifier allocations and recaptured the trace for each sequential sample.

Every width failed target-token parity. Width one produced target token `623`, while traced sequential decode produced `89454`.

The hybrid state also failed:

| Width | Recurrent PCC | Recurrent NMSE | Convolution PCC | Convolution NMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.9977930379508367 | 0.006652461864978162 | 0.853000382866104 | 0.369715937928293 |
| 2 | 0.9947017894585628 | 0.015245893040781115 | 0.8190965203538724 | 0.45411309589493515 |
| 4 | 0.9940086238177922 | 0.018922876020279887 | 0.7756611409995481 | 0.5698385668219748 |

The fallback did not amortize target execution:

| Width | Sequential median, seconds | Fallback median, seconds | End-to-end amortization |
| --- | ---: | ---: | ---: |
| 1 | 0.06181934999767691 | 1.5275257339817472 | 0.04047025108803542 |
| 2 | 0.12026785401394591 | 1.2534450989915058 | 0.09594983785944096 |
| 4 | 0.2376689650118351 | 1.2836667579831555 | 0.18514849242123463 |

Each width failed at least one `1.25` timing-spread check. The deterministic parity and state failures are also present at every width.

## Disposition

Do not load the MTP drafter on this execution base. A later change must provide a native short-width hybrid verifier first. It must preserve GDN recurrent and convolution state, paged-attention continuation, persistent trace buffers, and per-position target logits.

The result does not reject MTP as an architecture. It rejects only the packaged masked fallback as its target-verification base.

Production was restored to eager mode. `qwen38-p150x2.service` is active, and port `8000` returns a healthy response.
