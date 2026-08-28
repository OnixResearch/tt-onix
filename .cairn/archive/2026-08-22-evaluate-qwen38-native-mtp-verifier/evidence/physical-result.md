# Physical result

## Verdict

The traced hybrid packed-decode verifier is a valid target-verification base on P150x2. It is exact at widths `1`, `2`, and `4`. At width four, it is `2.122455302025925` times faster on device and `1.9469233481156285` times faster including diagnostic full-logit readback.

This result does not select or admit MTP. The drafter, local acceptance, rejection rollback, and end-to-end speculative loop remain unmeasured.

## Bound identity

- Package: `/nix/store/q5kbgzrfhgb7ap1bli2x05wvnabbfwfq-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Checkpoint and tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model source: `d102e7f4512060e09e580e40cf26a870bfa88c55`
- Physical PCIe device IDs: `0`, `1`
- Mesh: `(1, 2)`
- CPU affinity: `8-31`
- Samples: five for each width `1`, `2`, and `4`
- Authority BLAKE3: `c6146a5b5ea290afad2f5abbd99e73e987ed0d85e7d8df9ffb3b4ec0c56ee444`
- Final receipt BLAKE3: `2a4f171b36dcf31f4637d48706dfa3e037af100a5ca21bae91f47070b023ac19`
- Untraced receipt BLAKE3: `f87a06702f49047c921f78873d2eeceff1dadd8a478d7fab7f00328ec15ff473`

## Correctness

Every sample produced the same target tokens as sequential traced decode:

- width one: `89454`;
- width two: `89454, 4384`;
- width four: `89454, 4384, 1528, 58377`.

Every sample also reported:

- full-logit PCC `1.0` and NMSE `0.0`;
- GDN recurrent-state PCC `1.0` and NMSE `0.0`;
- GDN convolution-state PCC `1.0` and NMSE `0.0`;
- written paged-KV PCC `1.0` and NMSE `0.0`.

The verifier serializes same-block paged-KV writes. It batches projections, MLPs, and the LM head. It preserves ordered high-precision GDN recurrence inside each layer.

## Performance

| Width | Sequential median, seconds | Verifier replay median, seconds | Diagnostic readback median, seconds | End-to-end median, seconds | Device amortization | End-to-end amortization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.059328629984520376 | 0.06081783497938886 | 0.00875028400332667 | 0.06965781300095841 | 0.9755136795748619 | 0.8517153701581771 |
| 2 | 0.11720624199369922 | 0.07756988698383793 | 0.008836520981276408 | 0.08642445498844609 | 1.5109760572181796 | 1.35616987124036 |
| 4 | 0.23198413103818893 | 0.10929988999851048 | 0.009195436985464767 | 0.11915421902085654 | 2.122455302025925 | 1.9469233481156285 |

Sequential and verifier replay spreads stayed within `1.25` at every width. Full-logit host transfer remains separately visible because it is diagnostic. A speculative loop must keep logits on device for acceptance.

The untraced route was exact but failed economics. Its width-four end-to-end median was `0.7049244520312641` seconds, only `0.3292153965137219` of sequential performance. Tracing is therefore load-bearing.

## Disposition

Select the traced hybrid packed-decode implementation as the target-verification base for the next diagnostic change. Do not load it into production and do not alter eager serving.

At width four, the measured end-to-end budget for drafter generation plus acceptance is approximately `0.11282991201733239` seconds before the speculative iteration becomes slower than four ordinary traced steps. The next change must map the one-layer checkpoint MTP weights, measure local acceptance, implement exact rejection rollback, and compare emitted-token latency against this bound.

Production was restored to eager mode. `qwen38-p150x2.service` is active, and port `8000` returns a healthy response.
