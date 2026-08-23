# Physical result

## Scope

The authorized control ran one prompt and one recovery route. It used:

- package `/nix/store/b3w97pzld9hg2ndim8y9pkjx4rkx613k-qwen36-0.1.0`;
- runtime `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`;
- checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- prompt index `2`;
- route `snapshot`;
- PCIe devices `0` and `1`;
- mesh `(1, 2)`; and
- CPUs `8-31`.

The command had an `1800`-second bound and a `120`-second termination grace. It returned status `0` and reached the final ledger boundary after `191.3243064180133` seconds.

The operator isolated the eager service before the command. The command closed both devices cleanly, so no device reset was required. The operator restored the service with status `0`. The service was active and returned `{"model": "Qwen3.8-27B", "status": "ok"}` at `2026-08-23T11:43:16-04:00`.

## Program-cache control

The exact prewarm started with `14` program-cache entries. It finished with `406` entries.

The prewarm cache counts changed at these boundaries:

- target prefill: `14` to `175`;
- MTP prefill: `175` to `193`;
- MTP trace capture: `194` to `257`;
- target trace capture: `257` to `311`;
- native verification: `311` to `405`; and
- target-state comparison: `405` to `406`.

The measured pilot started with `406` entries. Recovery setup and state readback added two entries before target validation.

Target validation started with `408` entries. The count stayed at `408` through target reference-state construction, verification prefill, native verification, and target-state comparison.

## Completion boundaries

The exact prewarm completed at `183.81818358396413` seconds. The measured recovery completed at `185.7804959960049` seconds.

The previously blocked measured target-validation boundaries now completed:

- target reference state: `1.9436712540336885` seconds;
- verification prefill: `0.8269895939738491` seconds;
- native verifier: `1.821785319945775` seconds; and
- target-state comparison: `0.6168007919914089` seconds.

The measured validation tail took `5.227060028992128` seconds from target-reference-state start through target-state comparison.

The exact prewarm and measured signatures matched. Both recorded:

- draft tokens `[248045, 74455, 198]`;
- target tokens `[37132, 55404, 369]`;
- verifier tokens `[37132, 55404, 369, 279]`;
- accepted prefix length `0`;
- committed width `1`; and
- rollback applied `true`.

The measured route also recorded:

- persistent trace capture count `1`;
- valid persistent trace bindings;
- valid MTP K/V prefix;
- valid target output;
- recurrent, convolution, and paged-K/V rollback PCC `1.0`; and
- recurrent, convolution, and paged-K/V rollback NMSE `0.0`.

## Conclusion

The prior prompt-2 snapshot case stopped at `target-reference-state-start` after `1200` seconds. This exact same-prompt prewarm control completed the measured target-validation path without adding target-validation programs.

Within this bounded case, the result identifies cold target-validation compilation as the observed blocker. A parked persistent MTP trace and the recovered target state did not prevent warm target validation.

This result does not prove that cold compilation explains every prior noncompletion. It does not establish repeated-root stability, rollback selection, MTP speedup, traced-decode admission, or production readiness.

The prewarm cost remains part of process startup economics. The warm measured times are diagnostic and cannot become throughput evidence.

Production remains eager. The physical budget for this change is exhausted.

## Evidence

- Receipt: `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-target-validation-prewarm-receipt.json`
- Receipt BLAKE3: `04b8087eb04a8e609c0cb3bd83f419a21f1405360659bdb6aa24cf669da8c88a`
- Progress ledger: `pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-target-validation-prewarm-progress.json`
- Progress ledger BLAKE3: `9b95efa146fb38de18fbc7da0a4fb304dd919be579714f2ba8adb752d7e6e3da`

The committed files match the raw files byte for byte.
