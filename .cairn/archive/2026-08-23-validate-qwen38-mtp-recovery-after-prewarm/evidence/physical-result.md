# Physical result

## Scope

The authorized control used package `/nix/store/y625zi8mwspwgdcs031nlaizfhf4kmdl-qwen36-0.1.0`.

The package used:

- runtime `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`;
- checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- physical PCIe device IDs `0` and `1`;
- mesh `(1, 2)`;
- CPUs `8-31`;
- five exact prompt prewarms;
- ten discarded persistent warmup roots; and
- sixty measured roots in the accepted balanced schedule.

The command had an `1800`-second bound and a `120`-second termination grace.

## Operator result

The operator isolated the eager service at `2026-08-23T17:35:14-04:00`.

The control returned status `0` at `2026-08-23T17:45:19-04:00`. Both devices closed cleanly, so a reset was not necessary.

The operator restored eager service with status `0`. The service was active and healthy at `2026-08-23T17:46:50-04:00`.

## Control completion

The control completed all prewarm, warmup, measurement, cleanup, evaluation, and receipt boundaries.

Exact target-validation signatures matched for all warmup and measured roots. One persistent recovery trace capture served the measured control.

The program-cache count increased from `14` before prewarm to `408` after prewarm. Warmup increased the count to `411`.

Measurement started and ended with `411` entries. No measured root added a program.

The prewarm phase completed at `211.1302879070281` seconds. Warmup completed at `261.8227108370047` seconds.

Measurement completed at `599.5179990340257` seconds. The final ledger boundary completed at `600.3610406240332` seconds.

These elapsed values include diagnostic work. They are not throughput evidence.

## Fail-closed decision

The existing evaluator rejected the control.

All six prompt-2 snapshot roots failed the next-draft, feedback-state, and committed-prefix comparisons. Each failed root produced next draft token `248068` instead of the re-prefill reference token `55404`.

The exact draft, target, verifier, acceptance, commit-width, and rollback signatures still matched. Thus, those signatures do not prove the next MTP state after snapshot recovery.

Prompt 1 passed the `1.25` recovery-spread limit. Its re-prefill spread was `1.2431328607494236`. Its snapshot spread was `1.1343434209094907`.

The following prompt-route groups exceeded the limit:

- prompt 2 re-prefill: `1.2900968181734955`;
- prompt 2 snapshot: `1.2595939780227319`;
- prompt 3 re-prefill: `1.3995387256886076`;
- prompt 3 snapshot: `1.9977290784770578`;
- prompt 4 re-prefill: `1.4563729102362526`;
- prompt 4 snapshot: `1.811340742054217`;
- prompt 5 re-prefill: `2.549808916023789`; and
- prompt 5 snapshot: `1.5964495934030414`.

The aggregate re-prefill recovery median was `0.032089075015392154` seconds. The aggregate snapshot recovery median was `0.011660825490253046` seconds.

The snapshot-to-re-prefill median ratio was `0.38005667764112233`. This lower median does not override the state and stability failures.

The receipt records:

- `rollback_valid=false`;
- `snapshot_rollback_selected=false`;
- `mtp_selected=false`;
- `admission_eligible=false`; and
- `diagnostic_only=true`.

## Evidence identity

The committed receipt is:

`pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-recovery-prewarmed-control-receipt.json`

Its BLAKE3 is `9ab2ceef2cb3cbe710561ba0c56b61ad5155a3ce2159075deb3b37705ed0e24e`.

The committed progress ledger is:

`pkgs/qwen36/evidence/2026-08-23/qwen38-27b-mtp-recovery-prewarmed-control-progress.json`

Its BLAKE3 is `d1d13010d5bd734a6124a6f3e4ddefb67c5783864ca777f8298b905d359cc36d`.

Both committed files match their raw files byte for byte.

## Disposition

Exact prewarm removed cold target-validation compilation from measurement. It did not make persistent snapshot recovery correct or stable.

The snapshot route remains rejected. Persistent MTP recovery remains unadmitted.

Production remains eager. This result does not change the admitted `4.2306764265491275` tokens-per-second baseline.
