# B=8 re-earn on a buildable identity, 2026-10-26

Recorded 2026-10-26. This is the separately authorized campaign the 2026-10-22
procedure and the 2026-10-26 amendment predeclared. It re-earns the v3
interleaved B=8 receipt on the package this tree builds, promotes the binding to
that fresh receipt, and proves the promoted package imports the same execution
identity.

## What ran

| Input | Value |
| --- | --- |
| Source revision | `2ba2c19eab757267b4843ff5f4842e636ae8cc08` |
| Served package | `/nix/store/wxnsxbhlx59vx4cfwi0881szxjfb32lm-qwen38-0.1.0` |
| Execution identity | `/nix/store/qwnxwcppn16za67i710qm5drnzs57vg9-qwen36-execution-identity-0.1.0` |
| Runtime | `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0` |
| Model snapshot | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Qwen3.8-27B) |
| Measurement profile | unmeasured twin, BLAKE3 `28abfd7c4bdb5196676e52940374f39018f0404cd6969ed1277068ba3653c19f` |
| Bootstrap gate | BLAKE3 `2d12b78a36afccef4a173327d44022e4e50bfe4aeab54210343eb4a2bfbc3db3` |
| Physical devices | PCIe ids `[0, 1]`, no simulation |
| Roots | 8 fresh servers: `single-0 batch-1 single-2 batch-3 single-4 batch-5 single-6 batch-7` |
| Responses | 192 measured (24 per root) plus 64 warmup (8 per root) |

Launch environment was the serving configuration rather than the 2026-10-22
screen recipe: `QWEN35_GDN_DECODE_FUSION=step`, `QWEN35_GROUPED_PREFILL=1`,
`QWEN36_TRACED_PREFILL=1`, the wrapper's own `MESH_DEVICE`/`QWEN35_TEMP`/offline
variables, `HOME`/`TT_CACHE_PATH` at the model snapshot and `--max-sequence-length
2048 --max-generation-tokens 64`. No `TT_METAL_DISABLE_SFPLOADMACRO`, no custom
`PYTHONPATH`, no identity overrides and no external profile were present; each
root's `/proc/<pid>/environ` and `/proc/<pid>/cmdline` are retained verbatim.

## Why this configuration

Two corrections were forced by the recorded receipt and by source, not by
preference:

- **`step`, not `full`.** `qwen38-grouped-prefill-b8-rescreen` g8 records that
  the admitted B=8 receipt is served by the `step` GDN kernel
  (`QWEN35_GDN_DECODE_FUSION=step`). The 2026-10-22 screen recipe selected
  `full`; a `full` campaign reproduced 18.1 tok/s single-stream, exactly the
  documented `full` rate, against the receipt's 23.57. With `step` the same
  corpus reproduces 23.56 tok/s.
- **Grouped and traced prefill on.** The receipt's medians (`single` TTFT
  0.221 s, `batch` TTFT 1.057 s, group first token 1.807 s) are only reachable
  with `QWEN36_TRACED_PREFILL=1` and `QWEN35_GROUPED_PREFILL=1`, and
  `nixos/module.nix` defaults `groupedPrefill`/`tracedPrefill`/`gdnDecodeStep`
  to exactly these values. A `step` campaign with both off produced a batch
  TTFT of 14.2 s and a TTFT ratio of 12.9, outside the 9.0 bound; it was
  stopped before any root completed and is quarantined.

Both corrections are recorded in the campaign `events.jsonl`; the discarded
attempts are retained under the run directory as `aborted-prefill-off/` and
`aborted-fusion-full/` and are not part of the receipt.

## Six gates

| Gate | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| Aggregate decode speedup | 7.4131 | at least 6.0 | pass |
| Per-user decode retention | 0.9266 | at least 0.75 | pass |
| Per-user TTFT ratio | 4.5285 | at most 9.0 | pass |
| Tail-latency ratio | 1.0997 | at most 1.5 | pass |
| Exact token parity | true | exact | pass |
| Exact text parity | true | exact | pass |

Diagnostic only, not a gate: group-max TTFT ratio 7.6900.

Medians: single 23.56 aggregate tokens/s, 0.214 s TTFT, tail 0.0436 s; batch
174.65 aggregate and 21.83 per user, 0.969 s per-user TTFT, 1.741 s group first
token, tail 0.0480 s. Correctness: exact token and text parity, clean reset,
distinct state slices (`0..7` per measured batch run, `0` on every single
sample), active prefix. These match the receipt the 2026-10-22 tree carried
(7.4083 / 0.9260 / 4.5214 / 1.0797) to within the run-to-run spread.

An independent checker (`validate-receipt.py`, outside the builder) re-read the
192 responses, recomputed all 96 single/batch parity pairs from the raw bytes,
and required the identity block, the bounds, the root count and each root's
provenance to match the preflight: `passed`, no failures.

## Receipt and promotion

- Receipt: `pkgs/qwen36/evidence/2026-10-26/qwen38-batch-admission-b8-v3-receipt.json`,
  BLAKE3 `cf9fc9d417091a83bad96c71b58015dcab75529146247664039e6de1758f3e55`,
  `admission.admitted = true` with an empty `reasons` array.
- `pkgs/qwen36/batch-admission-b8-binding-v3.ncl` now names that digest and
  `physically_admitted = true`; `pkgs/qwen36/default.nix:1825` pins the new
  receipt path.
- Promoted package: `/nix/store/gzwghi7spl93iv4y6qb680cg8wyfbq6r-qwen38-0.1.0`,
  built device-free, install check passed.
- Promotion gate (`re-earn-promotion-gate-2026-10-26.json`): promoted
  `passthru.executionIdentity` equals the measured identity `qwnxwc…`; the
  promoted v3 profile differs from the measured twin only in
  `physically_admitted` and `identities.receipt_blake3`, and names the fresh
  digest; `passed: true`.

The deploy-side checker `scripts/qwen38-admission-containment.py`, run against
the promoted output with its wrapper and this receipt tree, reports
`can_serve: true`, `mode: v3-admitted`, `receipt_blake3 cf9fc9d4…` and
`passed: true`. The independent receipt validation is retained as
`re-earn-receipt-validation-2026-10-26.json`.

`re-earn-campaign-provenance-2026-10-26.json` binds the eight per-root
`provenance.json` records (server pid, observed argv, observed environment,
twin digest, profile path equality), each root's warmup/measured counts and
BLAKE3 manifests, the request hashes and the frozen launch environment. That
file is the campaign-level record; the raw responses, logs and readiness bodies
remain under `/home/brittonr/models/qwen38-b8-reearn-2026-10-26/`.

## Incidents, all retained

- A first attempt at 13:07 was orphaned when the host harness capped the
  launcher at 300 s. The orphan held both cards; it was stopped, the boards
  were reset with `tt-smi -r`, and the attempt is quarantined
  (`aborted-single-0-warmup-validator`).
- A validator bug counted eight single-route response files as one; the server
  output itself was valid (eight width-1 samples, 64 tokens each). The validator
  now takes the expected file count separately from the expected width.
- `batch-1` failed at startup with `TT_THROW: Failed to write file` because the
  root filesystem was full (41 MB free). Two stale TT-Metal kernel-binary cache
  trees keyed to 2026-10-21 builds (7.5 GB) were removed; the failed root is
  quarantined (`aborted-startup-batch-1`) and the campaign continued with a
  fresh root of the same index.
- Root numbering was normalised to the procedure's global scheme (single on
  even indices, batch on odd) before the retained campaign started; the earlier
  per-route numbering produced no measured data.

## Non-claims

- This admits candidate width 8 only, on this execution identity, with these
  exact launch variables. It does not admit B=16, arbitrary continuous
  batching, or batched production serving.
- The receipt does not record the GDN fusion route, the prefill mode or the
  sequence limit; only the campaign provenance does. A future admission that
  changes those variables is a new campaign, not a reuse of this receipt.
- The promoted package is a new store path with the same execution identity. The
  scheduler and prefix-cache receipts `qwen38-deploy` requires are keyed by
  package path and already named other, older outputs before this promotion, so
  this receipt-only promotion does not change whether deploy accepts the tree's
  own build; it remains a `--force` case until those admissions are re-earned.

  **Superseded 2026-10-26:** those two admissions were then re-earned on this
  promoted package. All three receipts now name `gzwghi7…`/`qwnxwc…`, and the
  deploy guard accepts the package without `--force`; see
  `all-admissions-promoted-2026-10-26.md`.
- No bound, corpus, prompt, run count, execution order or structural gate was
  edited to reach a verdict; the bounds are the ones the v3 policy carries.
