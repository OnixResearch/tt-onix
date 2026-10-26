# B=8 interleaved re-earn procedure, 2026-10-22

Read-only reconstruction and a predeclared runbook for a later, separately authorized campaign. **No campaign was run for this evidence.** The historical campaign had **eight fresh server roots and 192 measured responses**, not 192 server starts: eight prompts × three measured rounds × eight roots. Warmup responses are additional and excluded from the receipt.

## Sources and what can actually be reconstructed

- `../.. /optimize-gdn-decode-kernel-grid-and-prep/evidence/protocol-v3.md` (path without the space: `../../optimize-gdn-decode-kernel-grid-and-prep/evidence/protocol-v3.md`), lines 15–27 and 54–60: alternating routes, four roots per route, per-user TTFT, one warmup and three measured rounds. `protocol-v3-validation.md:11–24,80–85` records the eight roots, 192 responses, claimed package and a session gap between pairs.
- `pkgs/qwen36/batch-admission-b8-policy-v3.ncl:8–62`: the exact eight prompts, B=1/B=8, 64 generated tokens, rounds, order, bounds and structural requirements.
- `scripts/qwen38-batch-group.py:30–44,72–139`: client interface, preconnected sockets, a thread barrier, `/v1/completions`, saved `response-N.json`, and default retries. `pkgs/qwen36/service/qwen36_service/batch_service.py:47,379` sets the collection window to 0.01 seconds.
- `scripts/qwen38-step-probe.sh:27–59,75–95` supplies an identifiable **screen** launch recipe, not the saved campaign launcher: package binary `qwen38-p150x2-b8-admission`, localhost port 8000, model alias `Qwen3.8-27B`, sequence limit 2048 and generation limit 64. It selects the `full` GDN fusion route. Its own header explicitly says it does not produce admission evidence.
- `scripts/qwen36-batch-receipt.rs:21–42,245–285,288–417,438–670`: receipt command, input naming, identity/profile checks, state slices, parity and bounds.
- Receipt `pkgs/qwen36/evidence/2026-10-22/qwen38-batch-admission-b8-v3-receipt.json:1–89`: execution, identity, correctness, metrics and non-claims; response-hash contents were not used to infer launch provenance.

**Unproven provenance:** the named evidence does not preserve the historical root-launch command, resolved profile path, process argv, exact port, or complete environment. Source proves an override was possible; no stored record here proves whether it happened. Port 8000 and the launch flags below are a source-grounded, explicitly predeclared re-earn recipe, **not a recovered verbatim historical invocation**. No timestamp inference is used. `scripts/qwen38-prefill-cache-screen.py` is absent in this checkout, and the identified receipt/group/probe path does not require it.

A device-free JSON/wrapper inspection during this investigation produced:

```text
receipt package_identity: /nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0
receipt profile_id: qwen38-p150x2-batched-throughput-b8-v3
measured response count: 192
package: /nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0
wrapper identity: /nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0
generic profile_id: qwen38-p150x2-batched-throughput-b8-v2
package: /nix/store/frf73q2612gczcj1qr43zz48i3frj2q4-qwen38-0.1.0
wrapper identity: /nix/store/4115hgb2ri3168c9d0g4r4pjj8m8m06n-qwen36-execution-identity-0.1.0
generic profile_id: qwen38-p150x2-batched-throughput-b8-v2
```

This contradicts the validation prose's package/identity association; it does not identify the historical executable or profile file actually loaded. It must not become a conclusion that an override did occur.

## Exact identity and profile plumbing

### Installed paths and check-phase coverage

At the inspected `pkgs/qwen36/default.nix` revision:

| Role | Installed path under package `$PKG` | Source |
| --- | --- | --- |
| Explicit v3 export to measure | `$PKG/share/qwen36/batch-admission-b8-profile-v3.json` | staging/export at lines 393–400 |
| Generic B=8 export, v2 in the inspected packages | `$PKG/share/qwen36/batch-admission-b8-profile.json` | staging/export at lines 384–392 |
| B=8 admission wrapper default | the **generic** path above | lines 653–676, specifically 668–669 |

The phase is `installCheckPhase` at lines 1152–1154. It checks generic B=8 file existence at line 1234; lines 1381–1393 export it twice, compare determinism, and compare against the installed generic JSON; lines 1394–1408 check bounds and the historical generic receipt binding. Those are **not v3-export checks**. The `profile-v3` occurrences in this file identify its source/input/copy/export, not an analogous v3 install-check. A passing generic check must not substitute for the v3 gates below. These line numbers describe the inspected source, not a claim that an in-flight bootstrap edit has landed.

### `package_identity`

1. The Nix wrapper appends `--package-identity ${executionIdentity}` and `--runtime-identity ${tt-metal}` (`default.nix:673–676`). It exports `PYTHONPATH`, TT-Metal paths, mesh, offline mode and temperature, but **no identity environment variable supplies this datum** (`default.nix:653–661`).
2. `server.py:1338–1339` declares required CLI arguments. `server.py:1556–1570` passes `args.package_identity` and `args.runtime_identity` to `QwenEngine`; lines 232–234 retain them. Lines 307–314 pass those identities and the admission profile to the batch service.
3. The response writes `self.package_identity` to `x_onix_decode_evidence.package_identity` (`batch_service.py:782–808`, specifically 799). The receipt copies the canonical response identity into `identity.package_identity` (`qwen36-batch-receipt.rs:606–620,359–370`). It is an **execution-identity store path**, not necessarily the package output path.

The stored wrapper `/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0/bin/qwen38-p150x2-b8-admission:10` embeds `l4arf461…` and ends in `"$@"`. The corresponding `frf73q…` wrapper's line 10 embeds `4115hgb…`, not `l4arf461…`.

### `profile_id`

1. The wrapper supplies `--batch-admission-profile` with its generic package JSON; user arguments follow it. `server.py:1327–1328` uses ordinary argparse options, so a later occurrence of the profile option replaces the earlier value. This leaves `--package-identity` unchanged unless separately overridden.
2. `server.py:1367–1393` only requires an existing absolute file and parses it through `BatchAdmissionProfile.from_mapping`; there is **no package-containment check there**. The profile parser takes `profile_id` from that JSON (`batch_admission_core.py:164–166`). `server.py:1569–1570` passes the parsed profile and explicit measurement mode onward.
3. In admission mode `batch_service.py:725–741` sets the response profile ID from `self.batch_admission_profile.profile_id`; lines 797–808 emit measurement mode, package identity and that profile ID together.
4. The receipt requires every response profile ID to match the profile supplied to its own `--profile` option (`qwen36-batch-receipt.rs:558–565`). It enforces consistent nonempty response identities, but does not resolve the service profile's path or prove that the package exported it.

Thus source permits a process to report v3 from an explicit JSON while retaining a wrapper identity whose generic export is v2. **Possibility is not historical proof.** The re-earn must explicitly select the package's **own v3 export**, and must fail closed on any resolved profile path outside that package's canonical store directory.

## Required device-free bootstrap gate before any root

The final bootstrap contract is that execution identity depends on immutable service source, authority, policies, contracts, profile sources, patches, runtime and model source—not mutable receipt bindings, including the v3 binding and its siblings. Admission is carried by the exported profile JSON.

Before reserving devices, obtain two device-free package outputs from the bootstrap fix, differing only in the controlled receipt-binding state (retain the two source manifests and the exact fixture receipt). The unmeasured output is the package to measure; the admitted fixture output tests promotion and is **not** a substitute for earning a fresh receipt. Require and record:

```text
identity(binding=not-measured) == identity(binding=admitted)
profile_json(not-measured) != profile_json(admitted)
not-measured: physically_admitted=false, identities.receipt_blake3="not-measured"
admitted:    physically_admitted=true, identities.receipt_blake3=<named fixture receipt BLAKE3>
```

Except for `physically_admitted` and `identities.receipt_blake3`, the JSON must be equal. Immutable source/input manifests must be equal; binding deltas must be the only allowed source differences. Repeat the identity check for the actual promotion after the fresh receipt exists. This is a **required post-bootstrap gate, not a result established here**. In the inspected pre-fix source, `default.nix:125` still includes the v3 mutable binding in identity inputs; do not start roots merely because this runbook exists.

Save `bootstrap-gate.json` with `schema`, `unmeasured_package_store_path`, `admitted_fixture_package_store_path`, `unmeasured_package_identity`, `admitted_fixture_package_identity`, `identity_equal`, `profile_json_changed`, `profile_only_admission_fields_changed`, `fixture_receipt_blake3`, `passed`; also save both profiles, both execution-input manifests, fixture receipt, source manifests, checker command/source and stdout/stderr. All booleans must be true. A missing artifact or inconclusive comparison fails the gate. Building these controlled outputs is device-free preparation; do not invent or reuse a purported passed result.

## Predeclared command sequence for the later re-earn

Commands below are a recipe to execute later, **not commands run for this evidence**. Pin the repository revision, package output, model snapshot, runtime and all tool binaries before starting; preserve their paths/hashes. Use a new empty host artifact directory, never an old response directory. Use `GIT_PAGER=cat GIT_TERMINAL_PROMPT=0` for every git invocation and `timeout 30` for each finite command. Long-lived hardware servers are launched by a bounded launcher and supervised separately, not killed automatically after 30 seconds. Shell loops orchestrate individually bounded calls. A timeout is an abort, not permission to silently rerun a measured sample.

### 1. Device-free preflight and corpus

Set `PKG` to the **unmeasured, post-bootstrap** package store path, `MODEL` to the checkpoint snapshot named by the policy, and `RUN` to a new empty absolute artifact directory. Also set `PROMOTION_PROBE_PKG` and `FIXTURE_RECEIPT` to the controlled admitted-fixture output and receipt used for the bootstrap gate. No historical package above is pre-authorized by this document. Execute and save the bootstrap comparison first; no hardware action is allowed without its passed record.

```sh
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0
# PKG, MODEL, RUN, PROMOTION_PROBE_PKG and FIXTURE_RECEIPT are pinned inputs.
timeout 30 mkdir "$RUN"
timeout 30 git --no-pager rev-parse HEAD > "$RUN/source-revision.txt"
timeout 30 git --no-pager status --porcelain > "$RUN/source-status.txt"
timeout 30 nickel export --format json pkgs/qwen36/batch-admission-b8-policy-v3.ncl > "$RUN/policy.json"
export PKG MODEL RUN
export PROFILE="$PKG/share/qwen36/batch-admission-b8-profile-v3.json"
```

Run a device-free preflight checker before launch and before receipt construction. Its mandatory algorithm is:

1. Resolve `PKG` and `PROFILE` with strict realpath. Require the package to be a single `/nix/store/<hash>-<name>` directory and require `PROFILE` to be a regular file **inside that resolved directory**, equal to the resolved installed v3-export path above. Reject symlink escapes, missing files, external profile paths and generic/v2 selection.
2. Parse the wrapper's `--package-identity` and `--runtime-identity`, saving the wrapper bytes and resolved executable. Check the profile's `profile_id`, complete workload, admission bounds and immutable identity fields against `policy.json`. Require false/`not-measured`, as the receipt builder itself does (`qwen36-batch-receipt.rs:438–465`). Preserve a byte copy of the profile for audit, but pass the actual package path to the process and builder, **not the copy**.
3. Validate `bootstrap-gate.json`, its artifact hashes and source-manifest comparisons. Require this gate's unmeasured package path/identity to equal the package being launched. Require policy checkpoint/tokenizer/model revisions, model snapshot, two-card runtime and selected `step` route to match the frozen inputs.
4. Generate exactly eight request files `requests/request-00.json` through `request-07.json` in policy order, each containing `{"model":"Qwen3.8-27B","prompt":<policy prompt>,"max_tokens":64,"temperature":0,"stream":false}`. Preserve the bytes/hashes. The second prompt is the literal nonempty fixture sentence in the policy; do not replace it with an empty prompt.
5. Save `preflight.json` with the validated paths/identities, all input hashes, complete frozen launch environment, order and explicit pass/fail results. A driver/checker implementing these requirements must be retained and reviewed before device use; **the existing receipt builder does not implement path containment or bootstrap validation**. Missing enforcement is an abort, not a manual waiver.

### 2. Hardware: reserve both cards and start one fresh root

**Hardware: requires physical devices 0 and 1 and an approved exclusive window.** Use the owner's established isolate/restore mechanism (`qwen38-step-probe.sh:72–75` uses `ttwkv7-owner-control`). Record successful ownership before starting. Stop/restore on every exit. Do not run roots concurrently or mix the screen script's outputs into this receipt.

```sh
timeout 30 ttwkv7-owner-control isolate
```

Use this exact order, finishing/stopping each process before the next:

```text
root 0 single, root 1 batch, root 2 single, root 3 batch,
root 4 single, root 5 batch, root 6 single, root 7 batch
```

For each root set `INDEX` and `ROUTE` from that table and set `ROOT="$RUN/$ROUTE-$INDEX"`. Create `ROOT`, `ROOT/responses`, `ROOT/raw`, and `ROOT/mplcache`. A bounded launcher starts the following argv with a fresh PID/session, redirects stdout/stderr to `ROOT/serve.log`, and saves its PID:

```sh
# HARDWARE server argv; launch asynchronously, then retain and supervise its PID.
"$PKG/bin/qwen38-p150x2-b8-admission" \
  --host 127.0.0.1 --port 8000 \
  --model-path "$MODEL" --model-alias Qwen3.8-27B \
  --max-sequence-length 2048 --max-generation-tokens 64 \
  --batch-admission-profile "$PROFILE"
```

The launcher must use a clean, explicitly recorded environment, not inherit experimental overrides: retain the pinned host `PATH`; set `HOME="$MODEL"`, `TT_CACHE_PATH="$MODEL"`, `MPLCONFIGDIR="$ROOT/mplcache"`, `QWEN35_GDN_DECODE_FUSION=full`, `TT_METAL_DISABLE_SFPLOADMACRO=1`. The package wrapper supplies its own Python/runtime/mesh/offline/temperature variables. Do not inject a custom `PYTHONPATH`, identity flags, simulator, dtype/MLP-BF4 option, or external profile. Record the resulting process environment as well as the launch environment. These are predeclared re-earn settings, not claims about the lost historical environment.

Poll `timeout 30 curl --fail-with-body --silent --show-error --max-time 2 http://127.0.0.1:8000/healthz` at ten-second intervals, at most 60 polls; preserve each body and status. Abort if the PID dies, ownership changes, an existing server owns port 8000, or readiness is not reached. Before the first request, save `/proc/<pid>/cmdline` and `/proc/<pid>/environ` byte-for-byte, validate the **last effective** profile argument and identities against preflight, and write `ROOT/provenance.json` as specified below. Recheck profile bytes/realpath and process identity at the end of the root.

### 3. Hardware: warmup, then three rounds, without silent retries

Each root gets **one complete eight-prompt warmup round**, followed by measured rounds 0, 1, 2. Warmup files must never contain `-measured-` in the flat receipt-input directory. For a single root, send each prompt sequentially; no subsequent request starts until the previous response completes. For a batch root, release **eight requests**, not eight servers, together through the 10 ms collection window.

Single-route command, for each prompt index `P=0..7` in each round (`REQUEST` is its zero-padded request filename; `OUT` is the distinct raw output filename):

```sh
# HARDWARE request; retain exit status, HTTP status and stderr as separate artifacts.
timeout 30 curl --fail-with-body --silent --show-error --max-time 29 \
  -H 'Content-Type: application/json' --data-binary "@$REQUEST" \
  http://127.0.0.1:8000/v1/completions > "$OUT"
```

Batch-route command, once per warmup/measured round (`ROUND_DIR` is a new raw directory, never reused):

```sh
# HARDWARE group: the Python driver opens all eight connections before the barrier.
timeout 30 python3 scripts/qwen38-batch-group.py \
  --url http://127.0.0.1:8000 --timeout-seconds 29 --attempts 1 \
  --request "$RUN/requests/request-00.json" \
  --request "$RUN/requests/request-01.json" \
  --request "$RUN/requests/request-02.json" \
  --request "$RUN/requests/request-03.json" \
  --request "$RUN/requests/request-04.json" \
  --request "$RUN/requests/request-05.json" \
  --request "$RUN/requests/request-06.json" \
  --request "$RUN/requests/request-07.json" \
  --output-directory "$ROUND_DIR"
```

The default client retries up to four times and unlinks prior output (`qwen38-batch-group.py:43,85–86,129–138`); **override it with `--attempts 1`** so failure cannot erase campaign history. Its `choices` check is not admission validation. Independently require HTTP success, valid complete JSON, exactly 64 generated token IDs/completion tokens, intended width (1 or 8), `batched-traced`, `serving_route=step`, eligible physical admission mode, devices `[0,1]`, expected package/runtime/model/profile, clean reset, correct slices, and valid finite timings. Abort on any partial group, fallback, adapter error or rejected request, including warmup. No retry-to-pass; retain failures and restart an entirely new campaign only by a new explicit decision.

Keep raw warmup and measured files. Copy each successful measured response byte-for-byte into `ROOT/responses/response-P-measured-R.json`, where P is prompt 0–7 and R is measured round 0–2. This produces 24 files per root. The builder sorts filenames and indexes prompt-major/run-minor (`qwen36-batch-receipt.rs:469–499,633–640`); do not name files round-major or point it at nested raw output directories.

After the three rounds, record root completion, process health and log checks. TERM the saved PID, wait up to 30 seconds for exit and port release; forced termination or unexpected exit invalidates the campaign. Do not accidentally kill a process identified only by port. Start the next root fresh. Preserve root/pair adjacency and any gaps in the event log; do not resume from a failed or partial root and call it the same campaign. Eight roots yield **192 measured plus 64 warmup responses**.

### 4. Device-free: build and independently gate the receipt

Stop the final server and restore ownership before analysis:

```sh
timeout 30 ttwkv7-owner-control restore
```

Repeat the same containment/hash preflight against the actual package profile; it must still be unmeasured. Then, from the frozen repository with a provisioned Rust/Nix toolchain, run:

```sh
timeout 30 scripts/qwen36-batch-receipt.rs \
  --profile "$PROFILE" \
  --execution-order single,batch,single,batch,single,batch,single,batch \
  --single-response-root "$RUN/single-0/responses" \
  --single-response-root "$RUN/single-2/responses" \
  --single-response-root "$RUN/single-4/responses" \
  --single-response-root "$RUN/single-6/responses" \
  --batch-response-root "$RUN/batch-1/responses" \
  --batch-response-root "$RUN/batch-3/responses" \
  --batch-response-root "$RUN/batch-5/responses" \
  --batch-response-root "$RUN/batch-7/responses" \
  --output "$RUN/receipt.json"
timeout 30 b3sum "$RUN/receipt.json" > "$RUN/receipt.b3"
```

The executable shebang pins the Fenix toolchain and uses cargo script (`qwen36-batch-receipt.rs:1–11`); provision/cache these dependencies device-free before reserving cards. Receipt generation is device-free. Nonzero/timeout invalidates analysis; **exit zero alone does not imply admission**: the builder writes a rejection receipt with exit zero when bounds fail (`:232–242,337–356`). An independent retained checker must require `admission.admitted=true`, an empty reasons array, the expected canonical identities, the preflight/provenance records and all six gates below. Save rejection receipts unchanged, not just passing ones.

### 5. Device-free promotion, only after the receipt exists

The receipt hashes the exact saved response bytes, carries their execution identity, and records the selected profile/policy. `batch-admission-b8-binding-v3.ncl:5–6` is the mutable admission record: only after a passing fresh receipt exists and its digest is saved, set `physically_admitted=true` and `receipt_blake3` to that receipt's BLAKE3. `batch-admission-b8-profile-v3.ncl:3–8` merges those two fields into the immutable policy; the resulting installed v3 JSON carries admission.

Rebuild/export the promoted package device-free through the bootstrap-fixed packaging process. Require its wrapper execution identity to equal the measured identity, and require its profile to differ only in the two binding-derived fields and name the **fresh** digest. Archive `promotion-gate.json` plus both exports and manifests. A differing package output store path is allowed; a differing execution identity is not. If the gate fails, leave the result unpromoted and investigate; never edit response identities or rerun the receipt against invented identities. No binding edit or promotion occurs as part of this documentation task.

## Six receipt gates and additional abort conditions

| Gate | Frozen v3 requirement |
| --- | --- |
| Aggregate decode speedup | ≥ 6.0 |
| Per-user decode retention | ≥ 0.75 |
| Per-user TTFT ratio | ≤ 9.0 |
| Tail latency ratio | ≤ 1.5 |
| Token parity | exact across corresponding single/batch samples |
| Text parity | exact across corresponding single/batch samples |

The group-max TTFT ratio is diagnostic only, not a seventh performance gate. Structural gates additionally require physical devices `[0,1]`, no simulation, active prefix, clean reset, distinct state slices, no empty/invalid timings, one common package/runtime/model identity, exact v3 profile/width/route, the eight-prompt workload and 24 measured samples per root. The receipt computes ratios from medians of response metrics (`qwen36-batch-receipt.rs:321–355`), not from averaging four reported pair ratios.

Abort before roots on failed/missing bootstrap proof, unsupported/mismatched package, external or symlink-escaping profile, changed policy/corpus/model/runtime, nonempty output directory, missing reviewed containment/provenance checker, unavailable dependencies or device ownership. Abort the campaign on startup/health failure, timeout/nonzero command, process/identity/profile drift, wrong width/partial group, HTTP or adapter error, fallback/simulator, token count/parity/reset/state-slice failure, log contamination, overwritten/missing artifacts or an unplanned retry. A failed numerical bound produces a preserved rejected receipt, never relaxed bounds or a selectively remeasured root. A post-receipt promotion mismatch blocks promotion. Always stop owned processes and restore the two cards.

## Required provenance fields and saved artifacts

**New auditability requirement, not a claim about existing response fields:** retain a campaign-level `campaign-provenance.json` and a per-root `provenance.json` beside each response directory. Proposed exact fields:

```text
schema = "onix.qwen38.b8-reearn-provenance.v1"
package_store_path                 # strict resolved package output directory
batch_admission_profile_path        # strict resolved effective CLI argument
batch_admission_profile_blake3      # digest of bytes actually selected
package_identity                   # wrapper value, verified against responses
runtime_identity
model_identity
profile_id
source_revision
root_index                         # 0..7, in per-root records
route                              # single or batch, in per-root records
server_pid                         # actual owned process, in per-root records
server_argv                        # observed /proc argv, not just planned argv
server_environment                 # observed environment; retain securely
started_at
finished_at
bootstrap_gate_blake3
response_manifest_blake3
```

Record declared argv separately as `launch_argv`; check that the observed effective profile option resolves to the recorded `batch_admission_profile_path` inside `package_store_path`. Match each response's identities/profile to that record. Neither package identity nor profile ID alone is an adequate provenance record. The independent campaign checker must enforce this because the current builder does not. Bind these records to a final manifest alongside the receipt, including a digest of the receipt itself; do not modify the receipt's response bytes to insert provenance afterward.

Exact artifact classes to retain under the new run directory:

- `source-revision.txt`, `source-status.txt`, frozen driver/checker/group/receipt-tool sources and hashes, toolchain paths/versions, policy JSON, model/runtime/package paths and hashes, original wrapper bytes, canonical profile bytes/hash, immutable execution-input manifests and source-difference manifests.
- `bootstrap-gate.json`, its checker stdout/stderr, both fixture profile exports, both package identities/manifests and the named fixture receipt; `preflight.json` and all preflight logs.
- Eight canonical request JSON files plus hashes; `campaign-provenance.json`; a chronological `events.jsonl` recording command argv, start/end, route/index/round/prompt, exit/HTTP statuses, ownership transitions and gaps.
- For each of eight roots: `serve.log`, PID, raw process argv/environment, readiness bodies/statuses, `provenance.json`, all eight raw warmup responses and all 24 raw measured responses with client stdout/stderr/statuses; exactly 24 byte-identical flat `responses/response-P-measured-R.json` files. Record validation results and full content hashes.
- `receipt.json`, `receipt.b3`, builder stdout/stderr and exit status, independent six-gate/provenance validation report, and a final content-hash manifest for all retained artifacts. If compressing shipped responses, preserve original relative names and verify decompressed byte hashes. Do not retain only a summary table.
- After an approved promotion only: binding diff, promoted profile export, promoted package/wrapper/identity/manifests and `promotion-gate.json`. Failed/rejected campaigns keep all evidence with their verdict and never move the binding.

## Non-claims

No hardware, simulator, campaign, admission or promotion was run here. No timing, performance, request-latency or new retention claim is made. Historical reported ratios are not re-earned by this investigation. The original profile path and executable provenance remain unproven; override capability must not be turned into a historical conclusion. Bootstrap equality, path-containment enforcement and provenance capture are required future gates, not asserted existing results.

## Amendment 2026-10-22: the bootstrap gate is identity equality, not a second package build

The original text above asks for two package outputs differing only in the
binding and calls the unmeasured one "the package to measure". That formulation
is wrong and is kept only as history; it is not a waiver of anything below.

- An unmeasured package **cannot** build normally and **must not** be made to
  build: `installCheckPhase` requires the shipped v3 export to be admitted with
  the committed receipt, and the fixture tree fails closed. A bypassed
  `doInstallCheck = false` output is diagnostic only and may never be served or
  measured.
- The device-free bootstrap gate is **identity equality**: the admitted
  package's execution identity must equal the identity derived from the tree
  whose only difference is the binding text
  (`passthru.executionIdentity`, no package build), with byte-identical
  immutable input manifests, plus the admitted package's install check passing
  and the unmeasured build failing closed.
- **The campaign runs the admitted package**, not an unmeasured one. This is the
  same package whose export carries the admission, so there is no ambiguity
  about which profile the server loads.
- After the fresh receipt exists, the post-promotion rebuild is verified by the
  promotion gate: its execution identity must equal the measured identity it
  served, and its profile must differ only in `physically_admitted` and
  `identities.receipt_blake3`, naming the fresh digest.

Recorded in `bootstrap-gate.json` and `bootstrap-gate-2026-10-22.md`; no
hardware, campaign, receipt, admission or promotion was run.

## Amendment 2026-10-26: the measurement profile is the unmeasured twin

The builder guard ("input profile must be unadmitted and unmeasured") stays
exactly as written. The containment rule above is amended so the profile passed
to the server and the builder is an unmeasured twin of the packaged admitted
profile, with twin provenance as a hard precondition. All six gates and the
promotion gate are unchanged. See
[`re-earn-amendment-2026-10-26.md`](re-earn-amendment-2026-10-26.md).
