# Qwen3.8 staged decode comparison

## Result

The `norm-state` candidate reduced median B=1 decode-step time from 56.33 ms to 55.34 ms. The reduction was 1.76%. All four runs produced the same 64 tokens and text for `The capital of France is`.

The candidate/control time ratio was 0.9824004. This result does not meet the existing 0.97 full-step target in `pkgs/qwen36/gdn-decode-fusion-profile.ncl`. The candidate remains disabled by default. No serving route, numerical tolerance, or admission limit changed.

This diagnostic does not evaluate the complete fusion profile. That profile also requires component measurements, state checks, and five samples per root. This round supplies two full-sequence runs per mode, not those additional gates.

## Method

The comparison used the same Qwen package and TT-Metal runtime for all four physical runs:

```text
Qwen package: /nix/store/27zmwsbl7j3kl6r30rbqx7cv945grlmr-qwen38-0.1.0
TT-Metal: /nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0
Model: Qwen3.8-27B:1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
Physical PCIe device IDs: 0, 1
Order: off, norm-state, norm-state, off
```

Each run generated 64 tokens on the physical P150x2 mesh. The first token came from prefill. The diagnostic measured the remaining 63 decode steps after prefill and trace preparation. Each timing window included input transfer, trace replay, mesh synchronization, and winner readback.

Each process used one host thread per numerical library, private scratch space, a 96 GiB memory limit, and a 900-second timeout. The model weights and tensor caches were read-only. The BF16 decode override was absent. `TT_METAL_DISABLE_SFPLOADMACRO=1` remained set.

The [raw receipts and comparison](../pkgs/qwen36/evidence/2026-09-08/staged-abba/comparison.json) contain these measurements:

| Run | Mode | Median step | Mean step | Total for 63 steps |
| --- | --- | ---: | ---: | ---: |
| A1 | `off` | 56.341 ms | 56.839 ms | 3.580848 s |
| B1 | `norm-state` | 55.376 ms | 55.900 ms | 3.521674 s |
| B2 | `norm-state` | 55.237 ms | 55.632 ms | 3.504817 s |
| A2 | `off` | 56.326 ms | 56.518 ms | 3.560604 s |

The aggregate median pools 126 measured steps per mode. The comparison checks the run order, identities, device IDs, cleanup, complete sequences, exact token/text equality, and positive durations. Negative controls reject a token mutation, a zero duration, simulated evidence, a missing run, and wrong device IDs.

The first queue wrapper stopped during an earlier `off` run. Its isolated child completed the sequence and closed the devices. That receipt remains outside this balanced comparison. A named user unit then ran all four comparison processes and exited with status zero. Its [process status](../pkgs/qwen36/evidence/2026-09-08/staged-abba/process-status.txt) records completion.

## Startup reader correction

The independent source review identified a different problem in full-fusion startup. The `full` mode captures 20 candidates per device, but the old warm-up used the greedy reader. That reader expected one winner per device.

`pkgs/qwen36/warmup-tail-check.py` reproduced the exact 40-element reshape error without hardware. The corrected warm-up selects the reader for the captured mode. Positive cases cover greedy and sampled tails at widths one and four. Negative cases cover the wrong reader, malformed output widths, and read errors. Failed warm-up resets the adapter and does not admit requests.

The package checks pass with this correction. This round does not establish physical B=4 full-fusion correctness. The [earlier precision limits](qwen-gdn-staged-evaluation.md) remain separate from the reader defect.

## Checks and remaining limits

The following checks passed:

- Package build and device-free checks, including the new timing and reader controls.
- Canonical `qwen38` commands and legacy `qwen36` aliases.
- NixOS service selection, argument rejection, authority rejection, and companion wrapper checks.
- Nix formatting and Cairn validation.
- Ruff for the changed service modules and the two focused test files.
- A fresh basic Blackhole simulator smoke before hardware execution.
- Four physical B=1 sequences with matching tokens/text and successful cleanup.

The artifact-quality gate failed before and after the rename with the same message:

```text
artifact manifest entry count drift: expected 225, found 424
```

The seven new artifacts passed the scoped repository artifact check. Their generated entries now appear in the manifest under the permitted `pkgs` root. Unrelated manifest entries remain unchanged. The final whole-tree check reports 232 manifest entries and 431 observed artifacts. The pre-existing 199-entry gap remains.

A full manifest regeneration still failed the existing path policy. `quality/policy.ncl` permits `allowed_roots = ["cairn", "pkgs"]` for portable evidence, but the archive uses `.cairn`. The first rejection was `.cairn/archive/2026-08-11-add-rwkv7-persistent-decode-canary/evidence/acceptance.md`. This task did not expand the evidence policy or change that archive.

The full numerical simulator gate retains its unsupported matmul-control limitation. A basic simulator smoke does not replace that gate. Repository-wide acceptance is not claimed.

This result does not establish exact recurrent-state equality, other prompts, longer sequences, B=4 performance, full-fusion accuracy, or production admission. The task did not deploy a candidate or push changes. The service has no running process and retains its failed systemd state after the authorized stop.

## Search outcome

Two read-only workers reviewed the package boundary and the full-mode startup error. Each had an eight-minute, 20-call budget. The parent validated their findings with package checks and the reproducing fixture.

The hardware budget allowed four comparison runs and one retry. The interrupted wrapper consumed the retry. All four later runs completed. The round ends with a validated rename, a device-free startup fix, and a staged candidate that misses the timing target. The next distinct optimization work is GDN or MLP attribution at B=4, not another queued-readback trial.
