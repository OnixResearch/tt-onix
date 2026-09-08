# GDN precision candidate

## Goal and limits

Make the fused GDN state pass the existing NumPy tolerance without changing the equation or weakening the test. Numerical acceptance requires the same host and device binaries to pass a bounded hardware run.

The initial phase used source review, host tests, builds, and simulator checks without service changes. The later authorized hardware phase stopped the service and passed the bounded numerical test. No candidate deployment or firmware update occurred.

## Observed failure

After lock recovery, both tested TT-Metal runtimes executed the GDN kernel and failed the state comparison:

```text
rtol=0.001, atol=0.0001
Mismatched elements: 4890 / 32768 (14.9%)
Max absolute difference among violations: 0.00022145
```

See the [hardware record](evidence/tt-metal-0.77/hardware-attempt.md) for the exact package identities and logs. The matching summaries establish a baseline failure, not complete numerical equivalence.

## Source findings

The program factory forces `MathFidelity::HiFi2`. It does not consume the caller's compute configuration when it creates the kernel. The public wrapper also defaults to HiFi2, and the Qwen diagnostic caller explicitly requests HiFi2.

The [package-matched matrix-engine report](https://github.com/tenstorrent/tt-metal/blob/v0.74.0-rc6/tech_reports/matrix_engine/matrix_engine.md) describes the fidelity phases. HiFi2 omits the later SrcB mantissa phases. FP32 storage and FP32 destination accumulation therefore do not establish full multiplication precision.

The kernel and NumPy reference express the same real-number recurrence:

```text
decayed = exp(g) * state
delta = v - k @ decayed
next_state = decayed + beta * (transpose(k) @ delta)
output = q @ next_state
```

The NumPy reference applies beta before the outer product. This is equivalent over real numbers, but finite-precision ordering still needs consideration.

## Candidate changes

- Use HiFi4 in the wrapper default and program factory.
- Disable approximate math. Require FP32 destination accumulation, no L1 accumulation, and no full destination synchronization.
- Reject incompatible precision configurations rather than silently ignore them.
- Make the Qwen diagnostic caller request HiFi4.
- Keep all compute, reader, and writer source bodies unchanged.
- Keep the existing absolute and relative tolerances unchanged.

The host tests also reject a small corruption near the observed error scale. This check prevents a later tolerance increase from silently hiding that scale of error.

## Separate evidence scopes

`ttsim-gdn-reference-smoke` remains the full numerical gate. Its existing simulator limitation is not a pass.

The new `ttsim-gdn-input-validation` check uses `--validation-only`. It opens the simulated device, rejects invalid inputs, and verifies that the state remains unchanged. Its seven negative cases cover:

- Wrong input dtype.
- Wrong value dimension.
- HiFi2 precision.
- Approximate math.
- Disabled FP32 accumulation.
- L1 accumulation.
- Full destination synchronization.

This mode does not claim numerical correctness, recurrent-state carry, or cache reuse. The full smoke retains those checks and tests an explicit valid HiFi4 configuration on its final retry.

The Qwen device-free check inspects the fused call's AST. It accepts the HiFi4 caller and rejects mutated HiFi2 and missing-configuration fixtures.

## Investigation status

The source review used three related checks, not independent agent reviews:

| Candidate cause | Evidence | Remaining gap |
| --- | --- | --- |
| Recurrence or layout error | No equation discrepancy found in source review | No formal equivalence proof |
| Precision loss | HiFi2 is forced; documentation describes omitted mantissa phases | HiFi4 needs numerical execution evidence |
| Runtime migration regression | Both tested runtimes report the same state mismatch | This does not exclude other 0.77 regressions |

The confirmed defect is that the factory ignored precision settings. The later HiFi4 hardware pass supports the precision hypothesis for this workload. It does not prove accuracy for all shapes or models.

No full-model, serving-performance, or production admission follows from a host build or rejection-only simulator check.

## Validation results

The candidate build and these checks passed:

- Host oracle positive and negative tests, including the precision-regression fixture.
- Rejection of incompatible CLI modes.
- Ruff for `pkgs/ttsim/gdn-smoke.py`.
- TT-Metal host compilation.
- Qwen device-free checks, including the HiFi4 AST guard and negative mutations.
- Simulator bundle validation.
- Basic Blackhole simulated device open, operations, and close.
- All seven simulated GDN input-rejection cases.

Saved evidence:

- [GDN rejection-only receipt](evidence/tt-metal-0.77/gdn-hifi4-validation.json).
- [Basic simulator receipt](evidence/tt-metal-0.77/gdn-hifi4-basic-smoke.json).

The candidate runtime is `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0`. The checked Qwen package is `/nix/store/qw9llks4k65np226xygfrmzx1qcbs6nz-qwen36-0.1.0`.

The full numerical simulator gate still fails:

```text
ERROR: UnsupportedFunctionality: tensix_matmul_op: ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled
```

Local logs are `gdn-hifi4-checks.log`, `gdn-hifi4-final-checks.log`, and `gdn-hifi4-passing-checks.log` under `~/.cache/tt-upgrade-review/evidence/`. The final-checks log includes the failed numerical gate. The passing-checks log excludes that gate rather than reclassifying its result.

The deployed service still returned `ready: true` after the offline checks. The user then granted standing Tenstorrent maintenance authority, including service stops, recovery, and firmware updates. Restoration is not required after maintenance.

## Hardware result

At 01:50 EDT on 2026-09-08, `ttwkv7-owner-control isolate` stopped the service without an interactive sudo prompt. The same candidate runtime and smoke script then ran on physical Blackhole hardware. The test requested device 0, used slow dispatch, and disabled SFPLOADMACRO. Simulator environment overrides were absent.

The [hardware observations](evidence/tt-metal-0.77/gdn-hifi4-hardware.json) record a pass:

- Two heads with head dimension 128.
- Three recurrent steps.
- Fresh and aliased state storage.
- Program-cache reuse.
- Seven invalid-input and configuration rejections.
- A successful valid retry after rejection.

The largest recorded recurrent-state absolute error was `7.927790284156799e-05`. The original tolerances stayed at `atol=0.0001` and `rtol=0.001`.

The runtime was `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0`. UMD reported firmware 19.11.0 and KMD 2.10.0. Device-driver close completed normally. The local log is `~/.cache/tt-upgrade-review/evidence/gdn-hifi4-hardware.log`.

This is a physical hardware result, not a simulated result. It establishes the bounded GDN smoke only. Full-model accuracy, multi-chip inference, performance, and production deployment remain outside this result.

The service remains stopped under the user's authorization. No firmware update was necessary. The full numerical simulator gate still fails at its unsupported matmul flag.
