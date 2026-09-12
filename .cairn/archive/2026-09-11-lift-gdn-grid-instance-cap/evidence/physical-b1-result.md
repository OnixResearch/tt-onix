# Physical B=1 run: the chunked kernels execute and match the reference

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V8.

## Result

The chunked fused decode path executed on real P150x2 silicon and produced the
admitted reference token prefix. `PROBE_EXIT=0`.

```json
{
  "admission_eligible": false,
  "batch_width": 1,
  "cleanup_complete": true,
  "decode_step_count": 7,
  "decode_step_seconds": [0.0631, 0.0528, 0.0561, 0.0528, 0.0543, 0.0571, 0.0585],
  "diagnostic_only": true,
  "fusion_mode": "full",
  "generated_token_ids": [11751, 13, 198, 760, 6511, 314, 9564, 369],
  "measurement_mode": "diagnostic-gdn-sequence",
  "model_identity": "Qwen3.8-27B:1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  "physical_pcie_device_ids": [0, 1],
  "simulated": false,
  "text": " Paris.\nThe capital of Germany is"
}
```

## What this establishes

**The runtime argument layout agrees between the factories and the kernels.**
This was the single largest open risk in the change, and it was previously
verified only by inspection. The reader and writer moved their addresses from
arguments 1..6 to 2..7, and the compute kernel gained its first runtime
argument. A disagreement would have read the wrong address or the wrong scalar.

`fusion_mode` is `"full"`, so the chunked code ran. The gate was set explicitly;
without it the probe would have exercised `off` and proved nothing.

**The output matches the admitted reference.** The archived P150x2 screen
recorded control tokens `11751` and `13`, and the admitted decode-component
profile recorded the same two. This run produced
`11751, 13, 198, 760, ...` — the expected prefix, followed by the expected
continuation `" Paris.\nThe capital of Germany is"`.

**Real hardware.** `physical_pcie_device_ids: [0, 1]` and `simulated: false`.
Clean teardown: `cleanup_complete: true`, and the log shows the UMD cluster
destructor completing.

## What this does not establish

- **B=1 does not exercise chunking.** At B=1 the head count is 24, the chunk
  length is one, and the instance loop runs exactly once. The step times, around
  56 ms median, sit in the same band as the 56.33 ms baseline for that reason.
  This run tests the argument layout, which is what B=1 can test.
- **Chunking itself is untested on hardware.** That needs B=8, where the loop
  takes two instances per core.
- **The prefill scan kernels did not run.** `--sequence-steps` drives the decode
  sequence path. Task V9 needs its own run for the prefill site.
- **No admission.** The receipt is `admission_eligible: false` and
  `diagnostic_only: true` by design. No admission binding moved.
- **No throughput claim.** Eight tokens on one prompt is not a benchmark.

## Isolation and restoration

Devices were isolated with `ttwkv7-owner-control isolate`, which left
`qwen38-p150x2.service` inactive for the run. The service was restored
afterwards under the standing authority in `AGENTS.md`.

## Reproduction

The exact command is in `physical-b1-run-plan.md`. One correction to that plan:
the helper's subcommand is `isolate`, not `stop`.

A `HOME` and `MPLCONFIGDIR` in fresh temporary directories were used, matching
the service's own environment, because an earlier recorded attempt crashed
during TTNN import when `HOME` was absent.

## Log notes

`Failed to discover available ethernet links; falling back to 1 link` repeats
during startup. It also appears on the admitted path and is not related to this
change.

`Allocating device buffers is unsafe due to the existence of an active trace`
appears once. The probe does not execute that trace, and the receipt reports a
clean run.
