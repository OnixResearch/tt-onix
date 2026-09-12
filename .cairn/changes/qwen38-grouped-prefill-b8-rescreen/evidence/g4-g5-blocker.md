# g4/g5: the grouped path's bounded blocker, and the fail-closed route

Recorded 2026-09-12 for change `qwen38-grouped-prefill-b8-rescreen`.

## The candidate fails on its own startup warm-up

With `QWEN35_GROUPED_PREFILL=1` on the current variant package, the service
dies during startup warm-up before serving anything:

```
TT_FATAL: Input tensor shape Shape([4, 24, 128, 128]) does not match output
tensor shape Shape([16, 24, 128, 128])
RuntimeError: TT_FATAL @ ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_device_operation.cpp:108:
out_tensor.logical_shape() == input_tensor_a.logical_shape()
  ...
  batch_service.py:263 warm_up -> BatchDecodeAdapterError("startup warm-up failed")
```

The shapes are the whole finding. The active width is 4, so the planner derives
one group of four with the TP2-correct bound; the model's grouped assembly then
copies that group's GDN state block `[4, 24, 128, 128]` into the persistent
decode buffer, which is allocated with `batch_size = maximum_width = 16`
(`MAXIMUM_ACTIVE_BATCH_WIDTH`). The copy op requires equal shapes, so a group
narrower than the buffer cannot install.

**Conclusion:** `prefill_paged_grouped` supports exactly one shape — a single
group that fills the decode buffer. It cannot assemble a group narrower than the
buffer, and it cannot assemble several groups into one buffer either. The
change's premise was "reuse the grouped model path where its contract matches";
at TP2 with a 16-row buffer this contract does not match, so grouping is blocked
on a model-side partial-width state assembly rather than on bounds or numerics.

## The route now fails closed instead of crashing

Two preconditions were added to `plan_prefill_route`, both with named reasons:

| Condition | Reason |
| --- | --- |
| active width below the decode buffer's row capacity | `group-width-below-decode-capacity` |
| the plan needs more than one group | `multi-group-assembly-unsupported` |

With the served configuration (capacity 16, widths 1..8) the grouped route can
therefore never fire, which is the honest state: the candidate is refused, not
silently broken. A single group that fills the buffer is the only shape that
reaches the model, and that shape is asserted as a positive fixture.

Device-free result: `PASS grouped prefill eligibility`, `PASS grouped prefill
routing`, `qwen36 device-free check: PASS`.

Silicon check with the gate **on**, on the package that carries the
preconditions (`/nix/store/hc7rxal0xqx2i0pf1ci9d392l57l2094-qwen38-0.1.0`): the
service starts, reports ready, and serves a greedy request returning the admitted
prefix `11751, 13, 198, 760, …` — i.e. the fail-closed route does not crash.

```
ready=true
{"model": "Qwen3.8-27B", "ready": true, "startup_warmup": {"elapsed_seconds": 372.8…, "widths": [1, 4]}, "status": "ok"}
tokens: [11751, 13, 198, 760, 6511, 314, 9564, 369]
runtime_identity: /nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0
physical_devices: [0, 1] simulated: False
```

## What this closes and what it opens

- **g4 (state tests)** are not reachable on silicon for the grouped route,
  because no grouped launch survives warm-up. The device-free half is complete;
  the silicon half is blocked by the same defect and is recorded rather than
  skipped.
- **g5/g6 (B4 preservation, B8 rescreen)** cannot measure a grouped candidate
  that cannot launch. The exact blocker is this record, which the change's task
  text allows in place of simulator coverage.
- **Enabling work, if it is wanted:** size the persistent decode state to the
  served width (capacity 8 for a B=8-only deployment) so one group fills the
  buffer, *or* fix the model's assembly to write a group block into a row range.
  The first is a deployment decision, the second touches adopted model source.
  Neither is part of this change.

## A separate finding: identity churn vs the tracked admission

Measuring the baseline (gate off) on the identity created by the g3 wiring
(`fypi0bf4…`) reproduced the admission with every gate passing:

| Gate (unchanged) | Measured | Result |
| --- | ---: | --- |
| aggregate decode speedup >= 6.0 | 6.0642 | pass |
| per-user retention >= 0.75 | 0.7580 | pass |
| TTFT ratio <= 9.0 | 6.2212 | pass |
| tail-latency ratio <= 1.5 | 1.2844 | pass |
| correctness (parity, slices, reset, prefix) | all true | pass |

Batch TTFT was 13.988 s against 2.248 s single; batch aggregate 100.06 t/s.

That receipt was already superseded when the fail-closed preconditions moved the
identity again, so the refresh was repeated on the final identity and the
tracked artifacts now name it:

| Item | Value |
| --- | --- |
| receipt | `pkgs/qwen36/evidence/2026-09-12/qwen38-27b-batch-admission-b8-receipt.json` |
| receipt BLAKE3 | `d542c05a9c82571976212c5b77530a9bd8cb55adfa0dcfc08625355626eb18e3` |
| package identity | `dswrs8nvjcl4pc0vy79xrqwk6jly2yzi` (variant, runtime `6ccynlvv…`) |
| gates | speedup 6.0492, retention 0.7562, TTFT ratio 7.0230, tail ratio 1.2986, all correctness true |
| batch TTFT | 14.902 s against 2.122 s single; batch aggregate 100.16 t/s |

The B=8 binding's `receipt_blake3` and the install-phase assertion both follow
the new file, so the package can only build when the admitted binding and the
tracked receipt agree. The superseded `2026-09-11` receipt and the archived
`2026-08-26` rejection remain as history.

The practical consequence for the repository: **any service edit invalidates
the admission identity and requires a fresh matched run.** Keeping the identity
stable across this change's remaining work is why the grouped route stays gated
off, and why the refresh was deliberately the last measurement taken.
