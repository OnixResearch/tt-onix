# S3a fresh-identity screens and fresh B=4 admission (2026-08-28)

Owner decision (option 3, see `s3-decision.md`) removed the per-position
minimum PCC floor. The policy revision changed the execution identity, so
every physical gate re-ran under the rebuilt package.

## Identity

- Package: `/nix/store/hn61wf5gd61wf25y7q87bvvhr4arwj8i-qwen36-0.1.0`
- Execution identity: `/nix/store/yv113vpaa4s4n0isjj2bm4hgd6cqhrk9-qwen36-execution-identity-0.1.0`
- Runtime: packaged tt-metal 0.74.0, devices 0 and 1

## Distributional receipt: ADMITTED (sampled mode only)

Eight matched roots (`control,candidate,candidate,control` for the probe and
the component screen), 21m31s:

- admitted `true`, eligible modes `["sampled"]`, zero rejection reasons
- mean logit PCC `0.996948` (floor 0.995) PASS
- min logit PCC `0.802772` reported as diagnostic, not gated
- mean top-K TV `0.018480`, max `0.227952` (caps 0.10 / 0.25) PASS
- residual mass `0.022562` PASS
- stable-position agreement `1.0` over 220 positions PASS
- duplicate stability `1.0` / `1.0` over 240 positions
- corpus BLAKE3 `8779347d...` identical to the first capture: the model
  reproduces the same logits across the identity rebuild

Receipt: `.pi/evidence/qwen38/gdn-sampled-dist-screen-20260827/dist-receipt.json`
(copied to `s3a-screen/`).

## Component speed gates: PASS

- GDN ratio `0.946456` (bound 0.95)
- full-step ratio `0.939473` (bound 0.97), speedup `1.0644x`
- duplicate spread `1.065685` (bound 1.25)

## Fresh B=4 admission: ADMITTED

Four roots `single,batch,batch,single` with `QWEN35_GDN_DECODE_FUSION=full`
on the new identity, evaluated by `scripts/qwen36-batch-receipt.rs` against
the immutable B=4 policy:

- exact per-request token parity B=1 versus B=4 **within the fused path**:
  PASS (all four correctness flags true)
- aggregate decode speedup `3.5293` (floor 3.0) PASS
- per-user retention `0.8823` (floor 0.75) PASS
- TTFT ratio `4.0062` (cap 4.5) PASS
- tail latency ratio `1.1306` (cap 1.5) PASS

Receipt: `.pi/evidence/qwen38/b4-admission-fused-20260828/receipt.json`
(copied to `s3a-screen/`).

Contamination note: the original single-2 root accidentally captured
responses from the deployed production service (identity `ixb9s8mb...`,
mode `physical-batched-service`) because a parallel session restarted the
production unit mid-window and the health check accepted the foreign
listener. The root re-ran clean; the receipt evaluator's identity binding
caught the contamination before any claim was made. Runners that share
port 8000 with production need a port-free guard before each root.

## What remains before admission completes (D1/D2)

The per-request sampled-only routing is not wired. The fusion mode is
selected at model construction from the environment, so one loaded model
cannot serve both paths per request, and the 27B weights do not fit twice
on P150x2. The open design options:

1. Two resident decode traces (one per mode) with a request-mode router —
   needs a trace-memory budget and a hygiene exception for two resident
   decode traces.
2. Per-forward mode selection in `gdn/tp.py` — a kernel-path change that
   needs its own hazard audit and a new candidate validation.
3. Serve the fused identity for sampled traffic only, keep greedy on the
   current deployed identity in a separate process during off-peak windows —
   operational, not architectural, but leaves greedy on the slow path.

The change stays active until one option is selected, implemented, and
validated. Nothing is deployed; production still runs the previously
admitted identity.

## Not claimed

- No deployment. No routing change. The fused candidate stays off in
  production until D1 completes.
- No B=8 or B=16 claim, no continuous batching claim, no accuracy benchmark.
