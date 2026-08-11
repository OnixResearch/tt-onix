# Fused WKV kernel — Phase 2 planning and reference cores

Task K1-K4 (planning and reference portion). The device kernel source (a
custom TT-NN op with dataflow + compute kernels compiled against the
packaged tt-metal 0.74.0 headers) is the remaining, larger body of work
that must pass the hazard audits, simulator smoke, and physical A/B
admission. This note records the device-free planning and reference cores
that are complete and tested.

## What is done (device-free, tested)

The `fused_wkv` module in the evidence crate now provides, beyond the Phase 1
equation reference and capability contract:

- `plan_fused_dataflow` (K1): bounded byte budget for the BF16 model vectors
  and the FP32 recurrent state, ex. B4 = 786432 state bytes, 36864 vector
  bytes, 6144 output bytes. Verified it scales with batch and matches the
  hand-computed shapes.
- `plan_fused_ownership` (K3): deterministic round-robin of (request, head)
  state tiles to fixed core groups, one writer per tile, no cross-request
  sharing, with a stable per-core completion count. Rejects zero cores and
  verifies uniqueness/completeness.
- `decide_fused_dispatch` (K4): fail-closed selection. The fused path is used
  only when the capability is supported AND the fused path is physically
  admitted. The `physically_admitted` flag is owned by the evidence pipeline
  and is never fabricated here.
- `fused_wkv_step` (K2 recurrence): decay, both rank-update products, outer
  product, state write, and BF16 readout in one deterministic step.
- `attention_correction` (K2 boundary): group-norm the readout rows, add the
  r_k bonus, scale by the gate, and bound to the BF16 finite range. The
  output projection stays outside per the design decision.

All 23 `fused_wkv` tests pass (positive and negative), clippy is clean, and
the proposal/design/tasks gates pass.

## What remains (device kernel source)

The actual TT-NN device operation that dispatches these kernels on silicon:

- a `fused_wkv` TT-NN op struct + program factory (ProgramDescriptor pattern)
  using the packaged `ttnn/device_operation.hpp` and
  `tt-metalium/experimental/metal2_host_api/program.hpp`;
- reader/writer dataflow kernels using the packaged `cb_api.h` and NOC APIs;
- a compute kernel using `api/compute/compute_kernel_api.h` (matmul,
  elementwise, and the same tile APIs the packaged `bmm_fused_activation.hpp`
  uses);
- semaphore/completion handshakes matching the ownership plan;
- the runtime shell that calls `decide_fused_dispatch` and keeps the generic
  TT-NN fallback.

That source must then pass: the full LLK race audit set (config, semaphore,
mailbox, dataflow, source-register, NoC), instruction-latency and performance
audits, the package-matched Blackhole simulator smoke, and the physical A/B
and strict admission gates. This is the bulk of the remaining change.

## Non-claims

The planning and reference cores do not prove the device kernel is correct,
fast, or safe. They fix the contract the kernel must satisfy. No simulator or
physical result is claimed here.