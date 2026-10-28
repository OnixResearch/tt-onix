# Device-free scheduler core and shell (tasks s1–s4, 2026-09-27)

Device-free evidence only. Nothing here was executed on a Tenstorrent device or
simulator, and it makes no latency, throughput, or admission claim. Tasks s5
(simulator/hardware refill screen) and s6 (queue-inclusive comparison) remain open.

## Baseline (s1, rerun in s4)

Before any change, `qwen36-check` from qwen38
`/nix/store/6j5h0gvpr5k97yjfb9nak7ba3vifdg6k-qwen38-0.1.0` (tt-onix `3696f21`)
passed with 54 probes. The same suite, rebuilt with this change, is recorded below.

## What was added

- `pkgs/qwen36/continuous-slot-policy.ncl` and `continuous-slot-contracts.ncl`: the typed,
  closed policy. B4 only (`slot_capacity = 4`), queue capacity 32, decode window 16 steps,
  one refill per boundary, 8,192-token refill budget, two head bypasses, 1,800 s queue
  bound, 900 s device-effect stall bound. The build exports
  `share/qwen36/continuous-slot-policy.json`. The negative fixture
  `fixtures/invalid-continuous-slot-policy.ncl` has more refills per boundary than slots
  and must fail its export.
- `qwen36_service/continuous_slot_core.py` (s1): the pure core. Slots are `free`,
  `prefilling`, `active`, `complete`, or `quarantined`, each admission under a new
  generation. Inputs are explicit time, request bounds, resolved route, the KV block range
  (from the decode profile's `kv_pool_blocks`), and completion observations. Outputs are
  a new schedule plus typed effects (`RejectRequest`, `StartGroup`, `PrefillSlot`,
  `RetireSlot`, `DecodeWindow`).
- `BatchedTracedDecodeAdapter` slot contract (s3): `begin_slots`, `refill_slot`,
  `retire_slot`, `abandon_slots`, `decode_slots_greedy`, `decode_slots_sampled`, and
  `TraceBindingDriftError`. It follows the hardware refill result in
  `refill-coexistence.md`: the prefill uses `prefill_paged_slots` beside the resident
  trace, and every decode input, including the page table, is restaged before the next
  replay.
- `qwen36_service/continuous_slot_service.py` (s3): `ContinuousSlotCoordinator`, one
  worker thread for every device effect, including structured reads. A read waits for the
  group to drain.
- `server.py`: `--continuous-slot-policy PATH` selects the new coordinator at
  construction. It requires candidate decode mode and conflicts with the admission
  measurements. Without the flag the group coordinator serves, as before. A quarantined
  engine answers `/healthz` with 503. A streaming client disconnect cancels its request.

## Inactive-row and allocation contract (s3)

- A row that holds no request decodes token 0 at position 0 upward. Its page-table row is
  all scratch block, so its KV writes cannot reach a live or future run.
- A retired row is pointed at scratch before its KV run can be reassigned.
- A refill that compiles a program, or drops the masked prefill traces, releases the
  decode trace instead of replaying it. The group then admits nothing more and drains on
  a recaptured trace.
- Trace-binding drift, a failed reset, or one device effect longer than the stall bound
  quarantines the engine.

[INFERENCE] Idle rows decoding into the scratch block, and refilling a slot that held an
earlier user, follow from `prefill_paged_slots` overwriting the slot's GDN row and from
the scratch-block page table. The hardware probe covered only a never-used slot with its
own KV run. s5 must exercise both on the device.

## Probes (s2, s4)

Three probes were added to `pkgs/qwen36/check.py`:

- `continuous slot policy` (s2, pure core). Covers:
  - B4 initial admission in arrival order;
  - staggered unequal-length refill that leaves the other slot bit-identical;
  - stale-generation and out-of-order events rejected without change;
  - first-token and mid-window stops;
  - queued and running cancellation;
  - request and queue deadlines, and active timeouts;
  - queue-full, duplicate, oversized, and KV-exhausted requests;
  - the head-bypass bound across routes;
  - refill count and token budgets;
  - draining for waiting reads;
  - quarantine and release.
- `continuous slot worker` (s4, shell over a fake row-independent adapter). Covers:
  - every device call from one thread with no overlap;
  - seven staggered mixed-length requests equal to each request decoded alone, one group,
    refills beside a live decode, and streamed tokens in order;
  - the inactive-row input contract, checked on every window;
  - the sampled route;
  - device failure: the group fails and is reset before the next group;
  - an unsafe refill drains the group;
  - queued cancellation;
  - drift quarantine with no later device call;
  - stall quarantine from the waiting caller with no later device call;
  - refusal of fixed-group admission measurements.
- `continuous slot adapter` (s4, the real adapter on fake TTNN). Covers:
  - no refill before `begin_slots`;
  - route selection and reset;
  - the slot row only;
  - owned-slot, overlap, reserved-prefix, scratch, short-run, and out-of-group
    rejections with no state change;
  - capture once, then no restage while nothing changes;
  - full four-input restage after a refill and a retirement;
  - drift before and during a refill;
  - compile-triggered and stale-mask trace release with recapture;
  - a structured read closing the group.

Mutation checks were run by hand: each targeted invariant was removed in turn, and every
mutation failed its probe. For the pure core the mutations were the bypass bound, the
bypass count, the generation check, the budget, and the active timeout. For the shell:
the inactive position, draining, stall detection, reset after failure, row positions,
drift handling, and unknown-cancel reporting. For the adapter: ownership, overlap, full
restage, compile release, retirement scratch, drift type, and read closure. After the
harness races were removed, the worker probe passed 10 of 10 consecutive runs.
