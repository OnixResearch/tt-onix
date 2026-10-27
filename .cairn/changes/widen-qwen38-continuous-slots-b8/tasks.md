# Tasks: Widen the continuous-slot scheduler to a B=8 candidate

## Phase 1: Attribution

- [x] [serial] A1 Confirm the three places that pin capacity 4: the core constant, the Nickel contract predicate, and the served default policy. r[tenstorrent.qwen38_p150x2.continuous_slot_policy]
- [x] [serial] A2 Confirm the coordinator and adapter are capacity-generic: `begin_slots(width=slot_capacity)`, the resident trace capture, the KV pool, and the inactive rows. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)

## Phase 2: Implementation

- [x] [serial] I1 Make the pure core accept `slot_capacity` in {4, 8} and refuse everything else. r[tenstorrent.qwen38_p150x2.continuous_slot_policy]
- [x] [serial] I2 Add the B=8 contracts and policy with the same bounds and `slot_capacity = 8`; keep the B4 files byte-identical. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)
- [x] [serial] I3 Export the B=8 policy, bind it into the package execution identity, and keep B4 as the served default. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)
- [x] [serial] I4 Update the coordinator's capacity wording and any width-4 assumption found in A2. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)

## Phase 3: Verification

- [x] [parallel] V1 Add positive device-free probes: the core accepts 4 and 8, the B=8 policy exports deterministically, and both policies coexist. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)
- [x] [parallel] V2 Add negative probes: capacities 1, 5, 6, and 16 are refused by the core and by the closed contract. r[tenstorrent.qwen38_p150x2.continuous_slot_policy]
- [x] [serial] V3 Run the package build, the device-free check, and the B4 profile export comparison (byte-identical). r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)
- [x] [serial] V4 Run a bounded physical B=8 scheduler smoke: open a width-8 group, refill beside active users, and verify exact candidate parity and a clean reset. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)
- [x] [serial] V5 Run the existing queue-inclusive admission comparison with the B=8 workload and record the receipt or the exact failing gate. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/b8-admission-verdict.md)
- [x] [serial] V6 Correct the first campaign's unmatched roots, attribute the startup-DRAM difference, and rerun the same predeclared comparison on the package the tree builds. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/b8-admission-verdict.md)

## Phase 4: Completion

- [x] [serial] D1 Record the capacity inventory, the smoke result, the admission receipt, and every non-claim; move no served default. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md, evidence/b8-admission-verdict.md)
- [x] [serial] D2 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/capacity-and-smoke.md)

## Phase 5: Re-admission on the promoted package

- [x] [serial] V7 Re-earn the B=8 scheduler admission on the promoted package `gzwghi7…` and identity `qwnxwc…` with the same predeclared policy and gates; receipt `pkgs/qwen36/evidence/2026-10-26/qwen38-continuous-slot-admission-b8-receipt.json`, BLAKE3 `e5487ac5…`; all eleven gates pass. The 2026-10-21 receipt is superseded as current and retained as history. r[tenstorrent.qwen38_p150x2.continuous_slot_capacity]  (evidence/re-admission-promoted-2026-10-26.md)
