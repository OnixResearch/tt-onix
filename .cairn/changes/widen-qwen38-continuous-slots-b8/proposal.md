## Why

The group coordinator admits widths 8 and 16 on measured receipts, but the served
continuous-slot scheduler is contract-frozen at four slots. Its policy says "B4 only;
wider traces are out of scope", the Nickel contract pins `slot_capacity == 4`, and the
pure core rejects anything else. A deployed service therefore cannot serve more than four
concurrent users through the scheduler path, however well width 8 or 16 performs.

The width-8 decode trace already works: the B=8 admission binary captures and replays it,
and the scheduler's remaining width plumbing is a single `begin_slots(width=...)` call.
The gap is policy and validation, not a new kernel.

## What Changes

- Make the scheduler's slot capacity a profile-declared value in a closed set {4, 8},
  instead of a module constant equal to 4. Values outside the set stay refused.
- Add a B=8 policy and contracts with the same queue, window, refill, and deadline
  bounds as B4 and `slot_capacity = 8`. Keep B4 as the served default.
- Bind the B=8 policy into the package identity and export it beside B4.
- Extend the device-free probes: the core accepts the closed-set capacities, refuses
  everything else, and the B=8 policy exports deterministically.
- Run a bounded B=8 scheduler smoke on the production serve binary, then the existing
  continuous-slot admission comparison with the B=8 workload before admitting anything.

## Impact

- **Files**: `continuous_slot_core.py`, `continuous_slot_service.py` documentation, new
  B=8 policy and contracts, package exports and identity inputs, `check.py` probes, this
  change directory, and its evidence.
- **Testing**: the device-free Wiki check, package build, the existing scheduler probes,
  and a physical B=8 refill/coexistence smoke. The full admission comparison is a
  separate task with its own receipt.
- **Production**: the B4 profile remains the deployed default. The B=8 profile is a
  candidate until its own admission receipt passes; no default changes here.
