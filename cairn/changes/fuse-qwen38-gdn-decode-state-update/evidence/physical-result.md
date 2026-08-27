# Physical result

Status: pending

The fixed order is `control,candidate,candidate,control` on physical devices `0` and `1`.
The fixed candidate modes are `norm`, `state`, and `full`.

## Required evidence

- Package, runtime, checkpoint, tokenizer, mesh, and device identities.
- Exact generated tokens and text.
- Distinct recurrent and KV state slices.
- Clean reset and cleanup.
- At least five retained GDN and full-step samples per root.
- Duplicate-root spread at most `1.25`.
- Candidate GDN median ratio at most `0.95`.
- Candidate full-step median ratio at most `0.97`.
- Fresh B=4 admission for a selected identity.

## Non-claims

- Static operation counts do not prove speedup.
- Independent TT-Lang results do not transfer to TT-Metal 0.74.
- A component win does not admit production.
- No TT-Metal pin change is included.
