## Why

The P150x4 evaluation requires four compatible physical P150 devices. The available host exposes only device nodes `0` and `1`.

The operator confirmed that no additional devices are available. A physical P150x4 comparison cannot run on this host.

## Outcome

Return the P150x4 deployment candidate as unsupported on the available hardware.

This result does not reject P150x4 on technical merit. It rejects local support because the required physical topology is absent.

## What Changes

- Record the exact hardware-availability result.
- Keep the admitted P150x2 deployment unchanged.
- Add no P150x4 package, service, runtime, precision, or performance claim.
- Require a new Cairn change before any future P150x4 support claim.
- Preserve upstream P150x4 results as external information only.

## Impact

- **Deployment**: P150x4 remains unsupported.
- **Runtime**: no runtime code changes.
- **Configuration**: no P150x4 profile is added.
- **Evidence**: the archive records the two-device limit and operator decision.
- **Documentation**: no local P150x4 capacity or efficiency claim is allowed.

## Out of Scope

- Buying, renting, or reserving more devices.
- Estimating physical P150x4 performance from P150x2 results.
- Treating simulator or upstream results as local physical evidence.
- Changing the admitted P150x2 service.
