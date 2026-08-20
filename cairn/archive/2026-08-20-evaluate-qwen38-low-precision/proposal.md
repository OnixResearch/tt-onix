## Why

The package has opt-in BF4 flags for selected Qwen3.8 weight families.
The production service does not enable those flags and keeps their BFP8 defaults.

No named low-precision profile has complete operator, logit, recurrent-state, token-parity, and matched physical performance evidence.

## Outcome

Return every BF4 candidate as experimental and unadmitted.
Keep BFP8 as the only production precision profile for the affected weight families.

This result does not reject BF4 on technical merit.
It withholds admission because the required evidence set does not exist.

## What Changes

- Record the active service environment and BFP8 default.
- Keep all existing BF4 gates opt-in and outside production admission.
- Add no named production precision profile or speedup claim.
- Require a new validated Cairn change for future low-precision admission.
- Require layered numerical and matched physical evidence in that change.

## Impact

- **Service**: the production precision remains unchanged.
- **Runtime**: existing experimental flags remain opt-in.
- **Configuration**: no admitted BF4 profile is added.
- **Evidence**: the archive records the missing admission proof.
- **Documentation**: no BF4 quality or performance claim is allowed.

## Out of Scope

- Removing experimental BF4 flags.
- Estimating speed from weight byte counts.
- Claiming semantic quality from PCC alone.
- Changing activation or recurrent-state precision.
