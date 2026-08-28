## Why

The B=4 candidate passed every typed physical gate, but its receipt binds a monolithic package path. Updating admission metadata changes that path without changing executable behavior. This promotion cycle prevents a truthful deployed admission.

## What Changes

- Separate immutable execution policy from mutable receipt-binding metadata.
- Add a Nix execution-identity artifact that binds the model source, service code, patches, Python environment, TT-Metal runtime, and execution policy.
- Make served evidence report that stable execution identity while the deployment envelope carries the receipt and admission decision.
- Prove that receipt-only promotion preserves execution identity and that execution input changes do not.
- Run exact `single,batch,batch,single` roots on the unadmitted stable artifact.
- Admit B=4 only if every typed gate passes and the final deployed closure references the measured execution identity.

## Impact

- **Files**: `pkgs/qwen36`, the Qwen runtime specification, admission evidence, and adoption documentation.
- **Testing**: Baseline and post-change package checks, identity stability checks, positive and negative receipt tests, matched physical roots, Cairn gates, Onix closure checks, and final production verification.
