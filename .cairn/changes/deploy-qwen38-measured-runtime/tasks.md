# Tasks: Move the served Qwen3.8 runtime onto the measured execution identity

## Phase 1: Preconditions

- [x] [serial] A1 Confirm what the deployed system actually runs: generation, package, runtime identity, decode mode, and compare against every admission receipt this repository holds. r[tenstorrent.qwen38_p150x2.runtime_deployment]
- [x] [serial] A2 Confirm what a rebuild of this repository would deploy through the NixOS module defaults, and record the resulting package, runtime, and execution identities. r[tenstorrent.qwen38_p150x2.runtime_deployment]
- [x] [serial] A3 State explicitly which admitted capability the switch does and does not carry, including why the batch-admission route stays off in the service. r[tenstorrent.qwen38_p150x2.runtime_deployment.invalid]

## Phase 2: Preparation

- [x] [serial] I1 Run a readiness gate on the exact serving artifact a rebuild selects: start it, wait for readiness, serve one greedy request, and record the served identity. r[tenstorrent.qwen38_p150x2.runtime_deployment.valid]
- [ ] [serial] I2 Locate the host configuration that imports this module and record the exact edit that selects the measured package. r[tenstorrent.qwen38_p150x2.runtime_deployment]
- [ ] [serial] I3 Repair the artifact manifest so the artifact gate reports drift instead of a merge syntax error, without regenerating it past its owner's recorded decision. r[tenstorrent.qwen38_p150x2.runtime_deployment]

## Phase 3: Verification

- [ ] [serial] V1 Switch the host and confirm the service comes up on the new package and runtime identity. r[tenstorrent.qwen38_p150x2.runtime_deployment.valid]
- [ ] [serial] V2 Confirm the post-switch service answers a greedy request with the admitted token prefix and reports the new identities. r[tenstorrent.qwen38_p150x2.runtime_deployment.valid]
- [ ] [serial] V3 Confirm readiness, health, and the single-stream gate status after the switch, and record that batch serving remains off. r[tenstorrent.qwen38_p150x2.runtime_deployment.invalid]
- [ ] [serial] V4 Confirm the shared 0.77 output and the RWKV-7 and training checks are unaffected by the host-side change. r[tenstorrent.qwen38_p150x2.runtime_deployment.unchanged]

## Phase 4: Completion

- [ ] [serial] D1 Record the before/after receipt pair and every non-claim, including that no batch width is newly served. r[tenstorrent.qwen38_p150x2.runtime_deployment]
- [ ] [serial] D2 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.runtime_deployment]

## Verification Coverage

- `Scenario: The served artifact matches a measured execution identity` -> V1, V2, V3
- `Scenario: The switch carries no new admission` -> A3, V3, D1
- `Scenario: The shared runtime and its consumers are untouched` -> V4
