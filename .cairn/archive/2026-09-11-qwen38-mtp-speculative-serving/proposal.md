# Gated MTP speculative serving for the serialized greedy path

## Why

Every component of the MTP speculative round is now individually proven on
P150x2 against the repaired bf16 head: the draft chain is exact and cheap
(9.2 ms traced), the width-k native verifier is exact (PCC 1.0, 2.1x
amortization at width 4), snapshot rollback restores rejected state in
~12 ms, and the on-device pilot measured acceptance 2.4 drafts with a
1.382x aggregate emitted-token speedup at k=3 (1.333x at k=4). The fp8
conversion's MTP head was identified and removed as the acceptance killer.

What does not exist is the serving integration: the serialized service
decodes one token per traced step and has no speculative round, no
rollback in its request loop, and no economics guard.

## What Changes

- Add a pure, device-free serving round policy core: round scheduling
  (speculate only when the request has more than `draft_token_count`
  tokens remaining), the exact greedy commit rule (accepted draft prefix
  plus the verification bonus token, provably equal to sequential greedy
  output), and validation for every input.
- Add positive and negative device-free probes to the qwen36 check.
- Define the spec requirement: a gated speculative serving path MUST
  remain exact for greedy requests, MUST default off, MUST disengage on
  measured economics, and MUST NOT change sampled or batch behavior.

This change wires nothing into production by itself. The serving adapter,
trace-residency management, and the physical exactness screen are later
tasks inside this change and stay default-off until the owner reviews the
screen evidence.

## Impact

- **Files**: `pkgs/qwen36/service/qwen36_service/mtp_serving_core.py`,
  `pkgs/qwen36/check.py`, this change directory. Later tasks touch
  `pkgs/qwen36/service/qwen36_service/server.py` and the qwen36 module
  options.
- **Testing**: qwen36 device-free check (new probe), pinned Cairn strict
  validation.
- **Production**: unchanged. The gate defaults off and no serving decision
  changes in this change.
