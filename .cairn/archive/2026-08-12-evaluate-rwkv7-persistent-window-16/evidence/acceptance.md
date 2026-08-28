# Window-16 evaluation result

## Decision

Window `16` is rejected. Production remains on profile `rwkv7-p150x2-persistent-decode-v3` with admitted windows `[2, 4, 8]`.

The candidate remained non-admitted during all work. No production profile or deployment changed.

## Passed device-free gates

- The immutable candidate profile ID is `rwkv7-p150x2-persistent-decode-window16-candidate-v1`.
- Its BLAKE3 identity is `0fd161f7172276d1efbe51623b1d110947e68de40024109618547cf7a42bb01a`.
- The candidate runtime, evidence package, source boundary, and closure checks passed.
- The evidence package passed 145 library tests and all binary tests.
- The pinned FP32 CPU reference was deterministic across two runs.
- FLA `0.3.0` and official RWKV-LM revision `952102498e9ed367ea0a59ee64106916d474d30f` produced equal tokens.
- Their maximum state difference was `9.5367431640625e-06`.
- Their maximum final-logit difference was `1.9073486328125e-05`.
- The CPU receipt BLAKE3 is `6995c6bdb32eadfac6cb85b77eb1d131cc9af6548a7fc6d6ae32e54145935cbe`.

## Simulator result

The final simulator run passed with BLAKE3 `03d7c8438bda0cb1d96bb4a3e753ced4fd7dca9c17b917bb4a3ffcd8468f70f7`.

It proved one finite 16-step trace, one replay submission, no per-step host submissions, finite state, nonzero state, and state zeroing before release. It made no physical, performance, determinism, or admission claim.

## Physical correctness failure

The first physical A/B sample failed before throughput acceptance.

Normal dispatch on device `0` and persistent trace on device `1` produced the same 16 tokens:

`[36786, 34, 308, 459, 39743, 4811, 31063, 32227, 22799, 5230, 64390, 32074, 64356, 32487, 32515, 46705]`

The pinned CPU continuation is:

`[36786, 34, 308, 459, 39743, 4811, 31063, 32227, 22799, 5230, 46170, 53200, 32487, 32515, 55809, 47]`

The first mismatch is at zero-based index `10`. Six of the final six tokens differ. The harness stopped with `card 0 has invalid request attribution` because exact CPU token attribution is required.

The normal receipt BLAKE3 is `e9005a8347c9b2bd893477490335577376f4d4f22fd7e85f842e1a67d528cedf`. The persistent receipt BLAKE3 is `7db1a2bf950697fb04c2a0a9e65a74cb4d6b16906a1d67f9e164fe8faa382bf9`.

The persistent worker still completed four requests, acknowledged them, zeroed state, and released every slot. That cleanup does not override the token-parity failure.

## Stopped gates

The valid negative result ended the evaluation. The following gates did not run or did not reach acceptance:

- the separate completion, cancellation, timeout, and failure probe set;
- final profiler acceptance;
- five order-balanced one-card and two-card A/B samples;
- the bounded canary;
- aggregate monitoring and a `retain` recommendation.

The partial profiler preflight is not final profiler evidence. The aborted physical run is not throughput evidence.

## Required follow-up

A new evaluation can start only after the CPU continuation mismatch is fixed or explained. It must use a new immutable candidate and repeat every physical gate.
