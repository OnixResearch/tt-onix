# Corrected window-16 v2 evaluation result

## Decision

Reject candidate v2. Production remains on `rwkv7-p150x2-persistent-decode-v3` with windows `[2, 4, 8]`.

The candidate remained non-admitted. No deployment or observation started.

## Passed bounded gates

The immutable v2 profile ID is `rwkv7-p150x2-persistent-decode-window16-candidate-v2`.

Its BLAKE3 identity is `acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6`.

The package, source, closure, formatting, repository-quality, and Cairn gates passed. The preserved v1 runtime checks also passed.

The bounded Blackhole simulator smoke passed. Its aggregate receipt BLAKE3 is `b15d3d43a7683e886640837b1b490e5b3a9a023e2e52ef7df13572ac88e9a835`.

The simulator result makes no physical, performance, determinism, or admission claim.

## Physical correctness rejection

One bounded worker pair ran on physical devices `0` and `1`.

Normal dispatch and persistent trace produced the same token sequence:

`[36786, 34, 308, 459, 39743, 4811, 31063, 32227, 22799, 5230, 64390, 32074, 64356, 32487, 32515, 56537]`

The pinned CPU continuation is:

`[36786, 34, 308, 459, 39743, 4811, 31063, 32227, 22799, 5230, 46170, 53200, 32487, 32515, 55809, 47]`

The first mismatch is at zero-based index `10`. Six tokens differ.

The normal receipt BLAKE3 is `abcde0bb7c1539133a0a594c940e855bcd6641be2485d75956e67f7cb3648bf2`.

The persistent receipt BLAKE3 is `f0cd9446fdde594f306ce60f35f0185c61195228d6a44b4770827fef2359acda`.

The persistent trace matched its normal-device reference state exactly. Both state hashes are `6bcab4d2b093ad2614cbbde6c3b7620479e6b975f8dd2425bff357727c72eeff`.

This device-only state agreement does not establish CPU state parity. The failed CPU token gate rejects the candidate first.

All four persistent requests completed. The runtime acknowledged, zeroed, and released each slot.

The managed service stayed inactive. Both physical devices were unowned after the probe.

## Stopped gates

The following gates did not run after the correctness failure:

- terminal completion, cancellation, timeout, and failure probes;
- final profiler acceptance;
- five order-balanced one-card and two-card A/B samples;
- bounded canary and monitoring;
- production admission;
- deployment and observation.

The bounded rejection is the terminal evaluation result. A later attempt needs another diagnosis and a new immutable candidate.
