# Onix Tenstorrent migration provenance

The Onix package snapshot in this repository was imported from
[`onixcomputer/onix-core`](https://github.com/onixcomputer/onix-core) commit
`3a42d49ad3b449e94ed74e46118b2645af08c3fa` on 2026-07-24.

The standalone repository retained the exact package baseline commit
`3c4dcbb9f8de30ae0d43177c8bc25f12170ddd94` that `onix-core` consumed before
migration. `OnixResearch/tenstorrent.nix` is now the sole repository authority;
no external upstream remote is configured or followed.

## Included boundary

- reusable RWKV and ttWKV7 packages and device-free checks;
- bounded device wrappers and parameterized owner-control tooling;
- immutable partial and passing evidence fixtures and validators;
- the Tenstorrent simulator package and VibeThinker benchmark core.

## Excluded boundary

Machine definitions, inventory, service definitions, encrypted secrets, deployment
state, and unrelated Cairn changes remain in `onix-core`. Fine-grained development
history for the imported snapshot also remains available there.

Migration validation is device-free. It does not claim fresh hardware execution,
performance, general P150 compatibility, or exact BF16 parity beyond the retained
receipts' explicit claim boundaries.

## Post-migration full-device development

The reusable RWKV-7 P150x2 runtime and classifier were reviewed and ported from
the historical `onix-core` branch `rwkv-full-device-20260718`. The source work
starts at commit `15e70fef` and the third owner-isolation revision ends at commit
`9c1bed8e`.

The standalone package removes machine-specific runbooks. Its retained attempt-3
fixture records that devices 0 and 1 opened before a profiler and Watcher firmware
conflict caused BRISC overflow. Device-free package checks do not convert that
failure into a physical execution success claim.

<!-- r[impl tenstorrent.rwkv7_p150x2.no_hardware_claim] -->
<!-- r[verify tenstorrent.rwkv7_p150x2.no_hardware_claim.complete] -->
