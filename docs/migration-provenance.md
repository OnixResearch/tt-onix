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
