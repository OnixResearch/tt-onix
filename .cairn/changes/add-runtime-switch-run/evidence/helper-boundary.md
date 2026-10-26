# Helper boundary evidence

Captured 2026-10-26. The helper was built and exercised only against fake
`systemctl`, `lsof`, `tt-smi`, and record tools, and on the host only on paths
that refuse before any privileged command runs. No unit, device, or firmware was
touched.

## The boundary in both directions

- `pkgs/tt-model/default.nix` asserts that `$out/bin` and `tt_model.py` contain
  no `/dev/tenstorrent`, `tt-smi`, or `ttwkv7-owner-control`.
- `pkgs/tt-model-owner/check.nix` asserts that the helper *does* contain
  `/dev/tenstorrent` and `tt-smi`, and that it does *not* use the legacy
  `ttwkv7-owner-control`.

That pair is the invariant: the device-free tool cannot grow device access, and
the helper cannot lose the only code that knows how to stop a unit or reset a
board.

## Commands

```text
nix build .#checks.x86_64-linux.tt-model-owner
nix build .#tt-model
/nix/store/<tt-model-owner>/bin/tt-model-owner cards
/nix/store/<tt-model>/bin/tt-model run qwen38 --package /nix/store/…-qwen38-0.1.0
```

## Sandbox result

```text
tt-model-owner check: PASS
```

It exercised: `--help`; the state-root and GC-root override refusals; the missing
pointer; a pointer to another store path; `cards` with a fake `lsof` (owner, PID,
and the holder line); `reset` with the printed command and the fake `tt-smi`
invocation; `stop-owner` on an inactive unit; and `record` delegating to the
record tool with the pinned state root and GC-root directory.

## Host refusals

Without an installed pointer, the helper refuses before it resolves or runs any
command:

```text
$ tt-model-owner cards
tt-model-owner: no owner helper is installed at /var/lib/tenstorrent-models/owner-helper; refusing to act
exit 1
```

A serve profile is refused before the helper is consulted at all:

```text
$ tt-model run qwen38 --package /nix/store/inkfvs…-qwen38-0.1.0
tt-model: run refused: mode: qwen38: qwen38 is a serve profile; use `tt-model switch` to make it the owner
exit 1
```

Both stop before any privileged operation, which is why they were safe to run on
this host.

## The helper contract

| Subcommand | What it does | Refuses |
|---|---|---|
| `cards` | reports the owner unit, owner PID, and every holder as `<device> <pid> <unit> <command>` | overrides without the test flag; an unrecorded helper |
| `stop-owner` | stops the declared unit and verifies it stopped | inactive is reported, not an error |
| `start-owner` | starts the declared unit and verifies it is active | a masked unit |
| `reset` | runs the documented board reset over the declared devices | an empty device list |
| `record` | delegates to `tt-model record` with the pinned state and GC roots | arguments it cannot pass through |

Every acting command is printed with its exact argv before it runs, as `%q`-escaped
tokens on stderr.

## Non-claims

No unit was started or stopped, no device was opened, and no reset was performed.
The fake tools are the only executors in this evidence.
