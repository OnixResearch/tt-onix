# The served Qwen deployment on britton-desktop

The service on `127.0.0.1:8000` runs the Qwen3.8-27B package that the repository
builds, through the managed drop-in and the named Nix profile. This note is the
standing procedure for moving it, so that a package change does not need a hand-run
root command each time.

## What is served, and what it must carry

The served command is the package's `qwen38-p150x2-serve` with the eight-slot
continuous-slot policy and the hybrid prefix cache:

```text
/nix/var/nix/profiles/qwen38-p150x2/bin/qwen38-p150x2-serve --host 127.0.0.1 --port 8000 \
  --model-path /home/brittonr/models/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --model-alias Qwen3.8-27B --max-sequence-length 131072 --max-generation-tokens 8192 \
  --require-systemd-notify \
  --continuous-slot-policy /nix/var/nix/profiles/qwen38-p150x2/share/qwen36/continuous-slot-policy-b8.json \
  --prefix-cache-policy /nix/var/nix/profiles/qwen38-p150x2/share/qwen36/prefix-cache-policy.json
```

Two admission receipts name the package that is served, and a package change
invalidates both:

- the eight-slot scheduler admission,
  `pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`;
- the prefix-cache admission,
  `pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json`.

Both bind a package path and an execution identity. A rebuild of the same tree is a
different package, so a new build needs its own receipts before it is served. The
repository's rule is in
`.cairn/changes/readmit-qwen38-prefix-cache-b8/specs/qwen38-p150x2-runtime/spec.md`:
the served package carries the receipts of the features it serves.

## How the served version is chosen

The systemd owner and its drop-in are root-owned; they name a stable path that the
operator owns:

```text
/home/brittonr/.local/state/qwen38-p150x2/current
```

That path is a symlink to the package that is served. The root-owned layer pins the
shape of the command, the flags, and the environment; the operator's layer pins the
version. A package change is therefore a symlink update and a restart, with no root
and no password.

## Moving the service to a new package

`scripts/qwen38-deploy` does the whole move and then reads the served instance back.
It builds the package from the tree, refuses to serve a package that no admission
receipt names, performs the move, waits for readiness, and prints the profile and
identity the service reports:

```text
scripts/qwen38-deploy
scripts/qwen38-deploy --package /nix/store/<hash>-qwen38-0.1.0
```

The receipt guard is the point of it: a rebuild of the same tree is a different
package, and the service must not drift off a package that no receipt names. Pass
`--force` to serve such a package deliberately.

The move itself is the profile service's job. It validates the exact store output,
repoints the operator's symlink, restarts the owner through the host's passwordless
stop and start rules, and waits for readiness. If the service does not become ready
it points the symlink back at the previous package and restarts again.

```text
scripts/qwen38-profile-service deploy /nix/store/<hash>-qwen38-0.1.0
```

The profile service is host-local and is not in version control: it names this
host's model path, its service unit, and the operator's path. It needs no root.

### The one-time setup

The host's sudo rules already allow the unit's `systemctl stop` and `systemctl start`
without a password, which is what the move and the measurement campaigns need. No
new rule is needed.

The one root action is installing the managed drop-in, which names the operator's
path. It is done once, from a root shell:

```sh
/home/brittonr/git/tenstorrent.nix/scripts/qwen38-profile-service install-dropin
```

After that, the served version is the operator's symlink, and a package change is:

```text
ln -sfn /nix/store/<hash>-qwen38-0.1.0 /home/brittonr/.local/state/qwen38-p150x2/current
sudo -n systemctl stop qwen38-p150x2.service
sudo -n systemctl start qwen38-p150x2.service
```

`scripts/qwen38-deploy` does exactly that, with the receipt guard in front of it.

The operator's shell is fish, which has no heredoc. Copy commands from a file or a
terminal rather than from a chat window: a soft wrap in a terminal is harmless, but a
hard line break inside a command can break it.

## Measuring while the service runs

A measurement campaign needs the devices, and the served service owns them. The
campaign scripts stop the unit before the first root and start it again when they
finish, whether they succeed or fail:

```text
sudo /nix/store/sm8d6jpilwdy3bw3yq2lv8rr8jld26pb-systemd-261.2/bin/systemctl stop qwen38-p150x2.service
... measurement ...
sudo /nix/store/sm8d6jpilwdy3bw3yq2lv8rr8jld26pb-systemd-261.2/bin/systemctl start qwen38-p150x2.service
```

The campaign runs its own serve processes on another port, so the measurement is not
the served instance: the receipts name the package under test, and the served
instance is what the deployment switch changes.

## Verifying what is served

The service reports its own identity in each response's decode evidence. After a
switch, confirm the served profile and the identity:

```text
curl -s http://127.0.0.1:8000/healthz
scripts/qwen38-profile-service status
```

The response evidence names `decode_path` as `continuous-slot`, the scheduler profile
as `qwen38-p150x2-continuous-slot-b8-v1`, the prefix-cache profile as
`qwen38-p150x2-prefix-cache-v1`, and the package and runtime identities the receipts
name.

## Non-claims

- The switch moves the service between packages; it does not decide whether a package
  is admissible. The receipts do that.
- The historical owner command, the original `/run` rollback overrides, and the
  managed drop-in are host state, not repository state.
- A running service is a convenience for serving. It does not change any receipt.
