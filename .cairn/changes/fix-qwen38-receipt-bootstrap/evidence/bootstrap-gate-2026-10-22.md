# Device-free bootstrap gate, 2026-10-22

**Verdict: failed as specified (`passed: false`).** The comparisons over the two
controlled exports all pass, but the unmeasured package does not build through
the standard derivation: `installCheckPhase` requires the shipped v3 export to be
admitted, so a tree differing only in the binding (unmeasured) fails closed.
The unmeasured output was therefore only obtainable with `doInstallCheck =
false`. No hardware, simulator, campaign, receipt, admission or promotion was
run. Nothing here is a fresh receipt, and the historical ratios stay unearned.

## Commands and exit statuses

```text
nix build .#qwen38 --no-link --print-out-paths                          -> exit 0
  /nix/store/l4ikvhpg0qaa7glq9khhkj5fq4rdgkbi-qwen38-0.1.0           (admitted fixture tree)

nix build path:/tmp/qwen38-bootstrap-admitted-fixture#qwen38 --no-link --print-out-paths
                                                                       -> exit 1
  drv /nix/store/xvr6w7r75y44ap59iwvwnv1ds1dd6kr8-qwen38-0.1.0.drv
  expected output /nix/store/s8h2q78ryfvbrf7xylkjilyq60jrqz94-qwen38-0.1.0 (not produced)
  build log: /tmp/fixture-build.log; phase reached installCheckPhase, then exit 1

nix build --impure --no-link --print-out-paths --expr \
  '((builtins.getFlake "path:/tmp/qwen38-bootstrap-admitted-fixture")
     .packages.x86_64-linux.qwen38).overrideAttrs (o: { doInstallCheck = false; })'
                                                                       -> exit 0
  /nix/store/b6z3951wdi6gmyi4wfg0dpq2wlj2i5hh-qwen38-0.1.0           (unmeasured, check bypassed)
```

Cause, confirmed directly against the bypassed export: the check block at
`pkgs/qwen36/default.nix:1862` runs
`grep -Fq '"physically_admitted": true' "$out/share/qwen36/batch-admission-b8-profile-v3.json"`.
The unmeasured export contains `"physically_admitted": false`, so that grep
fails and the phase aborts. This is a real blocker in the bootstrap packaging:
the runbook's required unmeasured package cannot be built normally while the
install check insists on admission. A gate rerun should follow a reviewed fix
that lets the install check accept the unmeasured state from the unmeasured
tree (while still proving the promotion delta), rather than bypassing it.

## How the trees were proven to differ only in the binding

Working tree at `5e7fbc0e5e7dc3968cbc9dd48cc20fe60cc4ec52`, clean; the
repository was not modified.

```text
rsync -a --exclude=.git --exclude=.pi --exclude=result --exclude='result-*' \
  /home/brittonr/git/tt-onix/ /tmp/qwen38-bootstrap-admitted-fixture/
cp pkgs/qwen36/fixtures/batch-admission-b8-binding-v3-not-measured.ncl \
  /tmp/qwen38-bootstrap-admitted-fixture/pkgs/qwen36/batch-admission-b8-binding-v3.ncl
```

Tracked-file comparison (`git add -A; git write-tree` inside the copy, then
`git ls-tree -r` on both trees; the copy's own index only, no repository
mutation):

```text
copy tree: de79a5999fe0aa12e17a5e5d09a224292f350343
diff /tmp/head-lstree.txt /tmp/copy-lstree.txt
1694c1694
< 100644 blob dbb5a0e296072eafe7e909c0cb49ad666f275361  pkgs/qwen36/batch-admission-b8-binding-v3.ncl
---
> 100644 blob a33c29f1a7f2a3286fc3211467a7b159ff2586a3  pkgs/qwen36/batch-admission-b8-binding-v3.ncl
```

`a33c29f1…` is `git hash-object` of the fixture file, so the only tracked
content difference is the v3 binding. `diff -r -x .git -x .pi -x result -x
'result-*'` reports the same single file and shows only the comment plus the
two binding fields changing (`true`/receipt BLAKE3 -> `false`/`not-measured`).

## Comparisons

Paths under each package `$PKG`:

- wrapper `bin/qwen38-p150x2-serve`
- manifest `share/qwen36/execution-identity/share/qwen36/execution-inputs`
- profile `share/qwen36/batch-admission-b8-profile-v3.json`

```text
                                    admitted (l4ikvhpg…)            unmeasured (b6z3951w…)
--package-identity                  knkqyk5z28n5s7014xda35xda0gmbxik  knkqyk5z28n5s7014xda35xda0gmbxik
execution-inputs sha256             215e3492c21949adb9afe90fa52c5a3141f2901e40b583937d96503792eb01b7  215e3492c21949adb9afe90fa52c5a3141f2901e40b583937d96503792eb01b7
profile sha256                      a8de639df69dea28fe3dee090a6bd1258c502825e3006171a2aa9c4973619e4c  f01bdbaf7569bf96e2e18783ab90e671cc4d3dac90375aa302b81abae3849f72
```

`cmp` on the two execution-inputs files exits 0; the profile `cmp` exits 1.

Inline checker (exit 0, stdout):

```text
receipt_blake3: af2365ffb09002bb5674e175374efda1a81ea1a665c1f199b16a069344ac785e
admitted fields: (True, 'af2365ffb09002bb5674e175374efda1a81ea1a665c1f199b16a069344ac785e')
not-measured fields: (False, 'not-measured')
remainder equal: true
RESULT: PASS
```

The checker `b3sum`ed `pkgs/qwen36/evidence/2026-10-22/qwen38-batch-admission-b8-v3-receipt.json`,
loaded both JSONs, required those two field tuples, deleted exactly
`physically_admitted` and `identities.receipt_blake3` from each, and required
the remaining objects to compare equal (printing a unified diff and failing
otherwise).

So the measured bootstrap contract holds across the two exports: equal
execution identity and byte-identical immutable input manifests, with the
profile differing exactly in the two admission-derived fields and naming the
committed receipt digest on the admitted side. The gate still fails because
the unmeasured package is not buildable by the specified command.

## Non-claims

- No device was touched, no service was started or stopped, no deployment.
- No campaign, warmup, response, receipt, admission or promotion; no timing,
  latency, speedup or retention claim.
- The existing receipt's identity `l4arf461…` is neither reproduced nor used;
  only its BLAKE3 is named by the admitted export.
- Passing comparisons here do not substitute for the re-earn or the promotion
  gate, and the unmeasured build bypass must not become the measurement-time
  recipe without a reviewed packaging fix.

## Correction: the package-build form of this gate is wrong

The original formulation above measured the wrong thing and is retained as a
failure record, not as a waiver.

(a) An unmeasured package **cannot** build normally and **must not** be made to
build. `installCheckPhase` requires the shipped v3 export to be admitted, so the
fixture tree fails closed by design. The `doInstallCheck = false` output
(`b6z3951w…`) is recorded only as diagnostic evidence; nobody may serve it, and
it is **not** used as the gate.

(b) The gate the campaign actually needs is:

- identity equality between the admitted package and the fixture tree whose only
  difference is the binding text, computed device-free from the identity
  derivation (`passthru.executionIdentity`, no package build) and confirmed on
  the byte-identical input manifest;
- the admitted package's install check passing;
- the unmeasured build failing closed with the expected admission check.

The identity derivation proves the binding is not an identity input: the fixture
tree resolves to the **same** execution identity store path
(`knkqyk5z28n5s7014xda35xda0gmbxik-qwen36-execution-identity-0.1.0`) and the
manifest in that identity output (`share/qwen36/execution-inputs`, SHA-256
`215e3492…`) is byte-identical to the admitted package's installed manifest.

(c) The campaign therefore measures the **admitted** package. After a fresh
receipt exists, the post-promotion rebuild is a legitimate substitute for the
measured package because the execution identity is identical and the profile
delta is exactly `physically_admitted` and `identities.receipt_blake3`; the
promotion gate must repeat the identity comparison against the fresh receipt.

### Corrected gate evidence (device-free)

```text
nix eval --raw .#qwen38.passthru.executionIdentity
  -> /nix/store/knkqyk5z28n5s7014xda35xda0gmbxik-qwen36-execution-identity-0.1.0
nix eval --raw --impure --expr '((builtins.getFlake "path:/tmp/qwen38-bootstrap-admitted-fixture").packages.x86_64-linux.qwen38).passthru.executionIdentity'
  -> /nix/store/knkqyk5z28n5s7014xda35xda0gmbxik-qwen36-execution-identity-0.1.0
```

Identity manifest: `$identity/share/qwen36/execution-inputs` SHA-256
`215e3492c21949adb9afe90fa52c5a3141f2901e40b583937d96503792eb01b7`, `cmp`-equal
to the admitted package's installed manifest and to the diagnostic fixture
package's manifest.

Corrected checker stdout (exit 0):

```text
identity equality: PASS /nix/store/knkqyk5z28n5s7014xda35xda0gmbxik-qwen36-execution-identity-0.1.0
input manifests byte-identical: PASS sha256=215e3492c21949adb9afe90fa52c5a3141f2901e40b583937d96503792eb01b7
exact two-field profile delta: PASS; fixture receipt_blake3=af2365ffb09002bb5674e175374efda1a81ea1a665c1f199b16a069344ac785e
unmeasured build fail-closed condition: PASS (shipped export is not physically_admitted)
RESULT: PASS
```

The fail-closed evidence is the fixture tree's standard build exit 1
(`xvr6w7r75y44ap59iwvwnv1ds1dd6kr8-qwen38-0.1.0.drv`): the install check's
admission condition rejects `physically_admitted: false` / `"not-measured"`.
