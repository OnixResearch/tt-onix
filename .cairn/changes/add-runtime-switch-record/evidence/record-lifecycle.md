# Record lifecycle evidence

Captured 2026-10-26 on britton-desktop, unprivileged. The state root and GC-root
directory are temporary overrides, so nothing outside `/var/tmp` was written and
no store path outside the two roots was touched.

## Command

```text
TT_MODEL_STATE_ROOT=/var/tmp/rt/hoststate \
TT_MODEL_GC_ROOTS=/var/tmp/rt/hostroots \
TT_MODEL_SYSTEMCTL=none \
  /nix/store/<tt-model>/bin/tt-model record qwen38 \
    --package /nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0 \
    --repo-root . --force --json
```

Exit 0. The real Qwen package is admitted by no receipt, so `--force` was
required and the record says so:

```json
{
  "active": "/var/tmp/rt/hoststate/active.json",
  "admission_bypassed": true,
  "argv_blake3": "543a4911c5fd032e1cb97d80fe2c5cb07d9ec9028df073eeea554a0cbaa6bd81",
  "gc_roots": {
    "execution-identity": "/var/tmp/rt/hostroots/qwen38-execution-identity",
    "package": "/var/tmp/rt/hostroots/qwen38-package"
  },
  "history": "/var/tmp/rt/hoststate/history/20261026T150821.957514Z-qwen38.json",
  "package": "/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0",
  "profile": "qwen38",
  "registry_blake3": "01f4b10ca914aa693105cb36950c88ca15a25e9a06436c6bd4b9f2304899d821",
  "schema": "onix.tt-model.record.v1"
}
```

Only whole store paths became roots; the runtime root
`<package>/libexec/tt-metalium` is a subpath, so it is recorded but not rooted,
because the collector does not accept a subpath as a root target.

## The record

```text
schema           = onix.tt-model.record.v1
profile          = qwen38
profile_schema   = onix.runtime-profile.v1
mode             = serve
admission_bypassed = True
recorded_at      = 2026-10-26T15:08:21.957514+00:00
recorded_by      = tt-model 0.1.0 uid=1555
argv_blake3      = 543a4911c5fd032e1cb97d80fe2c5cb07d9ec9028df073eeea554a0cbaa6bd81
registry_blake3  = 01f4b10ca914aa693105cb36950c88ca15a25e9a06436c6bd4b9f2304899d821
package          = {path: /nix/store/inkfvs…-qwen38-0.1.0,
                    store_name: -qwen38-0.1.0,
                    execution_identity: /nix/store/fk8p9n…-qwen36-execution-identity-0.1.0}
runtime          = {kind: package-root, root: libexec/tt-metalium,
                    path: /nix/store/inkfvs…-qwen38-0.1.0/libexec/tt-metalium}
checkpoint       = {path: /home/brittonr/models/1d4bf0f2…,
                    revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,
                    authority_kind: generated-authority-blake3,
                    authority_path: /home/brittonr/git/tt-onix/pkgs/qwen36/generated/authority.json,
                    authority_blake3: c319f7ee…}
receipts         = [(continuous-slot-admission, admitted, 4573cea1…),
                    (prefix-cache-admission, admitted, eb09b2cc…)]
gates            = []
port             = 8000
reset            = required-before-native
env keys         = HF_HUB_OFFLINE, HF_MODEL, MESH_DEVICE, QWEN35_GDN_DECODE_FUSION,
                   QWEN35_GROUPED_PREFILL, QWEN35_TEMP, QWEN36_GDN_BF4,
                   QWEN36_GDN_FUSED_PREFILL, QWEN36_MTP_SERVING, QWEN36_SAMPLED_FUSION,
                   QWEN36_TRACED_PREFILL, TT_CACHE_PATH
previous         = None
```

`argv` is the full resolved command line, including the two policy paths expanded
from `@package@`.

## Roots and history on disk

```text
/var/tmp/rt/hostroots/qwen38-package -> /nix/store/inkfvs…-qwen38-0.1.0
/var/tmp/rt/hostroots/qwen38-execution-identity -> /nix/store/fk8p9n…-qwen36-execution-identity-0.1.0
/var/tmp/rt/hoststate/active.json
/var/tmp/rt/hoststate/history/20261026T150821.957514Z-qwen38.json
```

## Status against the record

With the legacy sources pointed at empty directories, the record alone is
`managed`, exit 0:

```text
verdict: managed
active record: /var/tmp/rt/hoststate/active.json
  profile: qwen38
  package: /nix/store/inkfvs…-qwen38-0.1.0
  execution identity: /nix/store/fk8p9n…-qwen36-execution-identity-0.1.0
operator symlink: absent
nix profile: absent
unit: unavailable (unit probe disabled by TT_MODEL_SYSTEMCTL)
```

With the real host sources, the same record is `broken`, exit 1, because the
operator symlink disagrees and dangles:

```text
verdict: broken
nix profile: ok (/nix/var/nix/profiles/qwen38-p150x2, generation qwen38-p150x2-5-link)
  packages: /nix/store/inkfvs…-qwen38-0.1.0
problem: operator symlink: /home/brittonr/.local/state/qwen38-p150x2/current: dangling symlink to /nix/store/wlsl6…-qwen38-0.1.0
```

## Sandbox lifecycle

The install check also exercises, in `$TMPDIR`: a forced record then `status`
managed with both roots; a second record chaining `previous` with two history
entries; a deleted package root, a missing package path, an unknown profile, and
a legacy symlink that disagrees, each failing with its own message; and a held
lock refusing a concurrent record.

## Non-claims

The record here names a real package and a real checkpoint, but nothing acts on
it. No unit was started or stopped, no device was opened, and no switch occurred.
