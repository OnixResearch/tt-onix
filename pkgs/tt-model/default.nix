# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.status]
# Read-only status command for the runtime-profile registry. It ships the
# registry export beside its own script and reads nothing else at build time.
#
# The command is hardware tooling: it must never be wired into the device-free
# quality gate, and its program text must not name a device node, tt-smi, or the
# owner helper. Its install check asserts both.
{
  lib,
  stdenvNoCC,
  python3,
  makeWrapper,
}:
stdenvNoCC.mkDerivation {
  pname = "tt-model";
  version = "0.1.0";

  dontUnpack = true;
  # python3 is the wrapper target and the install check's JSON reader.
  nativeBuildInputs = [
    makeWrapper
    python3
  ];

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin" "$out/share/tt-model"
    install -m 0444 ${./tt_model.py} "$out/share/tt-model/tt_model.py"
    install -m 0444 ${../../model-targets/generated/runtime-profiles.json} \
      "$out/share/tt-model/runtime-profiles.json"
    makeWrapper ${python3}/bin/python "$out/bin/tt-model" \
      --add-flags "-P" \
      --add-flags "$out/share/tt-model/tt_model.py"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    set -euo pipefail
    export HOME="$TMPDIR/home"
    export TT_MODEL_STATE_ROOT="$TMPDIR/state"
    export TT_MODEL_GC_ROOTS="$TMPDIR/gcroots"
    export TT_MODEL_PROFILE_ROOT="$TMPDIR/profiles"
    export TT_MODEL_SYSTEMCTL=none
    mkdir -p "$HOME" "$TT_MODEL_STATE_ROOT" "$TT_MODEL_GC_ROOTS" "$TT_MODEL_PROFILE_ROOT"

    # The shipped registry is the reviewed export, byte for byte.
    cmp "$out/share/tt-model/runtime-profiles.json" \
      ${../../model-targets/generated/runtime-profiles.json}

    # The program text never names a device node, tt-smi, or the owner helper.
    # The registry data is exempt: it records the device paths a launcher needs.
    if grep -R -E '/dev/tenstorrent|tt-smi|ttwkv7-owner-control' \
        "$out/bin" "$out/share/tt-model/tt_model.py"; then
      echo "tt-model program text names hardware tooling" >&2
      exit 1
    fi

    # A skipped unit probe is reported, not hidden: it degrades the verdict.
    "$out/bin/tt-model" status --json >status-degraded.json
    grep -F '"verdict": "degraded"' status-degraded.json
    grep -F '"schema": "onix.tt-model.status.v1"' status-degraded.json
    python3 - status-degraded.json <<'PY'
    import json
    import sys

    report = json.load(open(sys.argv[1]))
    assert report["problems"] == [], report["problems"]
    assert report["active"]["state"] == "absent"
    assert report["legacy"]["unit"]["state"] == "unavailable"
    assert report["unavailable"] == ["unit"]
    assert set(report["registry"]["profiles"]) == {
        "metalium-rs-rwkv7",
        "nanbeige43",
        "qwen38",
        "rwkv7-ttnn",
    }
    PY
    "$out/bin/tt-model" status >status-degraded.txt
    grep -F 'verdict: degraded' status-degraded.txt
    grep -F 'unit: unavailable' status-degraded.txt

    # Every source readable and absent is a clean "none".
    cat >"$TMPDIR/systemctl" <<'SH'
    #!/bin/sh
    printf '%s\n' \
      LoadState=not-found \
      UnitFileState=not-found \
      ActiveState=inactive \
      SubState=dead \
      FragmentPath=
    SH
    chmod +x "$TMPDIR/systemctl"
    TT_MODEL_SYSTEMCTL="$TMPDIR/systemctl" "$out/bin/tt-model" status --json >status-none.json
    grep -F '"verdict": "none"' status-none.json
    grep -F '"state": "not-found"' status-none.json
    TT_MODEL_SYSTEMCTL="$TMPDIR/systemctl" "$out/bin/tt-model" status >status-none.txt
    grep -F 'verdict: none' status-none.txt
    grep -F 'unit: not-found' status-none.txt

    # A dangling operator symlink is reported by path and target, and fails.
    mkdir -p "$(dirname "$HOME/.local/state/qwen38-p150x2/current")"
    ln -s /nix/store/does-not-exist-qwen38-0.1.0 \
      "$HOME/.local/state/qwen38-p150x2/current"
    if "$out/bin/tt-model" status --json >status-dangling.json; then
      echo "tt-model accepted a dangling operator symlink" >&2
      exit 1
    fi
    grep -F '"state": "dangling"' status-dangling.json
    grep -F '/nix/store/does-not-exist-qwen38-0.1.0' status-dangling.json
    "$out/bin/tt-model" status >status-dangling.txt || true
    grep -F 'dangling symlink to /nix/store/does-not-exist-qwen38-0.1.0' status-dangling.txt

    # A record whose package has no GC root is unrooted, and fails.
    rm "$HOME/.local/state/qwen38-p150x2/current"
    mkdir -p "$TMPDIR/fake-rwkv7-model-checkpoint"
    cat >"$TT_MODEL_STATE_ROOT/active.json" <<JSON
    {
      "schema": "onix.tt-model.record.v1",
      "profile": "metalium-rs-rwkv7",
      "package": {"path": "$TMPDIR/fake-rwkv7-model-checkpoint"},
      "runtime": {"path": "$TMPDIR"},
      "checkpoint": {"path": "$TMPDIR"},
      "recorded_at": "2026-10-26T00:00:00+0000"
    }
    JSON
    if "$out/bin/tt-model" status --json >status-unrooted.json; then
      echo "tt-model accepted an unrooted record" >&2
      exit 1
    fi
    grep -F '"verdict": "broken"' status-unrooted.json
    grep -F 'record GC root is missing' status-unrooted.json

    # A rooted record is the managed verdict, and the operator symlink must agree.
    ln -s "$TMPDIR/fake-rwkv7-model-checkpoint" \
      "$TT_MODEL_GC_ROOTS/metalium-rs-rwkv7"
    "$out/bin/tt-model" status --json >status-managed.json
    grep -F '"verdict": "managed"' status-managed.json
    ln -s "$TMPDIR/other-rwkv7-model-checkpoint" \
      "$HOME/.local/state/qwen38-p150x2/current"
    mkdir -p "$TMPDIR/other-rwkv7-model-checkpoint"
    if "$out/bin/tt-model" status --json >status-disagree.json; then
      echo "tt-model accepted a record and an operator symlink that disagree" >&2
      exit 1
    fi
    grep -F 'resolves to' status-disagree.json
    rm "$HOME/.local/state/qwen38-p150x2/current" "$TT_MODEL_GC_ROOTS/metalium-rs-rwkv7"

    # An unknown profile in the record is an error, not a guess.
    python3 - "$TT_MODEL_STATE_ROOT/active.json" <<'PY'
    import json
    import sys

    path = sys.argv[1]
    record = json.load(open(path))
    record["profile"] = "qwen38-not-in-the-registry"
    json.dump(record, open(path, "w"))
    PY
    if "$out/bin/tt-model" status --json >status-unknown.json; then
      echo "tt-model accepted an unknown record profile" >&2
      exit 1
    fi
    grep -F 'record names unknown profile' status-unknown.json

    # A mismatched registry is a usage error, never a guess.
    python3 - "$out/share/tt-model/runtime-profiles.json" "$TMPDIR/bad-schema.json" <<'PY'
    import json
    import sys

    document = json.load(open(sys.argv[1]))
    document["schema_version"] = 2
    json.dump(document, open(sys.argv[2], "w"))
    PY
    if TT_MODEL_REGISTRY="$TMPDIR/bad-schema.json" "$out/bin/tt-model" status \
        >bad-schema.log 2>&1; then
      echo "tt-model accepted a registry with the wrong schema version" >&2
      exit 1
    else
      test "$?" -eq 2
    fi
    grep -F 'schema_version must be 1' bad-schema.log

    python3 - "$out/share/tt-model/runtime-profiles.json" "$TMPDIR/duplicate.json" <<'PY'
    import json
    import sys

    document = json.load(open(sys.argv[1]))
    document["profiles"].append(document["profiles"][0])
    json.dump(document, open(sys.argv[2], "w"))
    PY
    if TT_MODEL_REGISTRY="$TMPDIR/duplicate.json" "$out/bin/tt-model" status \
        >duplicate.log 2>&1; then
      echo "tt-model accepted a registry with a duplicate profile" >&2
      exit 1
    else
      test "$?" -eq 2
    fi
    grep -F 'duplicate profile name' duplicate.log

    runHook postInstallCheck
  '';

  meta = {
    description = "Read-only status for the Tenstorrent runtime-profile registry";
    license = lib.licenses.lgpl21Only;
    mainProgram = "tt-model";
    platforms = [ "x86_64-linux" ];
  };
}
