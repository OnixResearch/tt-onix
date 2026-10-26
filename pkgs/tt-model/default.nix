# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.status]
# r[impl tenstorrent.runtime_profile.check]
# r[impl tenstorrent.runtime_profile.record]
# Read-only status, pure profile validation, and the active-record writer for the
# runtime-profile registry. It ships the registry export beside its own script.
#
# The command is hardware tooling: it must never be wired into the device-free
# quality gate, and its program text must not name a device node, tt-smi, or the
# owner helper. Its install check asserts both.
{
  lib,
  stdenvNoCC,
  b3sum,
  python3,
  makeWrapper,
  writeText,
}:
let
  # Fixture packages. Each store name ends with a registry store-name suffix so
  # the store-name rule and the entry-binary rule can be exercised separately.
  fixtureIdentity = stdenvNoCC.mkDerivation {
    name = "fixture-execution-identity";
    dontUnpack = true;
    installPhase = "mkdir -p $out && runHook postInstall";
  };
  fixtureQwen = stdenvNoCC.mkDerivation {
    name = "fixture-qwen38-0.1.0";
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/bin" "$out/libexec/tt-metalium" "$out/share/qwen36"
      printf '#!/bin/sh\nexit 0\n' >"$out/bin/qwen38-p150x2-serve"
      chmod +x "$out/bin/qwen38-p150x2-serve"
      ln -s ${fixtureIdentity} "$out/share/qwen36/execution-identity"
      runHook postInstall
    '';
  };
  fixtureNanbeige = stdenvNoCC.mkDerivation {
    name = "fixture-nanbeige43-0.1.0";
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/bin" "$out/libexec/tt-metalium"
      printf '#!/bin/sh\nexit 0\n' >"$out/bin/nanbeige43-p150x2"
      chmod +x "$out/bin/nanbeige43-p150x2"
      runHook postInstall
    '';
  };
  fixtureMetalium = stdenvNoCC.mkDerivation {
    name = "fixture-rwkv7-model-checkpoint";
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/bin"
      printf '#!/bin/sh\nexit 0\n' >"$out/bin/rwkv7-model-checkpoint"
      chmod +x "$out/bin/rwkv7-model-checkpoint"
      runHook postInstall
    '';
  };
  fixtureOther = stdenvNoCC.mkDerivation {
    name = "fixture-other-0.1.0";
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/bin" "$out/libexec/tt-metalium"
      printf '#!/bin/sh\nexit 0\n' >"$out/bin/qwen38-p150x2-serve"
      chmod +x "$out/bin/qwen38-p150x2-serve"
      runHook postInstall
    '';
  };
  # The repository files the registry's repository-relative paths resolve to.
  fixtureRepo = stdenvNoCC.mkDerivation {
    name = "fixture-runtime-registry-repo";
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/model-targets/generated" "$out/pkgs/qwen36/evidence/2026-10-21" \
        "$out/pkgs/nanbeige43/generated"
      cp ${../../model-targets/generated/consumer-identities.nix} \
        "$out/model-targets/generated/consumer-identities.nix"
      cp ${../../pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json} \
        "$out/pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json"
      cp ${../../pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json} \
        "$out/pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json"
      cp ${../../pkgs/nanbeige43/generated/authority.json} \
        "$out/pkgs/nanbeige43/generated/authority.json"
      runHook postInstall
    '';
  };
  # Dotted-path JSON patcher, so each fixture states exactly one difference.
  patchJson = writeText "patch-json.py" ''
    import json
    import sys

    base, out, *assignments = sys.argv[1:]
    profile = None
    if assignments and assignments[0] == "--profile":
        profile = assignments[1]
        assignments = assignments[2:]
    document = json.load(open(base, encoding="utf-8"))
    scoped = document
    if profile is not None:
        scoped = next(entry for entry in document["profiles"] if entry["name"] == profile)
    for assignment in assignments:
        path, _, raw = assignment.partition("=")
        value = json.loads(raw)
        parts = path.split(".")
        target = scoped
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        last = parts[-1]
        if isinstance(target, list):
            target[int(last)] = value
        else:
            target[last] = value
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
  '';
  # Holds the record lock while a second writer runs, then checks its refusal.
  lockDriver = writeText "hold-record-lock.py" ''
    import fcntl
    import os
    import subprocess
    import sys

    state_root, tt_model, *arguments = sys.argv[1:]
    os.makedirs(state_root, exist_ok=True)
    with open(os.path.join(state_root, ".record.lock"), "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [tt_model, "record", *arguments], capture_output=True, text=True
        )
    sys.stderr.write(result.stderr)
    expected = "another record operation holds the lock"
    sys.exit(0 if result.returncode == 1 and expected in result.stderr else 1)
  '';
in
stdenvNoCC.mkDerivation {
  pname = "tt-model";
  version = "0.1.0";

  dontUnpack = true;
  # b3sum is a runtime dependency: every recorded digest is verified with it.
  nativeBuildInputs = [
    b3sum
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
      --prefix PATH : ${lib.makeBinPath [ b3sum ]} \
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
    export TT_MODEL_NIX=none
    export TT_MODEL_REPO_ROOT="$TMPDIR/repo"
    mkdir -p "$HOME" "$TT_MODEL_STATE_ROOT" "$TT_MODEL_GC_ROOTS" "$TT_MODEL_PROFILE_ROOT"
    cp -r ${fixtureRepo} "$TT_MODEL_REPO_ROOT"
    chmod -R u+w "$TT_MODEL_REPO_ROOT"

    tt_model="$out/bin/tt-model"
    registry="$out/share/tt-model/runtime-profiles.json"

    fail() { echo "install check: $*" >&2; exit 1; }

    # expect_refusal DESCRIPTION PATTERN COMMAND...
    expect_refusal() {
      local description="$1" pattern="$2"
      shift 2
      if "$@" >expect.out 2>expect.err; then
        fail "$description: the command was accepted"
      fi
      if ! grep -F -- "$pattern" expect.out expect.err; then
        echo "--- stdout ---" >&2
        cat expect.out >&2
        echo "--- stderr ---" >&2
        cat expect.err >&2
        fail "$description: expected message not found: $pattern"
      fi
    }

    # A fake nix that always fails, to exercise the registration refusal.
    printf '#!/bin/sh\nexit 1\n' >"$TMPDIR/fake-nix"
    chmod +x "$TMPDIR/fake-nix"

    # A small checkpoint tree whose name carries the nanbeige revision prefix.
    small_checkpoint="$TMPDIR/ckpt-c1093c501cc6"
    mkdir -p "$small_checkpoint"
    printf 'one\n' >"$small_checkpoint/a"
    printf 'two\n' >"$small_checkpoint/b"
    printf 'three\n' >"$small_checkpoint/c"
    # A small checkpoint for the Qwen profile, so a record can be written in the
    # sandbox without the 73 GB host snapshot.
    qwen_checkpoint="$TMPDIR/ckpt-1d4bf0f2ff60"
    mkdir -p "$qwen_checkpoint"
    printf 'weights\n' >"$qwen_checkpoint/shard"
    python3 ${patchJson} "$registry" "$TMPDIR/qwen-record.json" --profile qwen38 \
      "checkpoint.path=\"$qwen_checkpoint\""

    # ---- shipped registry and program text -------------------------------
    cmp "$registry" ${../../model-targets/generated/runtime-profiles.json}
    if grep -R -E '/dev/tenstorrent|tt-smi|ttwkv7-owner-control' \
        "$out/bin" "$out/share/tt-model/tt_model.py"; then
      fail "program text names hardware tooling"
    fi

    # ---- status: no record, every source readable and absent --------------
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
    TT_MODEL_SYSTEMCTL="$TMPDIR/systemctl" "$tt_model" status --json >status-none.json
    grep -F '"verdict": "none"' status-none.json
    "$tt_model" status >status-degraded.txt
    grep -F 'verdict: degraded' status-degraded.txt

    # ---- check: the real qwen38 profile cannot admit a fixture package ----
    expect_refusal "qwen38 fixture package" "no receipt names this package" \
      "$tt_model" check qwen38 --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT"
    "$tt_model" check qwen38 --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT" --json \
      >check-qwen.json || true
    grep -F '"item": "receipt-package"' check-qwen.json
    grep -F '"verdict": "fail"' check-qwen.json

    # ---- check: nanbeige43 passes the cheap structural path ---------------
    python3 ${patchJson} "$registry" "$TMPDIR/nb.json" --profile nanbeige43 \
      "checkpoint.path=\"$small_checkpoint\""
    TT_MODEL_REGISTRY="$TMPDIR/nb.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT" --json >check-nb.json
    grep -F '"verdict": "pass"' check-nb.json
    grep -F '"name_matches": true' check-nb.json
    grep -F '"method": "store-presence"' check-nb.json

    # ---- check: each structural refusal ----------------------------------
    expect_refusal "package path shape" "not a /nix/store path" \
      "$tt_model" check qwen38 --package "$TMPDIR/not-a-package" --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "package existence" "package is missing" \
      "$tt_model" check qwen38 \
      --package /nix/store/00000000000000000000000000000000-missing-qwen38-0.1.0 \
      --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "package store name" "store name must end with -qwen38-0.1.0" \
      "$tt_model" check qwen38 --package ${fixtureOther} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "package registration" "package is not registered" \
      env TT_MODEL_NIX="$TMPDIR/fake-nix" "$tt_model" check qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "unknown profile" "no such runtime profile" \
      "$tt_model" check qwen38-not-in-the-registry --package ${fixtureQwen} \
      --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "missing candidate package" "pass --package or record one first" \
      "$tt_model" check qwen38 --repo-root "$TT_MODEL_REPO_ROOT"

    # ---- check: admission refusals ---------------------------------------
    python3 ${patchJson} "$registry" "$TMPDIR/bad-hash.json" --profile qwen38 \
      'admission.receipts.0.blake3="0000000000000000000000000000000000000000000000000000000000000000"'
    expect_refusal "receipt digest" "receipt bytes hash to" \
      env TT_MODEL_REGISTRY="$TMPDIR/bad-hash.json" "$tt_model" check qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT"

    cp "$TT_MODEL_REPO_ROOT/pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json" \
      "$TMPDIR/not-admitted.json"
    python3 ${patchJson} "$TMPDIR/not-admitted.json" "$TMPDIR/not-admitted-2.json" \
      "admission.admitted=false"
    not_admitted_hash=$(b3sum "$TMPDIR/not-admitted-2.json" | cut -d' ' -f1)
    python3 ${patchJson} "$registry" "$TMPDIR/not-admitted-registry.json" --profile qwen38 \
      "admission.receipts.0.path=\"$TMPDIR/not-admitted-2.json\"" \
      "admission.receipts.0.blake3=\"$not_admitted_hash\""
    expect_refusal "receipt admission verdict" "receipt no longer reports an admitted verdict" \
      env TT_MODEL_REGISTRY="$TMPDIR/not-admitted-registry.json" "$tt_model" check qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT"

    python3 ${patchJson} "$registry" "$TMPDIR/bad-authority.json" --profile nanbeige43 \
      'checkpoint.authority_blake3="0000000000000000000000000000000000000000000000000000000000000000"'
    expect_refusal "authority digest" "authority document hashes to" \
      env TT_MODEL_REGISTRY="$TMPDIR/bad-authority.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT"

    # ---- check: checkpoint refusals --------------------------------------
    python3 ${patchJson} "$registry" "$TMPDIR/bad-checkpoint.json" --profile nanbeige43 \
      "checkpoint.path=\"$TMPDIR/absent-checkpoint\""
    expect_refusal "checkpoint path" "checkpoint path is missing" \
      env TT_MODEL_REGISTRY="$TMPDIR/bad-checkpoint.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT"

    wrong_checkpoint="$TMPDIR/wrong-name"
    mkdir -p "$wrong_checkpoint"
    python3 ${patchJson} "$registry" "$TMPDIR/wrong-checkpoint.json" --profile nanbeige43 \
      "checkpoint.path=\"$wrong_checkpoint\""
    expect_refusal "checkpoint name" "checkpoint directory name does not carry revision" \
      env TT_MODEL_REGISTRY="$TMPDIR/wrong-checkpoint.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT"

    # ---- check: gate artifact digest -------------------------------------
    gate_hash=$(b3sum ${fixtureMetalium}/bin/rwkv7-model-checkpoint | cut -d' ' -f1)
    python3 ${patchJson} "$registry" "$TMPDIR/gate-ok.json" --profile metalium-rs-rwkv7 \
      'admission.gates.0.artifact="bin/rwkv7-model-checkpoint"' \
      "admission.gates.0.blake3=\"$gate_hash\""
    TT_MODEL_REGISTRY="$TMPDIR/gate-ok.json" "$tt_model" check metalium-rs-rwkv7 \
      --package ${fixtureMetalium} --repo-root "$TT_MODEL_REPO_ROOT" --json >check-gate.json
    grep -F '"verdict": "pass"' check-gate.json
    grep -F "\"artifact\": \"bin/rwkv7-model-checkpoint\"" check-gate.json

    python3 ${patchJson} "$registry" "$TMPDIR/gate-bad.json" --profile metalium-rs-rwkv7 \
      'admission.gates.0.artifact="bin/rwkv7-model-checkpoint"' \
      'admission.gates.0.blake3="0000000000000000000000000000000000000000000000000000000000000000"'
    expect_refusal "gate artifact digest" "gate artifact hashes to" \
      env TT_MODEL_REGISTRY="$TMPDIR/gate-bad.json" "$tt_model" check metalium-rs-rwkv7 \
      --package ${fixtureMetalium} --repo-root "$TT_MODEL_REPO_ROOT"

    # ---- check: deep re-hash is bounded ----------------------------------
    big_checkpoint="$TMPDIR/big-c1093c501cc6"
    mkdir -p "$big_checkpoint"
    truncate -s 2G "$big_checkpoint/shard"
    python3 ${patchJson} "$registry" "$TMPDIR/big-checkpoint.json" --profile nanbeige43 \
      "checkpoint.path=\"$big_checkpoint\""
    expect_refusal "deep size bound" "above the 1073741824-byte bound" \
      env TT_MODEL_REGISTRY="$TMPDIR/big-checkpoint.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT" --deep
    # The same command without --deep never walks the checkpoint.
    TT_MODEL_REGISTRY="$TMPDIR/big-checkpoint.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT" --json >check-shallow.json
    grep -F '"enabled": false' check-shallow.json
    # A small checkpoint can be hashed.
    TT_MODEL_REGISTRY="$TMPDIR/nb.json" "$tt_model" check nanbeige43 \
      --package ${fixtureNanbeige} --repo-root "$TT_MODEL_REPO_ROOT" --deep --json \
      >check-deep.json
    grep -F '"files": 3' check-deep.json
    grep -F '"enabled": true' check-deep.json

    # ---- record refusals --------------------------------------------------
    expect_refusal "record on a failing check" "record refused: check:" \
      env TT_MODEL_REGISTRY="$TMPDIR/qwen-record.json" "$tt_model" record qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "record of a job profile" "the active owner must be a serve profile" \
      "$tt_model" record metalium-rs-rwkv7 --package ${fixtureMetalium} \
      --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "record --force when unattended" "refused when TT_MODEL_UNATTENDED=1" \
      env TT_MODEL_REGISTRY="$TMPDIR/qwen-record.json" TT_MODEL_UNATTENDED=1 \
      "$tt_model" record qwen38 --package ${fixtureQwen} \
      --repo-root "$TT_MODEL_REPO_ROOT" --force
    env TT_MODEL_REGISTRY="$TMPDIR/qwen-record.json" python3 ${lockDriver} \
      "$TT_MODEL_STATE_ROOT" "$tt_model" qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT" --force

    # ---- record: a forced admission writes, and status reports managed ----
    TT_MODEL_REGISTRY="$TMPDIR/qwen-record.json" "$tt_model" record qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT" \
      --force --json >record.json
    grep -F '"admission_bypassed": true' record.json
    test -L "$TT_MODEL_GC_ROOTS/qwen38-package"
    test "$(readlink "$TT_MODEL_GC_ROOTS/qwen38-package")" = ${fixtureQwen}
    test -L "$TT_MODEL_GC_ROOTS/qwen38-execution-identity"
    python3 - "$TT_MODEL_STATE_ROOT" <<'PY'
    import json
    import sys

    record = json.load(open(f"{sys.argv[1]}/active.json"))
    assert record["schema"] == "onix.tt-model.record.v1", record["schema"]
    assert record["profile"] == "qwen38"
    assert record["mode"] == "serve"
    assert record["package"]["execution_identity"], record["package"]
    assert record["argv"][0].endswith("bin/qwen38-p150x2-serve"), record["argv"]
    assert record["argv_blake3"]
    assert record["registry_blake3"]
    assert record["admission_bypassed"] is True
    assert record["previous"] is None
    assert record["port"] == 8000
    assert record["reset"]["on_release"] == "required-before-native"
    assert {receipt["kind"] for receipt in record["receipts"]} == {
        "continuous-slot-admission",
        "prefix-cache-admission",
    }
    assert all(receipt["admitted"] for receipt in record["receipts"])
    assert record["env"]["QWEN35_GDN_DECODE_FUSION"] == "step"
    PY
    "$tt_model" status --json >status-managed.json
    grep -F '"verdict": "managed"' status-managed.json
    grep -F '"profile": "qwen38"' status-managed.json

    # A second record chains to the first.
    TT_MODEL_REGISTRY="$TMPDIR/qwen-record.json" "$tt_model" record qwen38 \
      --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT" \
      --force >record-second.txt
    python3 - "$TT_MODEL_STATE_ROOT" <<'PY'
    import glob
    import json
    import sys

    record = json.load(open(f"{sys.argv[1]}/active.json"))
    assert record["previous"]["profile"] == "qwen38", record["previous"]
    assert record["previous"]["record_blake3"]
    assert len(glob.glob(f"{sys.argv[1]}/history/*.json")) == 2
    PY

    # ---- status refusals on a real record --------------------------------
    cp "$TT_MODEL_STATE_ROOT/active.json" "$TMPDIR/active-backup.json"

    rm "$TT_MODEL_GC_ROOTS/qwen38-package"
    expect_refusal "record without its package root" "record GC root is missing" \
      "$tt_model" status --json
    ln -s ${fixtureQwen} "$TT_MODEL_GC_ROOTS/qwen38-package"

    python3 ${patchJson} "$TMPDIR/active-backup.json" "$TMPDIR/active-missing-package.json" \
      'package.path="/nix/store/00000000000000000000000000000000-qwen38-0.1.0"'
    cp "$TMPDIR/active-missing-package.json" "$TT_MODEL_STATE_ROOT/active.json"
    expect_refusal "record package absent" "record package is missing from the store" \
      "$tt_model" status --json

    python3 ${patchJson} "$TMPDIR/active-backup.json" "$TMPDIR/active-unknown.json" \
      'profile="qwen38-not-in-the-registry"'
    cp "$TMPDIR/active-unknown.json" "$TT_MODEL_STATE_ROOT/active.json"
    expect_refusal "record unknown profile" "record names unknown profile" \
      "$tt_model" status --json

    cp "$TMPDIR/active-backup.json" "$TT_MODEL_STATE_ROOT/active.json"

    # ---- a legacy source disagreeing with the record is a contradiction --
    mkdir -p "$(dirname "$HOME/.local/state/qwen38-p150x2/current")"
    ln -s ${fixtureOther} "$HOME/.local/state/qwen38-p150x2/current"
    expect_refusal "record and operator symlink disagree" "resolves to" \
      "$tt_model" status --json
    rm "$HOME/.local/state/qwen38-p150x2/current"

    runHook postInstallCheck
  '';

  meta = {
    description = "Status, validation, and the active record for the Tenstorrent runtime-profile registry";
    license = lib.licenses.lgpl21Only;
    mainProgram = "tt-model";
    platforms = [ "x86_64-linux" ];
  };
}
