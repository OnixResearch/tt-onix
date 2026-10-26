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
  fixtureJob = stdenvNoCC.mkDerivation {
    name = "fixture-job";
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/bin"
      cat >"$out/bin/job-run" <<'SH'
      #!/bin/sh
      receipt=""
      while [ "$#" -gt 0 ]; do
        case "$1" in
          --receipt)
            shift
            receipt="$1"
            ;;
        esac
        shift
      done
      printf 'job env=%s mesh=%s\n' "''${FAKE_JOB_ENV:-unset}" "''${MESH_DEVICE:-unset}"
      if [ -n "$receipt" ]; then
        printf '{"schema":"fake-job-receipt/v1","ok":true}\n' >"$receipt"
      fi
      exit "''${FAKE_JOB_EXIT:-0}"
      SH
      chmod +x "$out/bin/job-run"
      runHook postInstall
    '';
  };
  fixtureJobProfile = {
    schema = "onix.runtime-profile.v1";
    name = "fake-job";
    mode = "job";
    package = {
      attribute = "fixture-job";
      store_name = "-fixture-job";
    };
    entry = {
      binary = "bin/job-run";
      argv_prefix = [ ];
    };
    env = {
      FAKE_JOB_ENV = "1";
    };
    mesh = {
      name = "card-0";
      shape = [
        1
        1
      ];
      physical_device_ids = [ 0 ];
      fabric = "none";
    };
    limits = {
      max_sequence_length = 32;
      max_generation_tokens = 8;
    };
    checkpoint = {
      revision = "d81965cb4e1a9f96696b4f70b84212b8f2e43216";
      authority_kind = "model-target-profile-identity";
      authority_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
    };
    precision = {
      flags = [ ];
    };
    admission = {
      kind = "authority-digest";
    };
    readiness = {
      kind = "exit-code";
    };
    reset = {
      on_release = "never";
      device_paths = [ "/dev/tenstorrent/0" ];
    };
    model_target_name = "rwkv7-goose-world2.8-0.1b-p150x2";
    model_target_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
    non_claims = [ "fixture only" ];
  };
  fixtureRegistry = writeText "fixture-runtime-profiles.json" (
    builtins.toJSON {
      schema_version = 1;
      profiles = [ fixtureJobProfile ];
    }
  );
  fixtureServePackage =
    name:
    stdenvNoCC.mkDerivation {
      inherit name;
      dontUnpack = true;
      installPhase = ''
        mkdir -p "$out/bin" "$out/libexec/tt-metalium" "$out/share/serve"
        printf '#!/bin/sh\nexit 0\n' >"$out/bin/serve"
        chmod +x "$out/bin/serve"
        ln -s ${fixtureIdentity} "$out/share/serve/execution-identity"
        runHook postInstall
      '';
    };
  fixtureServeA = fixtureServePackage "a-fixture-serve-0.1.0";
  fixtureServeB = fixtureServePackage "b-fixture-serve-0.1.0";
  mkServeProfile =
    {
      name,
      port,
      timeout,
    }:
    {
      schema = "onix.runtime-profile.v1";
      inherit name;
      mode = "serve";
      package = {
        attribute = "fixture-serve";
        store_name = "-fixture-serve-0.1.0";
      };
      entry = {
        binary = "bin/serve";
        argv_prefix = [
          "--model-alias"
          "Fake-Serve"
        ];
      };
      env = { };
      mesh = {
        name = "P150x2";
        shape = [
          1
          2
        ];
        physical_device_ids = [
          0
          1
        ];
        fabric = "FABRIC_1D";
      };
      inherit port;
      limits = {
        max_sequence_length = 32;
        max_generation_tokens = 8;
      };
      checkpoint = {
        revision = "d81965cb4e1a9f96696b4f70b84212b8f2e43216";
        authority_kind = "model-target-profile-identity";
        authority_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
      };
      runtime = {
        root = "libexec/tt-metalium";
        execution_identity = "share/serve/execution-identity";
      };
      precision = {
        flags = [ ];
      };
      admission = {
        kind = "authority-digest";
      };
      readiness = {
        kind = "http-healthz";
        path = "/healthz";
        timeout_seconds = timeout;
      };
      reset = {
        on_release = "required-before-native";
        device_paths = [
          "/dev/tenstorrent/0"
          "/dev/tenstorrent/1"
        ];
      };
      model_target_name = "rwkv7-goose-world2.8-0.1b-p150x2";
      model_target_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
      non_claims = [ "fixture only" ];
    };
  fixtureServeProfile = mkServeProfile {
    name = "fake-serve";
    port = 18080;
    timeout = 5;
  };
  fixtureServeRegistry = writeText "fixture-serve-profiles.json" (
    builtins.toJSON {
      schema_version = 1;
      profiles = [
        (mkServeProfile {
          name = "fake-serve";
          port = 18080;
          timeout = 5;
        })
        (mkServeProfile {
          name = "fake-serve-timeout";
          port = 18081;
          timeout = 2;
        })
        (mkServeProfile {
          name = "fake-serve-identity";
          port = 18082;
          timeout = 5;
        })
        (mkServeProfile {
          name = "fake-serve-rollback";
          port = 18083;
          timeout = 5;
        })
        (mkServeProfile {
          name = "fake-serve-force";
          port = 18084;
          timeout = 5;
        })
      ];
    }
  );
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
      mkdir -p "$out/scripts"
      printf '#!/usr/bin/env python3\nimport json\n\nprint(json.dumps({"passed": False}))\n' \
        >"$out/scripts/fixture-containment.py"
      chmod +x "$out/scripts/fixture-containment.py"
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
  switchLockDriver = writeText "hold-switch-lock.py" ''
    import fcntl
    import os
    import subprocess
    import sys

    state_root, tt_model, *arguments = sys.argv[1:]
    os.makedirs(state_root, exist_ok=True)
    with open(os.path.join(state_root, ".switch.lock"), "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [tt_model, "switch", *arguments], capture_output=True, text=True
        )
    sys.stderr.write(result.stderr)
    expected = "another switch operation holds the lock"
    sys.exit(0 if result.returncode == 1 and expected in result.stderr else 1)
  '';
  runLockDriver = writeText "hold-run-lock.py" ''
    import fcntl
    import os
    import subprocess
    import sys

    state_root, tt_model, *arguments = sys.argv[1:]
    os.makedirs(state_root, exist_ok=True)
    with open(os.path.join(state_root, ".run.lock"), "a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [tt_model, "run", *arguments], capture_output=True, text=True
        )
    sys.stderr.write(result.stderr)
    expected = "another run operation holds the lock"
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
    # The sandbox has no /usr/bin/env, so give the fixture containment command an
    # absolute interpreter. The real repository script keeps its env shebang.
    printf '#!%s\nimport json\n\nprint(json.dumps({"passed": False}))\n' \
      ${python3}/bin/python >"$TT_MODEL_REPO_ROOT/scripts/fixture-containment.py"
    chmod +x "$TT_MODEL_REPO_ROOT/scripts/fixture-containment.py"

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

    # ---- run: job profiles only, through the owner helper -----------------
    run_state="$TMPDIR/runstate"
    mkdir -p "$run_state" "$TMPDIR/emptyhome" "$TMPDIR/emptyprofiles"
    cat >"$TMPDIR/fake-owner-ok" <<'SH'
    #!/bin/sh
    case "$1" in
      cards)
        echo 'owner_name=qwen38-p150x2.service'
        echo 'owner_pid=4242'
        echo 'holder /dev/tenstorrent/0 4242 qwen38-p150x2.service qwen38-p150x2-serve'
        ;;
      stop-owner) echo stopped ;;
      reset) echo reset ;;
      *) exit 2 ;;
    esac
    SH
    cat >"$TMPDIR/fake-owner-foreign" <<'SH'
    #!/bin/sh
    case "$1" in
      cards)
        echo 'owner_name=qwen38-p150x2.service'
        echo 'owner_pid=none'
        echo 'holder /dev/tenstorrent/1 777 other.service rogue'
        ;;
      *) exit 2 ;;
    esac
    SH
    cat >"$TMPDIR/fake-owner-fail" <<'SH'
    #!/bin/sh
    exit 3
    SH
    chmod +x "$TMPDIR/fake-owner-ok" "$TMPDIR/fake-owner-foreign" "$TMPDIR/fake-owner-fail"

    expect_refusal "run of a serve profile" "is a serve profile" \
      env TT_MODEL_STATE_ROOT="$run_state" TT_MODEL_OWNER="$TMPDIR/fake-owner-ok" \
      "$tt_model" run qwen38 --package ${fixtureQwen} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "run with a structural failure, even forced" "run refused: check:" \
      env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/fake-owner-ok" "$tt_model" run fake-job \
      --package ${fixtureOther} --repo-root "$TT_MODEL_REPO_ROOT" --force
    expect_refusal "run without the owner helper" "the owner helper is unavailable" \
      env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/absent-helper" "$tt_model" run fake-job \
      --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "run with a foreign card holder" "run refused: card-holder" \
      env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/fake-owner-foreign" "$tt_model" run fake-job \
      --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "run when the helper fails" "cards failed with status 3" \
      env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/fake-owner-fail" "$tt_model" run fake-job \
      --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT"

    python3 ${patchJson} ${fixtureRegistry} "$TMPDIR/bad-job-registry.json" --profile fake-job \
      'model_target_blake3="0000000000000000000000000000000000000000000000000000000000000000"'
    expect_refusal "run with an admission failure" "run refused: check:" \
      env TT_MODEL_REGISTRY="$TMPDIR/bad-job-registry.json" TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/fake-owner-ok" "$tt_model" run fake-job \
      --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT"
    expect_refusal "run --force when unattended" "refused when TT_MODEL_UNATTENDED=1" \
      env TT_MODEL_REGISTRY="$TMPDIR/bad-job-registry.json" TT_MODEL_UNATTENDED=1 \
      TT_MODEL_STATE_ROOT="$run_state" TT_MODEL_OWNER="$TMPDIR/fake-owner-ok" \
      "$tt_model" run fake-job --package ${fixtureJob} \
      --repo-root "$TT_MODEL_REPO_ROOT" --force
    env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$run_state" \
      python3 ${runLockDriver} "$run_state" "$tt_model" fake-job \
      --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT"

    # A forced admission bypass still runs, and records the bypass by exit status.
    env TT_MODEL_REGISTRY="$TMPDIR/bad-job-registry.json" TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/fake-owner-ok" "$tt_model" run fake-job \
      --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT" --force \
      --json -- --receipt "$TMPDIR/forced-receipt.json" >run-forced.json 2>run-forced.err
    grep -F '"exit_status": 0' run-forced.json
    test -e "$TMPDIR/forced-receipt.json"

    # A clean run: owner stopped, reset performed, receipt copied, env isolated.
    rm -f "$run_state/last-run.json"
    env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$run_state" \
      TT_MODEL_OWNER="$TMPDIR/fake-owner-ok" MESH_DEVICE=leaked FAKE_JOB_ENV=caller \
      "$tt_model" run fake-job --package ${fixtureJob} --repo-root "$TT_MODEL_REPO_ROOT" \
      --json -- --receipt "$TMPDIR/job-receipt.json" >run.json 2>run.err
    grep -F 'job env=1 mesh=unset' run.err
    grep -F '"leader"' run.json || true
    python3 - "$run_state" "$TMPDIR" <<'PY'
    import glob
    import json
    import os
    import sys

    state, tmp = sys.argv[1], sys.argv[2]
    event = json.load(open(os.path.join(state, "last-run.json")))
    assert event["schema"] == "onix.tt-model.run.v1", event["schema"]
    assert event["profile"] == "fake-job"
    assert event["package"].endswith("-fixture-job"), event["package"]
    assert event["exit_status"] == 0, event["exit_status"]
    assert event["owner_stopped"] is True
    assert event["reset_performed"] is True
    assert event["argv"][0].endswith("bin/job-run"), event["argv"]
    assert event["argv_blake3"]
    assert event["receipt"]["path"].endswith("job-receipt.json"), event["receipt"]
    assert len(event["receipt"]["blake3"]) == 64, event["receipt"]
    assert event["started"] and event["finished"]
    assert not os.path.exists(os.path.join(state, "active.json"))
    assert len(glob.glob(os.path.join(state, "runs", "*.json"))) == 2
    PY

    # status reports the run event without treating it as the owner
    env TT_MODEL_STATE_ROOT="$run_state" TT_MODEL_HOME="$TMPDIR/emptyhome" \
      TT_MODEL_PROFILE_ROOT="$TMPDIR/emptyprofiles" TT_MODEL_SYSTEMCTL=none \
      "$tt_model" status --json >run-status.json
    python3 - "$run_state" <<'PY'
    import json
    import sys

    report = json.load(open("run-status.json"))
    assert report["active"]["state"] == "absent", report["active"]
    assert report["last_run"]["state"] == "ok", report["last_run"]
    assert report["last_run"]["profile"] == "fake-job", report["last_run"]
    assert report["last_run"]["exit_status"] == 0, report["last_run"]
    assert report["verdict"] in ("none", "degraded"), report["verdict"]
    PY
    env TT_MODEL_STATE_ROOT="$run_state" TT_MODEL_HOME="$TMPDIR/emptyhome" \
      TT_MODEL_PROFILE_ROOT="$TMPDIR/emptyprofiles" TT_MODEL_SYSTEMCTL=none \
      "$tt_model" status >run-status.txt
    grep -F 'last run: fake-job' run-status.txt
    grep -F 'active record: absent' run-status.txt

    # ---- switch: serve profiles only, through the owner helper ------------
    # Each scenario uses its own port and a stub that shuts itself down, so no
    # scenario can inherit a previous listener or depend on a kill landing.
    cat >"$TMPDIR/stub.py" <<'PY'
    import http.server
    import sys
    import threading

    port = int(sys.argv[1])
    body = open(sys.argv[2], "rb").read()
    lifetime = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0


    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/healthz"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *arguments):
            pass


    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Timer(lifetime, server.shutdown).start()
    server.serve_forever()
    PY
    cat >"$TMPDIR/fake-switch-owner" <<'SH'
    #!/bin/sh
    case "$1" in
      cards)
        echo 'owner_name=qwen38-p150x2.service'
        if [ -n "$OWNER_RUNNING" ]; then echo 'owner_pid=4242'; else echo 'owner_pid=none'; fi
        if [ -n "$FOREIGN" ]; then echo 'holder /dev/tenstorrent/1 777 other.service rogue'; fi
        ;;
      stop-owner) echo 'stop-owner' >>"$HELPER_LOG" ;;
      reset) echo 'reset' >>"$HELPER_LOG" ;;
      start-owner)
        echo 'start-owner' >>"$HELPER_LOG"
        if [ -n "$START_FAIL" ]; then echo 'start failed' >&2; exit 1; fi
        if [ -n "$NO_START" ]; then exit 0; fi
        python3 "$STUB" "$PORT" "$STUB_BODY" 20 </dev/null >/dev/null 2>&1 &
        exit 0
        ;;
      *) exit 2 ;;
    esac
    SH
    chmod +x "$TMPDIR/fake-switch-owner"
    export HELPER_LOG="$TMPDIR/helper.log"
    fixture_identity=${fixtureIdentity}

    free_port() {
      python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
    }
    PORT_OK=$(free_port)
    PORT_IDENTITY=$(free_port)
    PORT_ROLLBACK=$(free_port)
    PORT_TIMEOUT=$(free_port)
    PORT_FORCE=$(free_port)
    python3 ${patchJson} ${fixtureServeRegistry} "$TMPDIR/serve-registry-1.json" \
      --profile fake-serve "port=$PORT_OK"
    python3 ${patchJson} "$TMPDIR/serve-registry-1.json" "$TMPDIR/serve-registry-2.json" \
      --profile fake-serve-identity "port=$PORT_IDENTITY"
    python3 ${patchJson} "$TMPDIR/serve-registry-2.json" "$TMPDIR/serve-registry-3.json" \
      --profile fake-serve-rollback "port=$PORT_ROLLBACK"
    python3 ${patchJson} "$TMPDIR/serve-registry-3.json" "$TMPDIR/serve-registry-4.json" \
      --profile fake-serve-timeout "port=$PORT_TIMEOUT"
    python3 ${patchJson} "$TMPDIR/serve-registry-4.json" "$TMPDIR/serve-registry.json" \
      --profile fake-serve-force "port=$PORT_FORCE"
    serve_registry="$TMPDIR/serve-registry.json"

    python3 - "$TMPDIR/body-ok.json" "$fixture_identity" <<'PY'
    import json
    import sys

    body = {
        "ready": True,
        "status": "ok",
        "model": "Fake-Serve",
        "model_identity": "Fake-Serve:d81965cb4e1a9f96696b4f70b84212b8f2e43216",
        "package_identity": sys.argv[2],
        "runtime_identity": "/nix/store/fake-runtime-identity",
        "profile_id": "fake-serve-v1",
        "physical_pcie_device_ids": [0, 1],
        "simulated": False,
    }
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(body, handle)
    wrong = dict(body)
    wrong["package_identity"] = "/nix/store/00000000000000000000000000000000-other"
    with open(sys.argv[1].replace("body-ok", "body-wrong"), "w", encoding="utf-8") as handle:
        json.dump(wrong, handle)
    PY

    expect_in() {
      local file="$1" pattern="$2"
      if ! grep -F -- "$pattern" "$file"; then
        echo "--- $file ---" >&2
        cat "$file" >&2
        fail "expected '$pattern' in $file"
      fi
    }

    switch_to() {
      local root="$1" profile="$2" package="$3" port="$4" body="$5"
      shift 5
      env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$root" \
        TT_MODEL_GC_ROOTS="$root/gcroots" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
        STUB="$TMPDIR/stub.py" STUB_BODY="$body" PORT="$port" \
        HELPER_LOG="$HELPER_LOG" "$@" \
        "$tt_model" switch "$profile" --package "$package" \
        --repo-root "$TT_MODEL_REPO_ROOT" --json
    }

    # S1: a successful switch commits the record, readiness, and identity.
    switch_root="$TMPDIR/switch-a"
    mkdir -p "$switch_root"
    switch_to "$switch_root" fake-serve ${fixtureServeA} "$PORT_OK" \
      "$TMPDIR/body-ok.json" >switch-ok.json 2>switch-ok.err
    expect_in switch-ok.json '"profile": "fake-serve"'
    expect_in switch-ok.json '"reset_performed": true'
    expect_in switch-ok.json '"observed_seconds"'
    expect_in switch-ok.json '"simulated": false'
    python3 - "$switch_root" "$fixture_identity" <<'PY'
    import json
    import sys

    state, identity = sys.argv[1], sys.argv[2]
    record = json.load(open(f"{state}/active.json"))
    assert record["schema"] == "onix.tt-model.record.v1"
    assert record["profile"] == "fake-serve"
    assert record["identity"]["package_identity"] == identity, record["identity"]
    assert record["identity"]["physical_pcie_device_ids"] == [0, 1]
    assert record["identity"]["runtime_identity"] == "/nix/store/fake-runtime-identity"
    assert record["identity"]["simulated"] is False
    assert record["readiness"]["observed_seconds"] > 0
    assert record["mesh"]["physical_device_ids"] == [0, 1]
    assert record["admission_bypassed"] is False
    PY
    cp "$switch_root/active.json" "$TMPDIR/previous-record.json"

    # S2: a wrong package identity rolls back and leaves the previous record intact.
    env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$switch_root" \
      TT_MODEL_GC_ROOTS="$switch_root/gcroots" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      STUB="$TMPDIR/stub.py" STUB_BODY="$TMPDIR/body-wrong.json" PORT="$PORT_IDENTITY" \
      "$tt_model" switch fake-serve-identity --package ${fixtureServeB} \
      --repo-root "$TT_MODEL_REPO_ROOT" >switch-mismatch.out 2>switch-mismatch.err || true
    expect_in switch-mismatch.err 'switch refused: identity'
    expect_in switch-mismatch.err 'package-identity'
    cmp "$TMPDIR/previous-record.json" "$switch_root/active.json"
    expect_in "$switch_root/switch-failure.json" '"restored": true'

    # S3: a start failure rolls back to no active record.
    start_root="$TMPDIR/switch-start-fail"
    mkdir -p "$start_root"
    env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$start_root" \
      TT_MODEL_GC_ROOTS="$start_root/gcroots" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      START_FAIL=1 "$tt_model" switch fake-serve-rollback --package ${fixtureServeA} \
      --repo-root "$TT_MODEL_REPO_ROOT" >start-fail.out 2>start-fail.err || true
    expect_in start-fail.err 'switch refused: start-owner'
    test ! -e "$start_root/active.json"
    expect_in "$start_root/switch-failure.json" '"restored": true'

    # S4: when the previous owner cannot be restarted, the record is withdrawn.
    rollback_root="$TMPDIR/switch-rollback-fail"
    mkdir -p "$rollback_root"
    cp "$TMPDIR/previous-record.json" "$rollback_root/active.json"
    env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$rollback_root" \
      TT_MODEL_GC_ROOTS="$rollback_root/gcroots" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      START_FAIL=1 OWNER_RUNNING=1 HELPER_LOG="$HELPER_LOG" \
      "$tt_model" switch fake-serve-rollback --package ${fixtureServeB} \
      --repo-root "$TT_MODEL_REPO_ROOT" >rollback-fail.out 2>rollback-fail.err || true
    expect_in rollback-fail.err 'rollback FAILED'
    test ! -e "$rollback_root/active.json"
    expect_in "$rollback_root/switch-failure.json" '"restored": false'
    expect_in "$rollback_root/switch-failure.json" '"rollback_error"'
    test -n "$(ls "$rollback_root"/history/*.rollback-failed.json 2>/dev/null)"

    # S5: readiness timeout rolls back.
    timeout_root="$TMPDIR/switch-timeout"
    mkdir -p "$timeout_root"
    env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$timeout_root" \
      TT_MODEL_GC_ROOTS="$timeout_root/gcroots" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      NO_START=1 "$tt_model" switch fake-serve-timeout --package ${fixtureServeA} \
      --repo-root "$TT_MODEL_REPO_ROOT" >timeout.out 2>timeout.err || true
    expect_in timeout.err 'switch refused: readiness'
    expect_in timeout.err 'not ready within 2s'
    test ! -e "$timeout_root/active.json"

    # S6: --force records an admission bypass.
    python3 ${patchJson} "$serve_registry" "$TMPDIR/serve-bad-target.json" \
      --profile fake-serve \
      'model_target_blake3="0000000000000000000000000000000000000000000000000000000000000000"'
    force_root="$TMPDIR/switch-force"
    mkdir -p "$force_root"
    env TT_MODEL_REGISTRY="$TMPDIR/serve-bad-target.json" TT_MODEL_STATE_ROOT="$force_root" \
      TT_MODEL_GC_ROOTS="$force_root/gcroots" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      STUB="$TMPDIR/stub.py" STUB_BODY="$TMPDIR/body-ok.json" PORT="$PORT_FORCE" \
      "$tt_model" switch fake-serve --package ${fixtureServeA} \
      --repo-root "$TT_MODEL_REPO_ROOT" --force --json >force.json 2>force.err
    expect_in force.json '"admission_bypassed": true'
    expect_in "$force_root/active.json" '"admission_bypassed": true'

    # S7: an unattended force is refused.
    expect_refusal "switch --force when unattended" "refused when TT_MODEL_UNATTENDED=1" \
      env TT_MODEL_REGISTRY="$TMPDIR/serve-bad-target.json" TT_MODEL_UNATTENDED=1 \
      TT_MODEL_STATE_ROOT="$TMPDIR/switch-unattended" TT_MODEL_GC_ROOTS="$TMPDIR/gc-unattended" \
      TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" "$tt_model" switch fake-serve \
      --package ${fixtureServeA} --repo-root "$TT_MODEL_REPO_ROOT" --force

    # S8: a structural failure is refused even with --force.
    expect_refusal "switch with a structural failure" "switch refused: check:" \
      env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$TMPDIR/switch-structural" \
      TT_MODEL_GC_ROOTS="$TMPDIR/gc-structural" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      "$tt_model" switch fake-serve --package ${fixtureOther} \
      --repo-root "$TT_MODEL_REPO_ROOT" --force

    # S9: a foreign card holder is refused before any stop.
    rm -f "$HELPER_LOG"
    expect_refusal "switch with a foreign card holder" "switch refused: card-holder" \
      env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_STATE_ROOT="$TMPDIR/switch-foreign" \
      TT_MODEL_GC_ROOTS="$TMPDIR/gc-foreign" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      FOREIGN=1 HELPER_LOG="$HELPER_LOG" "$tt_model" switch fake-serve \
      --package ${fixtureServeA} --repo-root "$TT_MODEL_REPO_ROOT"
    if [ -f "$HELPER_LOG" ] && grep -q 'stop-owner' "$HELPER_LOG"; then
      fail "the helper stopped the owner while a foreign holder was present"
    fi

    # S10: a job profile points at run.
    expect_refusal "switch of a job profile" "is a job profile" \
      env TT_MODEL_REGISTRY=${fixtureRegistry} TT_MODEL_STATE_ROOT="$TMPDIR/switch-job" \
      TT_MODEL_GC_ROOTS="$TMPDIR/gc-job" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      "$tt_model" switch fake-job --package ${fixtureJob} \
      --repo-root "$TT_MODEL_REPO_ROOT"

    # S11: a held switch lock is refused.
    env TT_MODEL_REGISTRY="$serve_registry" TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" \
      TT_MODEL_STATE_ROOT="$TMPDIR/switch-locked" \
      TT_MODEL_GC_ROOTS="$TMPDIR/switch-locked/gcroots" \
      python3 ${switchLockDriver} "$TMPDIR/switch-locked" "$tt_model" fake-serve \
      --package ${fixtureServeA} --repo-root "$TT_MODEL_REPO_ROOT" \
      || fail "the switch lock was not enforced"

    # S12: containment is never bypassed by --force.
    python3 ${patchJson} "$serve_registry" "$TMPDIR/serve-containment.json" \
      --profile fake-serve \
      'containment={"command":"scripts/fixture-containment.py","arguments":["@package@"],"verdict_field":"passed","verdict_value":true}'
    expect_refusal "switch with a failing containment" "switch refused: containment" \
      env TT_MODEL_REGISTRY="$TMPDIR/serve-containment.json" \
      TT_MODEL_STATE_ROOT="$TMPDIR/switch-containment" TT_MODEL_GC_ROOTS="$TMPDIR/gc-containment" \
      TT_MODEL_OWNER="$TMPDIR/fake-switch-owner" "$tt_model" switch fake-serve \
      --package ${fixtureServeA} --repo-root "$TT_MODEL_REPO_ROOT" --force

    runHook postInstallCheck
  '';

  meta = {
    description = "Status, validation, and the active record for the Tenstorrent runtime-profile registry";
    license = lib.licenses.lgpl21Only;
    mainProgram = "tt-model";
    platforms = [ "x86_64-linux" ];
  };
}
