{
  lib,
  rustPlatform,
  runCommand,
  fetchurl,
  nickel,
  jq,
  cacert,
  python312Packages,
  docker,
  b3sum,
  psmisc,
  tt-smi,
  tt-metal,
  coreutils,
  gnugrep,
  gnused,
}:
let
  # r[impl tenstorrent.training_platform.runtime_bundle]
  # r[impl tenstorrent.training_platform.configuration]
  # r[impl tenstorrent.training_platform.preflight]
  # r[impl tenstorrent.training_platform.launch]
  # r[impl tenstorrent.training_platform.evidence]
  core = rustPlatform.buildRustPackage {
    pname = "training-platform-core";
    version = "0.1.0";
    src = ./core;
    cargoLock.lockFile = ./core/Cargo.lock;
    doCheck = true;
    meta = {
      description = "Pure dual-P150A TT-XLA training-platform admission core";
      license = lib.licenses.lgpl21Only;
      mainProgram = "training-platform-core";
      platforms = lib.platforms.linux;
    };
  };

  descriptorPath = "${tt-metal}/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto";

  # r[impl tenstorrent.training_platform.runtime_bundle]
  # Exact TT-Blacksmith source revision from the admitted manifest.
  ttBlacksmithSource = fetchurl {
    name = "tt-blacksmith-6e44259c7f4342709ae7977cc3715cf5e2e19dac.tar.gz";
    url = "https://github.com/tenstorrent/tt-blacksmith/archive/6e44259c7f4342709ae7977cc3715cf5e2e19dac.tar.gz";
    hash = "sha256-svIdu7OzpJsx4fvhWSRT9p9/wp0p0/JVd3SZrAXcqkA=";
  };

  manifestJson = runCommand "training-platform-manifest.json" { nativeBuildInputs = [ nickel ]; } ''
    cp ${./manifest.ncl} manifest.ncl
    cp ${./contracts.ncl} contracts.ncl
    nickel export --format json manifest.ncl > "$out"
  '';

  configJson = runCommand "training-platform-config.json" { nativeBuildInputs = [ nickel ]; } ''
    cp ${./platform.ncl} platform.ncl
    cp ${./contracts.ncl} contracts.ncl
    nickel export --format json platform.ncl > "$out"
  '';

  policyJson =
    runCommand "training-platform-admission-policy.json" { nativeBuildInputs = [ nickel ]; }
      ''
        cp ${./admission-policy.ncl} admission-policy.ncl
        cp ${./manifest.ncl} manifest.ncl
        cp ${./platform.ncl} platform.ncl
        cp ${./contracts.ncl} contracts.ncl
        nickel export --format json admission-policy.ncl > "$out"
      '';

  # r[impl tenstorrent.training_platform.runtime_bundle]
  # Network access is restricted to this fixed-output build. Hardware runs
  # consume only the resulting read-only, hash-locked wheelhouse.
  wheelhouse =
    runCommand "training-platform-wheelhouse"
      {
        nativeBuildInputs = [ python312Packages.pip ];
        SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
        outputHashAlgo = "sha256";
        outputHashMode = "recursive";
        outputHash = "sha256-bTo6Sa93TXHT/Gbr2FpvUU15CqWWwLePU6BZ/jgBJRU=";
      }
      ''
        mkdir -p "$out"
        export HOME="$TMPDIR/home"
        mkdir -p "$HOME"
        pip download \
          --disable-pip-version-check \
          --no-cache-dir \
          --require-hashes \
          --no-deps \
          --only-binary=:all: \
          --dest "$out" \
          --requirement ${./wheelhouse-fetch.lock}
        cp ${./requirements.lock} "$out/requirements.lock"
      '';

  authorityJson =
    runCommand "training-platform-runtime-authority.json"
      {
        nativeBuildInputs = [
          jq
          b3sum
        ];
      }
      ''
        descriptor_blake3=$(b3sum ${descriptorPath} | cut -d' ' -f1)
        requirements_blake3=$(b3sum ${wheelhouse}/requirements.lock | cut -d' ' -f1)
        jq \
          --arg descriptor_path ${lib.escapeShellArg descriptorPath} \
          --arg descriptor_blake3 "$descriptor_blake3" \
          --arg wheelhouse_path ${lib.escapeShellArg "${wheelhouse}"} \
          --arg requirements_path ${lib.escapeShellArg "${wheelhouse}/requirements.lock"} \
          --arg requirements_blake3 "$requirements_blake3" \
          '. + {
            descriptor: {path: $descriptor_path, blake3: $descriptor_blake3},
            wheelhouse: {
              path: $wheelhouse_path,
              requirements_path: $requirements_path,
              requirements_blake3: $requirements_blake3
            }
          }' ${policyJson} > "$out"
      '';

  ownershipProbe = runCommand "training-platform-ownership-probe" { } ''
    cp ${./ownership-probe.sh} "$out"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  launcher = runCommand "training-platform-launcher" { } ''
    substitute ${./launcher.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by ownership_probe "${ownershipProbe}" \
      --subst-var-by authority "${authorityJson}" \
      --subst-var-by smoke_py "${./smoke.py}" \
      --subst-var-by container_smoke "${containerSmoke}"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  containerSmoke = runCommand "training-platform-container-smoke" { } ''
    cp ${./container-smoke.sh} "$out"
    chmod +x "$out"
  '';

  smoke = runCommand "training-platform-smoke" { } ''
    substitute ${./run-smoke.sh} "$out" \
      --subst-var-by launcher "${launcher}"
    chmod +x "$out"
    patchShebangs "$out"
  '';

  tools = runCommand "training-platform-tools" { } ''
    mkdir -p "$out/bin"
    ln -s ${launcher} "$out/bin/training-platform-launch"
    ln -s ${smoke} "$out/bin/training-platform-smoke"
    mkdir -p "$out/share/training-platform"
    ln -s ${manifestJson} "$out/share/training-platform/manifest.json"
    ln -s ${configJson} "$out/share/training-platform/platform.json"
    ln -s ${policyJson} "$out/share/training-platform/admission-policy.json"
    ln -s ${authorityJson} "$out/share/training-platform/runtime-authority.json"
    ln -s ${wheelhouse} "$out/share/training-platform/wheelhouse"
    ln -s ${descriptorPath} "$out/share/training-platform/p150_x2_mesh_graph_descriptor.textproto"
    ln -s ${ttBlacksmithSource} "$out/share/training-platform/tt-blacksmith.tar.gz"
  '';
in
# r[verify tenstorrent.training_platform.verification]
runCommand "training-platform-check"
  {
    nativeBuildInputs = [
      nickel
      jq
      b3sum
      coreutils
      gnugrep
      gnused
      python312Packages.pip
    ];
    CORE = "${core}/bin/training-platform-core";
    passthru = {
      inherit
        core
        tools
        manifestJson
        configJson
        policyJson
        authorityJson
        wheelhouse
        descriptorPath
        ttBlacksmithSource
        launcher
        ownershipProbe
        smoke
        ;
    };
  }
  ''
        set -euo pipefail
        cp -r ${./.} "$PWD/source"
        chmod -R u+w "$PWD/source"
        cd "$PWD/source"

        # Positive configuration admission: exports are deterministic.
        nickel export --format json manifest.ncl > manifest-first.json
        nickel export --format json manifest.ncl > manifest-second.json
        cmp manifest-first.json manifest-second.json
        nickel export --format json platform.ncl > platform-first.json
        nickel export --format json platform.ncl > platform-second.json
        cmp platform-first.json platform-second.json
        grep -F '"bundle_id": "tt-xla-training-2026-08-04"' manifest-first.json
        grep -F '"name": "p150_x2"' platform-first.json

        # Negative Nickel fixtures must fail with contract violations.
        for fixture in fixtures/invalid-*.ncl; do
          if nickel export --format json "$fixture" > /dev/null 2> fixture.log; then
            echo "negative fixture exported: $fixture" >&2
            exit 1
          fi
          grep -F "contract" fixture.log
        done

        # Positive core and package-owned authority admission.
        "$CORE" self-test | grep -F "self-test: PASS"
        "$CORE" admit-bundle manifest-first.json > bundle.json
        grep -F '"status":"admitted"' bundle.json
        "$CORE" validate-config platform-first.json manifest-first.json > validated.json
        grep -F '"status":"admitted"' validated.json
        "$CORE" authorized-launch-plan ${authorityJson} > plan.json
        grep -F '"status":"admitted"' plan.json
        grep -F '"runtime_authority_identity":"' plan.json
        grep -F 'ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04@sha256:8dcad76548126bb33c13ce554192cb9f27612fa3766b7c7e7ae6f591f2ab6503' \
          plan.json
        grep -F '${wheelhouse}' plan.json
        grep -F 'PYTHONPATH' plan.json

        # The ownership probe accepts empty KMD PID files and rejects busy or
        # unverifiable devices before Docker starts.
        ownership_fixture="$PWD/ownership-fixture"
        mkdir -p \
          "$ownership_fixture/devices" \
          "$ownership_fixture/proc/0" \
          "$ownership_fixture/proc/1"
        touch \
          "$ownership_fixture/devices/0" \
          "$ownership_fixture/devices/1" \
          "$ownership_fixture/proc/0/pids" \
          "$ownership_fixture/proc/1/pids" \
          "$ownership_fixture/hugepages"
        cat > "$ownership_fixture/fuser-free" <<'EOF'
    #!/usr/bin/env bash
    exit 1
    EOF
        chmod +x "$ownership_fixture/fuser-free"
        ${ownershipProbe} \
          "$ownership_fixture/devices" \
          "$ownership_fixture/proc" \
          "$ownership_fixture/fuser-free" \
          "$ownership_fixture/hugepages"

        busy_fixture_pid=1234
        printf '%s\n' "$busy_fixture_pid" > "$ownership_fixture/proc/1/pids"
        if ${ownershipProbe} \
          "$ownership_fixture/devices" \
          "$ownership_fixture/proc" \
          "$ownership_fixture/fuser-free" \
          "$ownership_fixture/hugepages"; then
          echo "ownership probe accepted a KMD-reported device owner" >&2
          exit 1
        fi
        : > "$ownership_fixture/proc/1/pids"
        rm "$ownership_fixture/proc/1/pids"
        if ${ownershipProbe} \
          "$ownership_fixture/devices" \
          "$ownership_fixture/proc" \
          "$ownership_fixture/fuser-free" \
          "$ownership_fixture/hugepages"; then
          echo "ownership probe accepted a missing KMD PID file" >&2
          exit 1
        fi

        # Positive preflight admission excludes stable board identities.
        "$CORE" admit-authorized-preflight fixtures/positive/preflight-snapshot.json \
          ${authorityJson} > preflight.json
        grep -F '"status":"admitted"' preflight.json
        grep -F '"device_count":2' preflight.json
        if grep -F 'board_id' preflight.json; then
          echo "portable preflight admission contains a board identity" >&2
          exit 1
        fi

        # The complete runtime closure resolves without an index.
        pip install --dry-run --ignore-installed --break-system-packages --no-index --no-deps \
          --find-links ${wheelhouse} --require-hashes \
          --requirement ${wheelhouse}/requirements.lock >offline-install.log
        grep -F 'Would install' offline-install.log
        if grep -E 'https?://|Looking in indexes' offline-install.log; then
          echo "offline install attempted network resolution" >&2
          exit 1
        fi

        # Runtime container scripts contain no package-index or Pip-upgrade path.
        if grep -E 'extra-index-url|--upgrade pip|pip download|https?://' \
          container-smoke.sh \
          ${../rwkv7-lora-p150a/container-probe.sh} \
          ${../rwkv7-lora-p150a/container-full-model.sh} \
          ${../rwkv7-lora-dual-p150a/container-collective.sh} \
          ${../rwkv7-lora-dual-p150a/container-dual-model.sh}; then
          echo "hardware container script retains a network package path" >&2
          exit 1
        fi

        # Negative preflight fixtures must fail with stable codes.
        check_rejected() {
          fixture="$1"; code="$2"
          if "$CORE" admit-preflight "$fixture" manifest-first.json platform-first.json \
            > /dev/null 2> rejected.log; then
            echo "negative preflight fixture admitted: $fixture" >&2
            exit 1
          fi
          grep -F "$code" rejected.log
        }
        check_rejected fixtures/negative/snapshot-one-device.json "device_count"
        check_rejected fixtures/negative/snapshot-busy-device.json "busy_device"
        check_rejected fixtures/negative/snapshot-old-firmware.json "old_firmware"
        check_rejected fixtures/negative/snapshot-no-hugepages.json "missing_hugepages"
        check_rejected fixtures/negative/snapshot-wrong-board.json "wrong_architecture"
        check_rejected fixtures/negative/snapshot-driver-mismatch.json "driver_mismatch"
        check_rejected fixtures/negative/snapshot-duplicate-device.json "duplicate_device"

        # Positive receipt admission built from live identities.
        BUNDLE_ID=$(jq -r '.identity_blake3' bundle.json)
        CONFIG_ID=$(jq -r '.identity_blake3' validated.json)
        PLAN_ID=$(jq -r '.identity_blake3' plan.json)
        jq -n \
          --arg bundle "$BUNDLE_ID" --arg config "$CONFIG_ID" --arg plan "$PLAN_ID" \
          '{
            schema_version: 1,
            kind: "dual_p150a_platform_smoke",
            bundle_identity: $bundle,
            config_identity: $config,
            launch_plan_identity: $plan,
            stages: [
              {name: "health_before", status: "pass"},
              {name: "preflight", status: "pass"},
              {name: "mesh_open", status: "pass"},
              {name: "smoke_execute", status: "pass"},
              {name: "mesh_close", status: "pass"},
              {name: "restore", status: "pass"},
              {name: "health_after", status: "pass"}
            ],
            artifacts: {smoke_output: "2c6f51c4f3e2a5f6d7a7e38fdde0b63f0e2be3d4566f6c0c38139cb2c46d2ab5"},
            non_claims: [
              "no training correctness claim",
              "no gradient parity claim",
              "no convergence claim",
              "no speedup claim",
              "no thermal endurance claim"
            ]
          }' > receipt-valid.json
        "$CORE" validate-terminal-receipt receipt-valid.json ${authorityJson} \
          > receipt-admission.json
        grep -F '"outcome":"success"' receipt-admission.json

        # A complete failure receipt validates but cannot satisfy success admission.
        jq '.stages = [
          {name: "health_before", status: "pass"},
          {name: "preflight", status: "fail"},
          {name: "mesh_open", status: "not_observed"},
          {name: "smoke_execute", status: "not_observed"},
          {name: "mesh_close", status: "not_observed"},
          {name: "restore", status: "not_observed"},
          {name: "health_after", status: "not_observed"}
        ] | .artifacts = {}' receipt-valid.json > receipt-failure.json
        "$CORE" validate-terminal-receipt receipt-failure.json ${authorityJson} \
          > receipt-failure-admission.json
        grep -F '"outcome":"failure"' receipt-failure-admission.json

        # Negative receipt fixtures must fail with stable codes.
        jq '.bundle_identity = "0000000000000000000000000000000000000000000000000000000000000000"' \
          receipt-valid.json > receipt-stale.json
        if "$CORE" validate-terminal-receipt receipt-stale.json ${authorityJson} \
          > /dev/null 2> rejected.log; then
          echo "stale receipt identity admitted" >&2
          exit 1
        fi
        grep -F "stale_identity" rejected.log

        jq '.stages = [{name: "health_before", status: "pass"}]' \
          receipt-valid.json > receipt-missing-stage.json
        if "$CORE" validate-terminal-receipt receipt-missing-stage.json ${authorityJson} \
          > /dev/null 2> rejected.log; then
          echo "receipt with missing stages admitted" >&2
          exit 1
        fi
        grep -F "missing_stage" rejected.log

        jq '.artifacts.device_serial_number = "2c6f51c4f3e2a5f6d7a7e38fdde0b63f0e2be3d4566f6c0c38139cb2c46d2ab5"' \
          receipt-valid.json > receipt-serial.json
        if "$CORE" validate-terminal-receipt receipt-serial.json ${authorityJson} \
          > /dev/null 2> rejected.log; then
          echo "receipt with serial number admitted" >&2
          exit 1
        fi
        grep -F "unsafe_field" rejected.log

        jq '.non_claims = ["no speedup claim"]' receipt-valid.json > receipt-overclaim.json
        if "$CORE" validate-terminal-receipt receipt-overclaim.json ${authorityJson} \
          > /dev/null 2> rejected.log; then
          echo "overclaiming receipt admitted" >&2
          exit 1
        fi
        grep -F "overclaiming" rejected.log

        # Current hardware evidence validates under the package-owned authority.
        "$CORE" validate-terminal-receipt \
          evidence/2026-10-28T181500Z-dual-p150a-smoke/receipt.json ${authorityJson} \
          > current-hardware-admission.json
        grep -F '"outcome":"success"' current-hardware-admission.json
        for artifact in evidence/2026-10-28T181500Z-dual-p150a-smoke/*.json; do
          "$CORE" validate-portable-artifact "$artifact" > /dev/null
        done
        test "$(b3sum evidence/2026-10-28T181500Z-dual-p150a-smoke/preflight-admission.json | cut -d' ' -f1)" = \
          "$(jq -r '.artifacts["preflight-admission.json"]' evidence/2026-10-28T181500Z-dual-p150a-smoke/receipt.json)"
        test "$(b3sum evidence/2026-10-28T181500Z-dual-p150a-smoke/workload-summary.json | cut -d' ' -f1)" = \
          "$(jq -r '.artifacts["workload-summary.json"]' evidence/2026-10-28T181500Z-dual-p150a-smoke/receipt.json)"

        # Historical evidence is retained only as a sanitized disclosure. It
        # cannot satisfy current platform admission after the driver correction.
        grep -F '"status": "superseded"' \
          evidence/2026-08-04-dual-p150a-smoke/receipt.json
        if "$CORE" validate-terminal-receipt \
          evidence/2026-08-04-dual-p150a-smoke/receipt.json ${authorityJson} \
          > /dev/null 2> rejected.log; then
          echo "superseded historical receipt was admitted" >&2
          exit 1
        fi
        grep -F 'unknown_field' rejected.log
        for artifact in evidence/2026-08-04-dual-p150a-smoke/*.json; do
          "$CORE" validate-portable-artifact "$artifact" > /dev/null
        done
        jq -n '{board_id: "0000040331921011"}' > unsafe-artifact.json
        if "$CORE" validate-portable-artifact unsafe-artifact.json \
          > /dev/null 2> rejected.log; then
          echo "portable artifact admitted a stable board identity" >&2
          exit 1
        fi
        grep -F 'unsafe_field' rejected.log

        # Production launchers carry only package-owned platform authority.
        grep -F '${authorityJson}' ${launcher}
        grep -F 'authorized-launch-plan' ${launcher}
        grep -F 'validate-portable-artifact "$artifact"' ${launcher}
        grep -F '2> "$PRIVATE_ROOT/preflight-error.log"' ${launcher}
        grep -F 'if ! mkdir "$EVIDENCE_ROOT"' ${launcher}
        grep -F '/proc/driver/tenstorrent' ${launcher}
        grep -F '${ownershipProbe}' ${launcher}
        if grep -E -- '--manifest|--config|--descriptor|--wheels-dir' ${launcher}; then
          echo "production launcher accepts caller-controlled platform authority" >&2
          exit 1
        fi

        if ${launcher} --evidence-root "$PWD/outside-authority" -- true \
          > /dev/null 2> rejected.log; then
          echo "production launcher accepted an unapproved evidence root" >&2
          exit 1
        fi
        grep -F 'evidence root is outside package authority' rejected.log

        mkdir -p "$out"
        cp manifest-first.json "$out/manifest.json"
        cp platform-first.json "$out/platform.json"
        cp plan.json "$out/launch-plan.json"
        cp preflight.json "$out/preflight-admission.json"
        cp receipt-valid.json "$out/receipt-valid.json"
  ''
