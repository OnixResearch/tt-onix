{
  lib,
  rustPlatform,
  runCommand,
  fetchurl,
  nickel,
  jq,
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
      license = lib.licenses.mit;
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

  launcher = runCommand "training-platform-launcher" { } ''
    substitute ${./launcher.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker"
    chmod +x "$out"
  '';

  containerSmoke = runCommand "training-platform-container-smoke" { } ''
    substitute ${./container-smoke.sh} "$out" \
      --subst-var-by pjrt_version "1.5.0.dev20260804002041" \
      --subst-var-by pjrt_sha256 "ae57e7a2e4f7c1c9533135e14db5a33201caa3ff1aa2278de9a88ebc9d4d3fe1"
    chmod +x "$out"
  '';

  smoke = runCommand "training-platform-smoke" { } ''
    substitute ${./run-smoke.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by smoke_py "${./smoke.py}" \
      --subst-var-by container_smoke "${containerSmoke}"
    chmod +x "$out"
  '';

  tools = runCommand "training-platform-tools" { } ''
    mkdir -p "$out/bin"
    ln -s ${launcher} "$out/bin/training-platform-launch"
    ln -s ${smoke} "$out/bin/training-platform-smoke"
    mkdir -p "$out/share/training-platform"
    ln -s ${manifestJson} "$out/share/training-platform/manifest.json"
    ln -s ${configJson} "$out/share/training-platform/platform.json"
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
      coreutils
      gnugrep
      gnused
    ];
    CORE = "${core}/bin/training-platform-core";
    passthru = {
      inherit
        core
        tools
        manifestJson
        configJson
        descriptorPath
        ttBlacksmithSource
        launcher
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

    # Positive core admission.
    "$CORE" self-test | grep -F "self-test: PASS"
    "$CORE" admit-bundle manifest-first.json > bundle.json
    grep -F '"status":"admitted"' bundle.json
    "$CORE" validate-config platform-first.json manifest-first.json > validated.json
    grep -F '"status":"admitted"' validated.json
    "$CORE" launch-plan manifest-first.json platform-first.json \
      ${descriptorPath} > plan.json
    grep -F '"identity_blake3":"' plan.json
    grep -F 'ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04@sha256:8dcad76548126bb33c13ce554192cb9f27612fa3766b7c7e7ae6f591f2ab6503' \
      plan.json
    grep -F 'PYTHONPATH' plan.json

    # Positive preflight admission.
    "$CORE" admit-preflight fixtures/positive/preflight-snapshot.json \
      manifest-first.json platform-first.json > preflight.json
    grep -F '"status":"admitted"' preflight.json

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
    "$CORE" validate-receipt receipt-valid.json manifest-first.json platform-first.json \
      ${descriptorPath} > receipt-admission.json
    grep -F '"status":"admitted"' receipt-admission.json

    # Negative receipt fixtures must fail with stable codes.
    jq '.bundle_identity = "0000000000000000000000000000000000000000000000000000000000000000"' \
      receipt-valid.json > receipt-stale.json
    if "$CORE" validate-receipt receipt-stale.json manifest-first.json platform-first.json \
      ${descriptorPath} > /dev/null 2> rejected.log; then
      echo "stale receipt identity admitted" >&2
      exit 1
    fi
    grep -F "stale_identity" rejected.log

    jq '.stages = [{name: "health_before", status: "pass"}]' \
      receipt-valid.json > receipt-missing-stage.json
    if "$CORE" validate-receipt receipt-missing-stage.json manifest-first.json \
      platform-first.json ${descriptorPath} > /dev/null 2> rejected.log; then
      echo "receipt with missing stages admitted" >&2
      exit 1
    fi
    grep -F "missing_stage" rejected.log

    jq '.artifacts.device_serial_number = "2c6f51c4f3e2a5f6d7a7e38fdde0b63f0e2be3d4566f6c0c38139cb2c46d2ab5"' \
      receipt-valid.json > receipt-serial.json
    if "$CORE" validate-receipt receipt-serial.json manifest-first.json platform-first.json \
      ${descriptorPath} > /dev/null 2> rejected.log; then
      echo "receipt with serial number admitted" >&2
      exit 1
    fi
    grep -F "unsafe_field" rejected.log

    jq '.non_claims = ["no speedup claim"]' receipt-valid.json > receipt-overclaim.json
    if "$CORE" validate-receipt receipt-overclaim.json manifest-first.json \
      platform-first.json ${descriptorPath} > /dev/null 2> rejected.log; then
      echo "overclaiming receipt admitted" >&2
      exit 1
    fi
    grep -F "overclaiming" rejected.log

    # The checked-in hardware receipt must re-validate against the current
    # admitted identities. A stale receipt fails this gate.
    "$CORE" validate-receipt evidence/2026-08-04-dual-p150a-smoke/receipt.json \
      manifest-first.json platform-first.json ${descriptorPath} \
      > checked-in-admission.json
    grep -F '"status":"admitted"' checked-in-admission.json

    mkdir -p "$out"
    cp manifest-first.json "$out/manifest.json"
    cp platform-first.json "$out/platform.json"
    cp plan.json "$out/launch-plan.json"
    cp preflight.json "$out/preflight-admission.json"
    cp receipt-valid.json "$out/receipt-valid.json"
  ''
