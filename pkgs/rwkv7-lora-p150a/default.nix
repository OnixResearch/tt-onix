{
  lib,
  rustPlatform,
  runCommand,
  nickel,
  jq,
  b3sum,
  coreutils,
  gnugrep,
  docker,
  psmisc,
  tt-smi,
  trainingPlatform,
  platformEvidenceReceipt,
  cpuReference,
}:
let
  # r[impl tenstorrent.rwkv7_p150a_finetuning.configuration]
  # r[impl tenstorrent.rwkv7_p150a_finetuning.authority]
  core = rustPlatform.buildRustPackage {
    pname = "rwkv7-lora-p150a-core";
    version = "0.1.0";
    src = ./core;
    cargoLock.lockFile = ./core/Cargo.lock;
    doCheck = true;
    meta = {
      description = "Pure single-P150A RWKV-7 LoRA bring-up admission core";
      license = lib.licenses.mit;
      mainProgram = "rwkv7-lora-p150a-core";
      platforms = lib.platforms.linux;
    };
  };

  platformCore = "${trainingPlatform.passthru.core}/bin/training-platform-core";
  platformManifest = trainingPlatform.passthru.manifestJson;
  platformConfig = trainingPlatform.passthru.configJson;
  platformDescriptor = trainingPlatform.passthru.descriptorPath;

  profileJson = runCommand "rwkv7-lora-p150a-profile.json" { nativeBuildInputs = [ nickel ]; } ''
    cp ${./contracts.ncl} contracts.ncl
    cp ${./profile.ncl} profile.ncl
    mkdir fixtures
    cp ${./fixtures/base-profile.ncl} fixtures/base-profile.ncl
    nickel export --format json profile.ncl > "$out"
  '';

  # r[impl tenstorrent.rwkv7_p150a_finetuning.device]
  probeRunner = runCommand "rwkv7-lora-p150a-probe" { } ''
    substitute ${./run-probe.sh} "$out" \
      --subst-var-by core "${core}" \
      --subst-var-by platform_core "${trainingPlatform.passthru.core}" \
      --subst-var-by jq "${jq}/bin/jq" \
      --subst-var-by docker "${docker}/bin/docker" \
      --subst-var-by ttsmi "${tt-smi}/bin/tt-smi" \
      --subst-var-by b3sum "${b3sum}/bin/b3sum" \
      --subst-var-by fuser "${psmisc}/bin/fuser" \
      --subst-var-by probe_py "${./probes/wkv7_probe.py}" \
      --subst-var-by container_probe "${./container-probe.sh}"
    chmod +x "$out"
  '';

  tools = runCommand "rwkv7-lora-p150a-tools" { } ''
    mkdir -p "$out/bin" "$out/share/rwkv7-lora-p150a"
    ln -s ${core}/bin/rwkv7-lora-p150a-core "$out/bin/rwkv7-lora-p150a-core"
    ln -s ${probeRunner} "$out/bin/rwkv7-lora-p150a-probe"
    ln -s ${profileJson} "$out/share/rwkv7-lora-p150a/profile.json"
  '';
in
# r[verify tenstorrent.rwkv7_p150a_finetuning.configuration.valid]
# r[verify tenstorrent.rwkv7_p150a_finetuning.authority.valid]
# r[verify tenstorrent.rwkv7_p150a_finetuning.authority.invalid]
runCommand "rwkv7-lora-p150a-check"
  {
    nativeBuildInputs = [
      nickel
      jq
      b3sum
      coreutils
      gnugrep
    ];
    CORE = "${core}/bin/rwkv7-lora-p150a-core";
    PLATFORM_CORE = platformCore;
    PROBE_RUNNER = "${probeRunner}";
    PLATFORM_MANIFEST = "${platformManifest}";
    PLATFORM_CONFIG = "${platformConfig}";
    PLATFORM_DESCRIPTOR = "${platformDescriptor}";
    PLATFORM_RECEIPT = "${platformEvidenceReceipt}";
    CPU_RECEIPT = "${cpuReference}/receipt.json";
    passthru = {
      inherit core tools profileJson;
    };
  }
  ''
    set -euo pipefail
    cp -r ${./.} "$PWD/source"
    chmod -R u+w "$PWD/source"
    cd "$PWD/source"

    # Valid profile exports deterministically and carries a stable identity.
    nickel export --format json profile.ncl >profile-first.json
    nickel export --format json profile.ncl >profile-second.json
    cmp profile-first.json profile-second.json
    first_identity=$(b3sum profile-first.json | cut -d' ' -f1)
    second_identity=$(b3sum profile-second.json | cut -d' ' -f1)
    test "$first_identity" = "$second_identity"
    grep -F '"profile_id": "rwkv7-lora-p150a-single"' profile-first.json
    grep -F '"bundle_id": "tt-xla-training-2026-08-04"' profile-first.json

    # Negative Nickel fixtures must fail with contract violations.
    # Nickel splits the blame message across stdout and stderr, so the
    # check captures both streams before it searches for the contract
    # violation marker.
    for fixture in fixtures/invalid-*.ncl; do
      if nickel export --format json "$fixture" >fixture.out 2>fixture.err; then
        echo "negative profile fixture exported: $fixture" >&2
        exit 1
      fi
      cat fixture.out fixture.err >fixture.log
      grep -F "contract" fixture.log
    done

    # Core self-test exercises built-in positive and negative fixtures.
    "$CORE" self-test | grep -F "self-test: PASS"

    # Core re-checks every Nickel contract before admitting the profile.
    "$CORE" admit-profile profile-first.json >profile-admission.json
    grep -F '"status":"admitted"' profile-admission.json
    grep -F '"identity_blake3":' profile-admission.json

    # The platform receipt must re-validate against the admitted platform
    # identities before accelerator admission can bind it.
    "$PLATFORM_CORE" validate-receipt "$PLATFORM_RECEIPT" "$PLATFORM_MANIFEST" \
      "$PLATFORM_CONFIG" "$PLATFORM_DESCRIPTOR" >platform-admission.json
    grep -F '"status":"admitted"' platform-admission.json

    # Prerequisite admission binds platform and CPU receipts to the profile.
    cpu_receipt_blake3=$(b3sum "$CPU_RECEIPT" | cut -d' ' -f1)
    "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      >prerequisite-admission.json
    grep -F '"status":"admitted"' prerequisite-admission.json
    grep -F '"prerequisite_identity":' prerequisite-admission.json
    "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      >prerequisite-admission-again.json
    cmp prerequisite-admission.json prerequisite-admission-again.json

    # Negative: a stale platform admission blocks compilation and device work.
    jq '.status = "rejected"' platform-admission.json >stale-admission.json
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission stale-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      > /dev/null 2> rejected.log; then
      echo "stale platform admission was admitted" >&2
      exit 1
    fi
    grep -F "stale_prerequisite" rejected.log

    # Negative: a failing platform stage blocks prerequisite admission.
    jq '.stages[2].status = "fail"' "$PLATFORM_RECEIPT" >failing-receipt.json
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt failing-receipt.json \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      > /dev/null 2> rejected.log; then
      echo "failing platform stage was admitted" >&2
      exit 1
    fi
    grep -F "failing_stage" rejected.log

    # Negative: a drifted model identity breaks the CPU authority binding.
    jq '.model.revision = "ffffffffffffffffffffffffffffffffffffffff"' \
      profile-first.json >drifted-profile.json
    if "$CORE" admit-prerequisites \
      --profile drifted-profile.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "$cpu_receipt_blake3" \
      > /dev/null 2> rejected.log; then
      echo "drifted model revision was admitted" >&2
      exit 1
    fi
    grep -F "model_mismatch" rejected.log

    # The probe runner shell must exist and reject missing arguments even
    # though hardware runs happen outside the sandbox.
    test -x "$PROBE_RUNNER"
    if "$PROBE_RUNNER" > /dev/null 2> runner-usage.log; then
      echo "probe runner accepted an empty invocation" >&2
      exit 1
    fi
    grep -F "usage:" runner-usage.log

    # Positive probe receipt admission against the live profile identity.
    profile_identity=$(jq -r '.identity_blake3' profile-admission.json)
    jq --arg identity "$profile_identity" '.profile_identity = $identity' \
      fixtures/probe-receipt-base.json >probe-receipt.json
    "$CORE" admit-probe probe-receipt.json profile-first.json >probe-admission.json
    grep -F '"status":"admitted"' probe-admission.json

    # Negative: host placement is rejected as forbidden fallback.
    jq '.probe_result.device.type = "CPU"' probe-receipt.json >probe-host.json
    if "$CORE" admit-probe probe-host.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "host placement probe was admitted" >&2
      exit 1
    fi
    grep -F "host_fallback" rejected.log

    # Negative: a tolerance failure is rejected.
    jq '.probe_result.tolerances_satisfied = false' probe-receipt.json >probe-tolerance.json
    if "$CORE" admit-probe probe-tolerance.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "tolerance-failing probe was admitted" >&2
      exit 1
    fi
    grep -F "tolerance_failure" rejected.log

    # Negative: a stale probe graph is rejected.
    jq '.profile_identity = "0000000000000000000000000000000000000000000000000000000000000000"' \
      probe-receipt.json >probe-stale.json
    if "$CORE" admit-probe probe-stale.json profile-first.json \
      > /dev/null 2> rejected.log; then
      echo "stale probe receipt was admitted" >&2
      exit 1
    fi
    grep -F "stale_probe_receipt" rejected.log

    # Negative: a malformed CPU receipt identity blocks admission.
    if "$CORE" admit-prerequisites \
      --profile profile-first.json \
      --platform-admission platform-admission.json \
      --platform-receipt "$PLATFORM_RECEIPT" \
      --cpu-receipt "$CPU_RECEIPT" \
      --cpu-receipt-blake3 "not-a-hash" \
      > /dev/null 2> rejected.log; then
      echo "malformed CPU receipt identity was admitted" >&2
      exit 1
    fi
    grep -F "malformed_receipt" rejected.log

    mkdir -p "$out"
    cp profile-first.json "$out/profile.json"
    cp profile-admission.json "$out/profile-admission.json"
    cp platform-admission.json "$out/platform-admission.json"
    cp prerequisite-admission.json "$out/prerequisite-admission.json"
    cp probe-admission.json "$out/probe-admission-fixture.json"
  ''
