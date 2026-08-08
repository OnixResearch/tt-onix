{
  lib,
  rustPlatform,
  targetProfile,
}:
let
  packageSource = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ./.
      ../model-target-core
    ];
  };
  sourceRootPath = "source/model-target-report";
in
rustPlatform.buildRustPackage {
  pname = "model-target-report";
  version = "0.1.0";

  src = packageSource;
  sourceRoot = "source/model-target-report";
  cargoLock.lockFile = ./Cargo.lock;

  postInstall = ''
    mkdir -p "$out/share/model-target-report" "$out/share/model-target"
    echo '${targetProfile.profile_identity}' >"$out/share/model-target/profile-blake3"
    cp ${./fixtures/complete-synthetic-candidate.json} \
      "$out/share/model-target-report/complete-synthetic-candidate.json"
    cp ${./fixtures/qwen3.6-unreviewed.json} \
      "$out/share/model-target-report/qwen3.6-unreviewed.json"
    cp ${./fixtures/qwen3.6-reviewed.json} \
      "$out/share/model-target-report/qwen3.6-reviewed.json"
    cp ${./receipts/qwen3.6-candidate-verdict.json} \
      "$out/share/model-target-report/qwen3.6-candidate-verdict.json"
  '';

  doCheck = true;
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    set -euo pipefail

    "$out/bin/model-target-report" --self-test
    "$out/bin/model-target-report" report \
      "$out/share/model-target-report/complete-synthetic-candidate.json" \
      >complete.json
    grep -F '"status": "eligible"' complete.json
    grep -F '"simulator_change_eligible": true' complete.json
    grep -F '"runnable_output_exposed": false' complete.json

    # r[verify tenstorrent.model_targets.qwen3_6_candidate_report.complete]
    # r[verify tenstorrent.model_targets.qwen3_6_candidate_report.machine_facts]
    "$out/bin/model-target-report" report \
      "$out/share/model-target-report/qwen3.6-reviewed.json" \
      >reviewed.json
    cmp reviewed.json "$out/share/model-target-report/qwen3.6-candidate-verdict.json"
    grep -F '"status": "eligible"' reviewed.json
    grep -F '"simulator_change_eligible": true' reviewed.json
    grep -F '"runnable_output_exposed": false' reviewed.json
    grep -F 'Qwen bring-up tt-metal base' reviewed.json
    grep -F 'custom ttl Gated DeltaNet kernels' reviewed.json
    grep -F 'metadata-only scope excludes the 26 checkpoint shards' reviewed.json

    # r[verify tenstorrent.model_targets.qwen3_6_candidate_report.incomplete]
    "$out/bin/model-target-report" report \
      "$out/share/model-target-report/qwen3.6-unreviewed.json" \
      >unsupported.json
    grep -F '"status": "unsupported"' unsupported.json
    grep -F '"simulator_change_eligible": false' unsupported.json
    grep -F '"runnable_output_exposed": false' unsupported.json
    grep -F 'missing reviewed candidate fact: upstream_identity' unsupported.json

    printf '{' >malformed-candidate.json
    if "$out/bin/model-target-report" report malformed-candidate.json >invalid.log 2>&1; then
      echo "candidate report accepted malformed JSON" >&2
      exit 1
    fi
    grep -F 'invalid candidate facts' invalid.log

    if grep -R -E '/dev/tenstorrent|MeshDevice|TT_VISIBLE_DEVICES|owner-control' \
      ${./src} ../model-target-core/src; then
      echo "candidate report contains a hardware access surface" >&2
      exit 1
    fi

    runHook postInstallCheck
  '';

  meta = {
    description = "Deterministic device-free model candidate report";
    license = lib.licenses.mit;
    mainProgram = "model-target-report";
    platforms = lib.platforms.linux;
  };
}
