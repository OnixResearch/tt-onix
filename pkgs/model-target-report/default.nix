{
  lib,
  rustPlatform,
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
    mkdir -p "$out/share/model-target-report"
    cp ${./fixtures/complete-synthetic-candidate.json} \
      "$out/share/model-target-report/complete-synthetic-candidate.json"
    cp ${./fixtures/qwen3.6-unreviewed.json} \
      "$out/share/model-target-report/qwen3.6-unreviewed.json"
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

    "$out/bin/model-target-report" report \
      "$out/share/model-target-report/qwen3.6-unreviewed.json" \
      >unsupported.json
    grep -F '"status": "unsupported"' unsupported.json
    grep -F '"simulator_change_eligible": false' unsupported.json
    grep -F '"runnable_output_exposed": false' unsupported.json
    grep -F 'missing reviewed candidate fact: upstream_identity' unsupported.json

    if "$out/bin/model-target-report" report /dev/null >invalid.log 2>&1; then
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
