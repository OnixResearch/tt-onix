{
  b3sum,
  evidencePackage,
  lib,
  runCommand,
}:
let
  fixture = builtins.path {
    path = ./fixtures/terminal-attempt-3;
    name = "rwkv7-p150x2-terminal-attempt-3-fixture";
  };
  terminalSchemaVersion = 1;
  falseSuccess = builtins.toJSON {
    diagnostic = "multi-owner restoration health window expired; rollback timer remains armed";
    outcome = "verified_success";
    schema_version = terminalSchemaVersion;
  };
in
# r[verify tenstorrent.rwkv7_p150x2.attempt3_retention.recorded]
# r[verify tenstorrent.rwkv7_p150x2.attempt3_retention.false_success]
runCommand "rwkv7-p150x2-terminal-attempt-3-evidence"
  {
    nativeBuildInputs = [ b3sum ];
  }
  ''
    set -eu
    (cd ${fixture} && b3sum --check package-files.blake3)

    classifier=${lib.getExe' evidencePackage "rwkv7-p150x2-attempt3"}
    "$classifier" --fixture-root ${fixture} > classification.json
    grep -F '"outcome": "terminal_firmware_configuration_failure"' classification.json
    grep -F '"opened_device_ids": [' classification.json
    grep -F '"profiler_enabled": true' classification.json
    grep -F '"watcher_enabled": true' classification.json
    grep -F '"brisc_firmware_overflow": true' classification.json
    grep -F '"model_operation_verified": false' classification.json
    grep -F '"fresh_hardware_success": false' classification.json

    cp -R ${fixture} changed-fixture
    chmod -R u+w changed-fixture
    printf '%s\n' ${lib.escapeShellArg falseSuccess} > changed-fixture/terminal-outcome.json
    if "$classifier" --fixture-root "$PWD/changed-fixture" > false-success.json 2>&1; then
      echo 'attempt-3 classifier accepted a false success claim' >&2
      exit 1
    fi
    grep -F 'attempt-3 terminal outcome mismatch' false-success.json

    cp classification.json "$out"
  ''
