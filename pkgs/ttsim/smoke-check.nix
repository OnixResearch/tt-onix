{
  runCommand,
  jq,
  ttsim-blackhole-smoke,
  ttsim-runtime-validator,
}:
runCommand "ttsim-blackhole-single-chip-smoke"
  {
    nativeBuildInputs = [
      jq
      ttsim-blackhole-smoke
      ttsim-runtime-validator
    ];
  }
  ''
    set -euo pipefail
    mkdir -p "$out"
    ttsim-blackhole-smoke --receipt "$out/receipt.json"
    ttsim-runtime-validator validate "$out/receipt.json"
    jq -e '.simulated == true' "$out/receipt.json" >/dev/null
    jq -e '.status == "passed"' "$out/receipt.json" >/dev/null
    jq -e '.observations.device_opened == true' "$out/receipt.json" >/dev/null
    jq -e '.observations.firmware_compiled == true' "$out/receipt.json" >/dev/null
    jq -e '.observations.representative_operations_completed == true' \
      "$out/receipt.json" >/dev/null
    jq -e '.observations.clean_close == true' "$out/receipt.json" >/dev/null
  ''
