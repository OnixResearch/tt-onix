{
  runCommand,
  coreutils,
  jq,
  python3,
  tt-metal,
  ttsim,
  ttsim-blackhole-smoke,
}:
let
  smokeTimeoutSeconds = 3600;
  killGraceSeconds = 30;
  python = ttsim-blackhole-smoke.python;
in
runCommand "ttsim-gdn-fp32-reference-smoke"
  {
    nativeBuildInputs = [
      coreutils
      jq
    ];
  }
  ''
    set -euo pipefail
    mkdir -p "$out" runtime
    for entry in ${tt-metal}/libexec/tt-metalium/*; do
      ln -s "$entry" "runtime/$(basename "$entry")"
    done
    mkdir -p runtime/build runtime/generated runtime/home
    export TT_METAL_HOME="$PWD/runtime"
    export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
    export HOME="$TT_METAL_HOME/home"
    export XDG_CACHE_HOME="$HOME/.cache"
    export TT_METAL_PROFILER_DIR="$TT_METAL_HOME/generated/profiler"
    export PYTHONPATH=${tt-metal}/${python3.sitePackages}
    cd runtime

    ${python}/bin/python ${./gdn-smoke.py} --host-only > "$out/host-checks.log" 2>&1
    ${ttsim}/bin/ttsim-blackhole-sfpu \
      timeout --signal=TERM --kill-after=${toString killGraceSeconds}s \
      ${toString smokeTimeoutSeconds} \
      ${python}/bin/python ${./gdn-smoke.py} --result "$out/observations.json" \
      > "$out/device-checks.log" 2>&1 || {
        cat "$out/device-checks.log"
        exit 1
      }
    jq --arg metal '${tt-metal}' --arg simulator '${ttsim}' \
      --arg metal_version '${tt-metal.version}' --arg simulator_version '${ttsim.version}' \
      '. + {
        simulated: true,
        tt_metal: $metal,
        ttsim: $simulator,
        tt_metal_version: $metal_version,
        simulator_version: $simulator_version,
        non_claims: ["hardware execution", "performance", "full-model correctness", "trace capture"]
      }' "$out/observations.json" > "$out/result.json"
    jq -e '.simulated == true and .status == "passed" and .post_rejection_retry == "passed"' \
      "$out/result.json" >/dev/null
    cat "$out/result.json"
  ''
