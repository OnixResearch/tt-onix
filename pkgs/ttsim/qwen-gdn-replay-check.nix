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
  # Qwen3.8-27B GDN at TP=2: 48 value heads / 2 = 24 heads per shard,
  # head_dim 128 (4 tiles). This is one representative recurrence primitive,
  # not a full-model replay.
  qwenHeadCount = 24;
  qwenHeadTiles = 4;
  qwenSteps = 3;
  smokeTimeoutSeconds = 3600;
  killGraceSeconds = 30;
  python = ttsim-blackhole-smoke.python;
in
runCommand "ttsim-qwen-gdn-replay"
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

    ${python}/bin/python ${./gdn-smoke.py} \
      --head-count ${toString qwenHeadCount} \
      --head-tiles ${toString qwenHeadTiles} \
      --steps ${toString qwenSteps} \
      --host-only > "$out/host-checks.log" 2>&1
    if ${ttsim}/bin/ttsim-blackhole-sfpu \
      timeout --signal=TERM --kill-after=${toString killGraceSeconds}s \
      ${toString smokeTimeoutSeconds} \
      ${python}/bin/python ${./gdn-smoke.py} \
      --head-count ${toString qwenHeadCount} \
      --head-tiles ${toString qwenHeadTiles} \
      --steps ${toString qwenSteps} \
      --workload-name qwen38-gdn-replay \
      --result "$out/observations.json" > "$out/device-checks.log" 2>&1; then
      jq --arg metal '${tt-metal}' --arg simulator '${ttsim}' \
        --arg metal_version '${tt-metal.version}' --arg simulator_version '${ttsim.version}' \
        '. + {
          simulated: true,
          tt_metal: $metal,
          ttsim: $simulator,
          tt_metal_version: $metal_version,
          simulator_version: $simulator_version,
          non_claims: ((.non_claims // []) + ["hardware execution", "performance", "full-model correctness", "trace capture"])
        }' "$out/observations.json" > "$out/result.json"
      jq -e '
        .simulated == true
        and .status == "passed"
        and .workload == "qwen38-gdn-replay"
        and .head_count == ${toString qwenHeadCount}
        and .head_dim == 128
        and .recurrent_steps == ${toString qwenSteps}
        and .post_rejection_retry == "passed"
      ' "$out/result.json" >/dev/null
    else
      if ! grep -Fq 'ALU_ACC_CTRL_Zero_Flag_disabled_src=1' "$out/device-checks.log" \
        || ! grep -Fq 'not modeled' "$out/device-checks.log"; then
        cat "$out/device-checks.log"
        exit 1
      fi
      jq -n --arg metal '${tt-metal}' --arg simulator '${ttsim}' \
        --arg metal_version '${tt-metal.version}' --arg simulator_version '${ttsim.version}' \
        '{
          schema: "onix.ttsim-qwen-gdn-replay.v1",
          workload: "qwen38-gdn-replay",
          status: "blocked",
          simulated: true,
          head_count: ${toString qwenHeadCount},
          head_dim: 128,
          recurrent_steps: ${toString qwenSteps},
          blocker: "ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled",
          tt_metal: $metal,
          ttsim: $simulator,
          tt_metal_version: $metal_version,
          simulator_version: $simulator_version,
          non_claims: ["hardware execution", "performance", "full-model correctness", "trace capture"]
        }' > "$out/result.json"
      jq -e '
        .simulated == true
        and .status == "blocked"
        and .workload == "qwen38-gdn-replay"
        and .head_count == ${toString qwenHeadCount}
        and (.blocker | contains("ALU_ACC_CTRL_Zero_Flag_disabled_src"))
      ' "$out/result.json" >/dev/null
    fi
    cat "$out/result.json"
  ''
