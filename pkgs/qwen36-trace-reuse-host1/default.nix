{
  lib,
  runCommand,
  makeWrapper,
  nickel,
  qwen36,
}:
let
  program = "qwen36-p150x2-trace-reuse-host1-admission";
in
runCommand "qwen36-trace-reuse-host1"
  {
    nativeBuildInputs = [
      makeWrapper
      nickel
    ];
    meta = {
      description = "Opt-in Qwen trace reuse with a bounded host thread pool";
      mainProgram = program;
      platforms = [ "x86_64-linux" ];
      license = lib.licenses.asl20;
    };
  }
  ''
    mkdir -p "$out/bin" "$out/share/qwen36-trace-reuse-host1"
    nickel export --format json ${./profile.ncl} \
      > "$out/share/qwen36-trace-reuse-host1/profile.json"
    hostThreads=$(nickel export --format json --field host_threads ${./profile.ncl})
    test "$hostThreads" = 1
    makeWrapper ${qwen36}/bin/qwen36-p150x2-trace-reuse-admission "$out/bin/${program}" \
      --set OMP_NUM_THREADS "$hostThreads" \
      --set MKL_NUM_THREADS "$hostThreads" \
      --set OPENBLAS_NUM_THREADS "$hostThreads" \
      --set TT_METAL_DISABLE_SFPLOADMACRO "1" \
      --unset TT_METAL_SLOW_DISPATCH_MODE \
      --unset TT_METAL_SIMULATOR_SOC_DESC

    # Both checks stop at argument parsing, before a device opens.
    OMP_NUM_THREADS=invalid MKL_NUM_THREADS=invalid OPENBLAS_NUM_THREADS=invalid \
      "$out/bin/${program}" --help > help.log 2> help-errors.log
    grep -F -- '--trace-reuse-admission' help.log
    if grep -i 'invalid' help-errors.log; then
      echo "the wrapper did not replace the inherited thread budget" >&2
      exit 1
    fi
    if "$out/bin/${program}" --model-path /nonexistent-qwen-model --unknown-host-budget-option > invalid.log 2>&1; then
      echo "the diagnostic command accepted an unknown option" >&2
      exit 1
    fi
    grep -F 'unrecognized arguments: --unknown-host-budget-option' invalid.log
    if "$out/bin/${program}" --model-path /nonexistent-qwen-model --decode-mode eager > wrong-mode.log 2>&1; then
      echo "the diagnostic command accepted eager mode" >&2
      exit 1
    fi
    grep -F -- '--trace-reuse-admission requires candidate decode mode' wrong-mode.log
  ''
