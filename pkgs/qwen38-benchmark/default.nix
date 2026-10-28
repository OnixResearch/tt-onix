{
  lib,
  stdenvNoCC,
  python3,
  makeWrapper,
}:

stdenvNoCC.mkDerivation {
  pname = "qwen38-benchmark";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ./.;
    filter = path: _type: baseNameOf path != "__pycache__";
  };

  nativeBuildInputs = [ makeWrapper ];
  buildInputs = [ python3 ];

  doCheck = true;
  checkPhase = ''
    runHook preCheck
    ${python3}/bin/python3 qwen38_benchmark.py --self-test
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/libexec/qwen38-benchmark" "$out/share/qwen38-benchmark"
    install -m755 qwen38_benchmark.py "$out/libexec/qwen38-benchmark/qwen38_benchmark.py"
    install -m644 corpus.json "$out/share/qwen38-benchmark/corpus.json"
    makeWrapper ${python3}/bin/python3 "$out/bin/qwen38-benchmark" \
      --add-flags "$out/libexec/qwen38-benchmark/qwen38_benchmark.py" \
      --set QWEN38_BENCH_CORPUS "$out/share/qwen38-benchmark/corpus.json"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/qwen38-benchmark" --self-test
    runHook postInstallCheck
  '';

  meta = {
    description = "Repeatable OpenAI-compatible benchmark for Qwen serving stacks";
    mainProgram = "qwen38-benchmark";
    license = lib.licenses.lgpl21Only;
    platforms = lib.platforms.unix;
  };
}
