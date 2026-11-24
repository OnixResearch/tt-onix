{
  lib,
  stdenvNoCC,
  fetchFromGitHub,
  python3,
  makeWrapper,
}:

let
  # Pinned compatibility wrapper for the portable serving benchmark migrated to
  # OnixResearch/inference-lab (inference-lab#7).
  # Explicit rollback pin: OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a
  inferenceLabRev = "fd90c12231f6664c64e72c47050d90fd14f968e6";
  inferenceLabSrc = fetchFromGitHub {
    owner = "OnixResearch";
    repo = "inference-lab";
    rev = inferenceLabRev;
    hash = "sha256-3VbKenyiglxNX9MNV35AQrO1ioJdcfM5KMN89jTnVwQ=";
  };
in
stdenvNoCC.mkDerivation {
  pname = "qwen38-benchmark";
  version = "0.1.0";

  src = "${inferenceLabSrc}/pkgs/qwen38-benchmark";

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
    install -m644 agentic_trace.json "$out/share/qwen38-benchmark/agentic_trace.json"
    install -m644 PROVENANCE.json "$out/share/qwen38-benchmark/PROVENANCE.json"
    makeWrapper ${python3}/bin/python3 "$out/bin/qwen38-benchmark" \
      --add-flags "$out/libexec/qwen38-benchmark/qwen38_benchmark.py" \
      --set QWEN38_BENCH_CORPUS "$out/share/qwen38-benchmark/corpus.json" \
      --set QWEN38_BENCH_AGENTIC_TRACE "$out/share/qwen38-benchmark/agentic_trace.json"
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/qwen38-benchmark" --self-test
    runHook postInstallCheck
  '';

  passthru = {
    upstreamRepo = "OnixResearch/inference-lab";
    upstreamRev = inferenceLabRev;
    rollbackDonorCommit = "5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a";
  };

  meta = {
    description = "Repeatable OpenAI-compatible benchmark for Qwen serving stacks (pinned from OnixResearch/inference-lab)";
    mainProgram = "qwen38-benchmark";
    license = lib.licenses.lgpl21Only;
    platforms = lib.platforms.unix;
  };
}
