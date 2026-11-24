{
  lib,
  stdenvNoCC,
  fetchFromGitHub,
  makeWrapper,
  python3,
  tt-npe,
}:
let
  # Pinned portable six-passage retention evaluator migrated to
  # OnixResearch/inference-lab (pkgs/retention-eval).
  # Explicit rollback pin: OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a
  inferenceLabRev = "fd90c12231f6664c64e72c47050d90fd14f968e6";
  inferenceLabSrc = fetchFromGitHub {
    owner = "OnixResearch";
    repo = "inference-lab";
    rev = inferenceLabRev;
    hash = "sha256-3VbKenyiglxNX9MNV35AQrO1ioJdcfM5KMN89jTnVwQ=";
  };
  python = python3.withPackages (p: [
    p.torch
    p.transformers
  ]);
in
stdenvNoCC.mkDerivation {
  pname = "tenstorrent-eval-tools";
  version = "0.1.0";
  src = lib.cleanSourceWith {
    src = ./.;
    filter = path: _type: baseNameOf path != "__pycache__";
  };
  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;
  postUnpack = ''
    cp ${inferenceLabSrc}/pkgs/retention-eval/retention.py "$sourceRoot/retention.py"
    cp -r ${inferenceLabSrc}/pkgs/retention-eval/data "$sourceRoot/data"
    cp ${inferenceLabSrc}/pkgs/retention-eval/PROVENANCE.json "$sourceRoot/RETENTION-PROVENANCE.json"
    chmod -R u+w "$sourceRoot/retention.py" "$sourceRoot/data" "$sourceRoot/RETENTION-PROVENANCE.json"
  '';
  doCheck = true;
  checkPhase = ''
    ${python}/bin/python -m unittest discover -s tests -v
  '';
  installPhase = ''
    mkdir -p $out/share/tenstorrent-eval-tools $out/bin
    cp -r *.py data LICENSE.tensixfuse RETENTION-PROVENANCE.json $out/share/tenstorrent-eval-tools/
    makeWrapper ${python}/bin/python $out/bin/tenstorrent-eval \
      --unset PYTHONPATH --unset PYTHONHOME \
      --add-flags "-I $out/share/tenstorrent-eval-tools/cli.py"
    ln -s ${tt-npe}/bin/tt-npe $out/bin/tt-npe
    ln -s ${tt-npe}/bin/npe_analyze_noc_trace_dir $out/bin/npe_analyze_noc_trace_dir
  '';
  passthru = {
    retentionUpstreamRepo = "OnixResearch/inference-lab";
    retentionUpstreamRev = inferenceLabRev;
    rollbackDonorCommit = "5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a";
  };
  meta = {
    description = "Analytic tile traffic and provenance-bound six-passage retention tools";
    license = lib.licenses.lgpl21Only;
    mainProgram = "tenstorrent-eval";
    platforms = [ "x86_64-linux" ];
  };
}
