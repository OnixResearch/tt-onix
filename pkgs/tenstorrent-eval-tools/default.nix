{
  lib,
  stdenvNoCC,
  inference-lab-source,
  makeWrapper,
  python3,
  tt-npe,
}:
let
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
    cp ${inference-lab-source.src}/pkgs/retention-eval/retention.py "$sourceRoot/retention.py"
    cp -r ${inference-lab-source.src}/pkgs/retention-eval/data "$sourceRoot/data"
    cp ${inference-lab-source.src}/pkgs/retention-eval/PROVENANCE.json "$sourceRoot/RETENTION-PROVENANCE.json"
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
    retentionUpstreamRepo = inference-lab-source.repo;
    retentionUpstreamRev = inference-lab-source.rev;
    rollbackDonorCommit = inference-lab-source.rollbackDonorCommit;
  };
  meta = {
    description = "Analytic tile traffic and provenance-bound six-passage retention tools";
    license = lib.licenses.lgpl21Only;
    mainProgram = "tenstorrent-eval";
    platforms = [ "x86_64-linux" ];
  };
}
