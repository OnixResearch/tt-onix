{
  lib,
  stdenvNoCC,
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
  src = lib.cleanSource ./.;
  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;
  doCheck = true;
  checkPhase = ''
    ${python}/bin/python -m unittest discover -s tests -v
  '';
  installPhase = ''
    mkdir -p $out/share/tenstorrent-eval-tools $out/bin
    cp -r *.py data LICENSE.tensixfuse $out/share/tenstorrent-eval-tools/
    makeWrapper ${python}/bin/python $out/bin/tenstorrent-eval \
      --unset PYTHONPATH --unset PYTHONHOME \
      --add-flags "-I $out/share/tenstorrent-eval-tools/cli.py"
    ln -s ${tt-npe}/bin/tt-npe $out/bin/tt-npe
    ln -s ${tt-npe}/bin/npe_analyze_noc_trace_dir $out/bin/npe_analyze_noc_trace_dir
  '';
  meta = {
    description = "Analytic tile traffic and provenance-bound six-passage retention tools";
    license = lib.licenses.lgpl21Only;
    mainProgram = "tenstorrent-eval";
    platforms = [ "x86_64-linux" ];
  };
}
