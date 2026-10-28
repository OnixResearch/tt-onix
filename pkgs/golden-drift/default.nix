{
  lib,
  stdenvNoCC,
  python3,
  makeWrapper,
  fetchurl,
  jq,
  tt-kmd,
  tt-smi,
  tt-flash,
  tt-metal,
}:
let
  goldenVersion = "v4.0.0";
  golden = fetchurl {
    url = "https://github.com/tenstorrent/tt-sw-manifest/releases/download/${goldenVersion}/golden.json";
    hash = "sha256-aqMv0Pj2YgDgcnKq9BkDOhgjz3XB4INp5IHiC/Gh5Kk=";
  };
  goldenData = builtins.fromJSON (builtins.readFile golden);
  actual = builtins.toJSON {
    kmd = tt-kmd.version;
    smi = tt-smi.version;
    flash = tt-flash.version;
    sfpi = tt-metal.sfpiVersion;
    metal = tt-metal.version;
    firmware = "not-pinned";
  };
  matching = builtins.toJSON {
    inherit (goldenData)
      kmd
      smi
      flash
      sfpi
      ;
    metal = goldenData."metal-version";
  };
  drifting = builtins.toJSON (
    {
      inherit (goldenData)
        kmd
        smi
        flash
        sfpi
        ;
      metal = goldenData."metal-version";
    }
    // {
      kmd = "0.0.0";
    }
  );
in
stdenvNoCC.mkDerivation {
  pname = "golden-drift";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ./.;
    filter = path: _type: baseNameOf path != "__pycache__";
  };

  nativeBuildInputs = [
    makeWrapper
    jq
  ];
  buildInputs = [ python3 ];

  doCheck = true;
  checkPhase = ''
    runHook preCheck
    ${python3}/bin/python3 golden_drift.py \
      --golden ${golden} --actual ${builtins.toFile "matching.json" matching} --strict
    if ${python3}/bin/python3 golden_drift.py \
      --golden ${golden} --actual ${builtins.toFile "drifting.json" drifting} --strict; then
      echo "golden-drift: a drifted fixture passed strict mode" >&2
      exit 1
    fi
    # The real report is allowed to have no drift; the synthetic fixtures above
    # prove the strict gate. Only require a well-formed report here.
    ${python3}/bin/python3 golden_drift.py \
      --golden ${golden} --actual ${builtins.toFile "actual.json" actual} --json \
      | jq -e '.schema == "onix.golden-drift.v1" and (.components | length) == 6' >/dev/null
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/libexec/golden-drift" "$out/share/golden-drift"
    install -m755 golden_drift.py "$out/libexec/golden-drift/golden_drift.py"
    install -m644 ${golden} "$out/share/golden-drift/golden.json"
    echo ${lib.escapeShellArg actual} >"$out/share/golden-drift/actual.json"
    makeWrapper ${python3}/bin/python3 "$out/bin/golden-drift" \
      --add-flags "$out/libexec/golden-drift/golden_drift.py" \
      --set GOLDEN_DRIFT_GOLDEN "$out/share/golden-drift/golden.json" \
      --set GOLDEN_DRIFT_ACTUAL "$out/share/golden-drift/actual.json" \
      --set GOLDEN_DRIFT_VERSION ${lib.escapeShellArg goldenVersion}
    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/golden-drift" --json | jq -e '.schema == "onix.golden-drift.v1"' >/dev/null
    runHook postInstallCheck
  '';

  meta = {
    description = "Report drift between the packaged Tenstorrent stack and tt-sw-manifest golden.json";
    homepage = "https://github.com/tenstorrent/tt-sw-manifest";
    license = lib.licenses.asl20;
    mainProgram = "golden-drift";
    platforms = lib.platforms.linux;
  };
}
