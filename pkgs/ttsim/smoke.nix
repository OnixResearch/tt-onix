{
  lib,
  stdenvNoCC,
  replaceVars,
  writeTextFile,
  makeWrapper,
  coreutils,
  python3,
  tt-metal,
  ttsim,
  ttsim-runtime-validator,
}:
let
  python = python3.withPackages (
    pythonPackages: with pythonPackages; [
      click
      graphviz
      loguru
      ml-dtypes
      networkx
      numpy
      pandas
      pyyaml
      seaborn
    ]
  );
  smokeScript = writeTextFile {
    name = "ttsim-smoke.py";
    text = builtins.readFile ./smoke.py;
    executable = true;
  };
  runner = replaceVars ./run-smoke.sh {
    ttMetal = tt-metal;
    ttMetalVersion = tt-metal.version;
    inherit python ttsim;
    pythonSitePackages = python3.sitePackages;
    inherit smokeScript;
    validator = ttsim-runtime-validator;
    hostSystem = stdenvNoCC.hostPlatform.system;
  };
in
stdenvNoCC.mkDerivation {
  pname = "ttsim-blackhole-smoke";
  version = "0.1.0";

  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin"
    cp ${runner} "$out/bin/ttsim-blackhole-smoke"
    chmod +x "$out/bin/ttsim-blackhole-smoke"
    wrapProgram "$out/bin/ttsim-blackhole-smoke" \
      --prefix PATH : ${lib.makeBinPath [ coreutils ]}
    runHook postInstall
  '';

  meta = {
    description = "Bounded Blackhole 1x1 TT-NN smoke on TTsim";
    license = lib.licenses.mit;
    mainProgram = "ttsim-blackhole-smoke";
    platforms = [
      "aarch64-linux"
      "x86_64-linux"
    ];
  };
}
