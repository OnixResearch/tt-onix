{
  lib,
  fetchurl,
  python3,
  writeShellApplication,
  symlinkJoin,
}:
let
  simulator = python3.pkgs.buildPythonPackage {
    pname = "tt-lang-sim";
    version = "1.1.6";
    format = "wheel";
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/85/cc/cb675e7d9b89757a064956ee9df9bc6a625c95abea2e096e5c9e89561fd1/tt_lang_sim-1.1.6-py3-none-any.whl";
      hash = "sha256-K3XtnllD8zhUKEHc8xD0UaSuO0aKM3nM+zcUIWXeOn0=";
    };
    dependencies = with python3.pkgs; [
      pydantic
      torch
      numpy
      greenlet
      pyyaml
      typing-extensions
      ml-dtypes
    ];
    pythonImportsCheck = [
      "ttl.sim"
      "sim_stats"
    ];
    # Setup entry points download toolchains or copy tutorials into mutable trees.
    postInstall = ''
      rm -f "$out/bin/tt-lang-setup" "$out/bin/tt-lang-setup-tutorials"
    '';
    meta.license = lib.licenses.asl20;
  };
  runtime = python3.withPackages (_: [ simulator ]);
  command =
    name: module:
    writeShellApplication {
      inherit name;
      text = ''
        unset PYTHONPATH PYTHONHOME
        export PYTHONNOUSERSITE=1
        exec ${runtime}/bin/python -I -m ${module} "$@"
      '';
    };
in
symlinkJoin {
  name = "tt-lang-sim-1.1.6";
  paths = [
    (command "tt-lang-sim" "ttl.sim.ttlang_sim")
    (command "tt-lang-sim-stats" "sim_stats")
  ];
  passthru = {
    inherit simulator;
  };
  meta = {
    description = "Isolated device-free TT-Lang simulator and modeled traffic statistics";
    homepage = "https://github.com/tenstorrent/tt-lang";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
    mainProgram = "tt-lang-sim";
  };
}
