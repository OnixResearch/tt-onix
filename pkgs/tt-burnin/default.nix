{
  lib,
  fetchFromGitHub,
  python3Packages,
  versionCheckHook,
}:
python3Packages.buildPythonApplication (finalAttrs: {
  pname = "tt-burnin";
  version = "0.4.4";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-burnin";
    tag = "v${finalAttrs.version}";
    hash = "sha256-ZHc9qhOZE8fv/oj0bHGTyncZNT1mJtk1pFAPCRY4TtE=";
  };

  build-system = with python3Packages; [
    setuptools
    setuptools-scm
  ];

  dependencies = with python3Packages; [
    pyluwen
    tt-tools-common
    jsons
  ];

  # tt-burnin 0.4.4 still pins pyluwen ~=0.9.0; the coordinated diagnostics
  # stack now carries 0.10.0, which tt-smi 6.7 and tt-flash 4.1 require.
  pythonRelaxDeps = [ "pyluwen" ];

  nativeCheckInputs = [
    versionCheckHook
  ];

  meta = {
    mainProgram = "tt-burnin";
    description = "Command line utility to run a high power consumption workload on TT devices";
    homepage = "https://github.com/tenstorrent/tt-burnin";
    changelog = "https://github.com/tenstorrent/tt-burnin/blob/${finalAttrs.src.tag}/CHANGELOG.md";
    maintainers = with lib.maintainers; [ RossComputerGuy ];
    license = with lib.licenses; [ asl20 ];
  };
})
