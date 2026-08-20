{
  lib,
  python3Packages,
  fetchFromGitHub,
  versionCheckHook,
}:
python3Packages.buildPythonApplication rec {
  pname = "tt-topology";
  version = "1.2.20";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-topology";
    tag = "v${version}";
    hash = "sha256-oPhzNnlZszcXLSy29xfbhU5ML+twgeu2U794zdqSssI=";
  };

  build-system = with python3Packages; [
    setuptools
    setuptools-scm
  ];

  dependencies = with python3Packages; [
    elasticsearch
    pydantic
    pyluwen
    tt-tools-common
    networkx
    matplotlib

    # Needed for "pkg_resources"
    setuptools
  ];

  nativeCheckInputs = [
    versionCheckHook
  ];

  pythonRemoveDeps = [
    "black"
    "pre-commit"
  ];

  # tt-topology 1.2.20 still caps Pyluwen below 0.9, while tt-smi 6.2.1 requires Pyluwen 0.9.0.
  pythonRelaxDeps = [
    "elasticsearch"
    "networkx"
    "matplotlib"
    "pyluwen"
    "setuptools"
  ];

  # Tests are broken
  dontUsePytestCheck = true;

  meta = {
    mainProgram = "tt-topology";
    description = "Command line utility used to flash multiple NB cards on a system to use specific eth routing configurations";
    homepage = "https://github.com/tenstorrent/tt-topology";
    changelog = "https://github.com/tenstorrent/tt-topology/blob/${src.tag}/CHANGELOG.md";
    maintainers = with lib.maintainers; [ RossComputerGuy ];
    license = with lib.licenses; [ asl20 ];
  };
}
