{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  pyyaml,
  requests,
  tabulate,
  pyluwen,
  pytestCheckHook,
  tt-tools-common,
}:
buildPythonPackage rec {
  pname = "tt-flash";
  version = "4.1.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-flash";
    tag = "v${version}";
    hash = "sha256-7ImAI+A229NzNECu3dL6WvsPO0cn8Id/p8/QqyvZ/H8=";
  };

  build-system = [
    setuptools
  ];

  dependencies = [
    tabulate
    pyyaml
    requests
    pyluwen
    tt-tools-common
  ];

  nativeCheckInputs = [
    pytestCheckHook
  ];

  pytestFlags = [
    "tests/test_validate_p300.py"
  ];

  pythonImportsCheck = [ "tt_flash" ];

  # tt-flash 4.1.0 pins pyyaml 6.0.2 and tabulate 0.9.0; nixpkgs carries newer
  # compatible releases. pyluwen ~=0.10.0 and tt-tools-common ~=1.5 are met.
  pythonRelaxDeps = [
    "pyyaml"
    "tabulate"
  ];

  meta = {
    description = "Tenstorrent Firmware Update Utility";
    homepage = "https://tenstorrent.com";
    maintainers = with lib.maintainers; [ RossComputerGuy ];
    license = with lib.licenses; [ asl20 ];
  };
}
