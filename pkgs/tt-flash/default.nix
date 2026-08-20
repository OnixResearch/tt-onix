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
  version = "3.10.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-flash";
    tag = "v${version}";
    hash = "sha256-wE8qDgoXiYeUbrcGY46JnPVT4neNGu3U5DTXlMuewjc=";
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

  # tt-flash 3.10.0 still caps Pyluwen below 0.9, while tt-smi 6.2.1 requires Pyluwen 0.9.0.
  pythonRelaxDeps = [
    "pyluwen"
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
