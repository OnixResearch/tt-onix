{
  lib,
  python3Packages,
  fetchFromGitHub,
}:
python3Packages.buildPythonPackage (finalAttrs: {
  pname = "vllm-tt-plugin";
  version = "0.1.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "vllm-tt-plugin";
    rev = "bef89e429e202caa38d3e3c8a24da4e8cc02405a";
    hash = "sha256-DsSfRqTTdNO/a5NVqUed7x5hkK0X2BMNODekxnGNiMI=";
  };

  # Upstream caps setuptools below 81; nixpkgs ships 83. The plugin does not
  # use removed setuptools APIs, so relax only the build-time cap.
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail 'requires = ["setuptools>=77.0.3,<81.0.0", "wheel"]' \
                     'requires = ["setuptools>=77.0.3", "wheel"]'
  '';

  build-system = with python3Packages; [
    setuptools
    wheel
  ];

  dependencies = with python3Packages; [
    tblib
  ];

  # vLLM is intentionally absent: it must come from the locally built TT
  # `VLLM_TARGET_DEVICE=empty` environment, not the CUDA wheel on PyPI.
  dontCheckRuntimeDeps = true;
  pythonImportsCheck = [ "vllm_tt_plugin" ];

  meta = {
    description = "vLLM platform plugin that runs models on Tenstorrent hardware";
    homepage = "https://github.com/tenstorrent/vllm-tt-plugin";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ RossComputerGuy ];
    platforms = lib.platforms.linux;
  };
})
