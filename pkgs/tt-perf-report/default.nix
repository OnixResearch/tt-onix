# tt-perf-report analyzes tt-metal profiler op traces. Pure-Python setuptools
# package with a `tt-perf-report` console script.
{
  lib,
  python3Packages,
  fetchurl,
}:
python3Packages.buildPythonApplication rec {
  pname = "tt-perf-report";
  version = "1.4.1";
  pyproject = true;

  # The pinned hash is the GitHub tag tarball hash. fetchFromGitHub hashes the
  # unpacked tree instead, so fetch the archive itself and let stdenv unpack it.
  src = fetchurl {
    url = "https://github.com/tenstorrent/tt-perf-report/archive/refs/tags/v${version}.tar.gz";
    hash = "sha256-YlZIAVOtEDvOAio/YGT1ckDQmQjhJdlXkPXF5TpjLxc=";
  };

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    pandas
    matplotlib
  ];

  # Upstream pins matplotlib==3.10.9 exactly; nixpkgs tracks a newer patch
  # release of the same minor series.
  pythonRelaxDeps = [ "matplotlib" ];

  nativeCheckInputs = with python3Packages; [
    pytestCheckHook
    pytest-mock
  ];

  # Upstream renders charts during a few code paths; keep checks headless and
  # away from the unwritable build-home config directory.
  env.MPLBACKEND = "Agg";
  preCheck = ''
    export MPLCONFIGDIR="$(mktemp -d)"
  '';

  # The CLI has no --version flag, so assert the wheel metadata and the script
  # instead of using versionCheckHook.
  doInstallCheck = true;
  postInstallCheck = ''
    export MPLCONFIGDIR="$(mktemp -d)"
    PYTHONPATH="$out/${python3Packages.python.sitePackages}" \
      ${python3Packages.python.interpreter} -c 'import importlib.metadata as m, sys; v = m.version("tt-perf-report"); assert v == "${version}", f"expected tt-perf-report ${version}, got {v}"; print(f"tt-perf-report {v}")'
    "$out/bin/tt-perf-report" --help > /dev/null
  '';

  pythonImportsCheck = [ "tt_perf_report" ];

  meta = {
    description = "Analyze TT-Metal profiler traces for throughput, bottlenecks and optimization hints";
    homepage = "https://github.com/tenstorrent/tt-perf-report";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ RossComputerGuy ];
    mainProgram = "tt-perf-report";
  };
}
