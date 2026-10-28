{
  lib,
  python3Packages,
  fetchPypi,
}:
python3Packages.buildPythonPackage (finalAttrs: {
  pname = "faster-fifo";
  version = "1.5.2";
  pyproject = true;

  src = fetchPypi {
    pname = "faster_fifo";
    inherit (finalAttrs) version;
    hash = "sha256-oqVE/vTW4x69S7THu/ztMV10HDt8ytI/4KNZ17QIUn4=";
  };

  build-system = with python3Packages; [
    setuptools
    cython
  ];

  # numpy is a development/test dependency; the queue uses pickle.
  pythonImportsCheck = [ "faster_fifo" ];

  meta = {
    description = "Fast multiprocessing queue backed by a shared-memory circular buffer";
    homepage = "https://github.com/alex-petrenko/faster-fifo";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ RossComputerGuy ];
  };
})
