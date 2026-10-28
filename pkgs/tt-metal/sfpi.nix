{
  lib,
  stdenv,
  fetchurl,
  runCommand,
  autoPatchelfHook,
  ncurses,
  isl_0_23,
  mpfr,
  libmpc,
  xz,
  zstd,
  expat,
}:
let
  version = "7.71.0";
  releaseBuild = "862";
  releaseHashes = {
    aarch64DebianTxz = "52ec6b98542bd5edf0848f9dc4d8346ef445839632bb5f951566f7ef9e081d12";
    aarch64DebianDeb = "42c11ad358b1773b3755d197a8c87ff7444de5cba9f3a73b166397ba5d5342e7";
    x86_64DebianTxz = "550304c067b198abe20020810f0f39d330f74ab28c92a21441cc1aa71eef4b0b";
    x86_64DebianDeb = "4a654505bd7bfe63e50391591f978425d537e97e8f2468272a43d5f9bf9a5b48";
    x86_64FedoraTxz = "dc2820d438af1117e2c3cab033b13487b8147d06f65e542ecf69fe72e1c127a7";
    x86_64FedoraRpm = "734e0e48da513ab5b5adba5c06084594a6d76ff9155234f858a4c1557e9d99d3";
  };
in
runCommand "sfpi-${version}"
  {
    inherit version;
    passthru = {
      inherit releaseBuild releaseHashes;
    };

    nativeBuildInputs = [
      autoPatchelfHook
    ];

    buildInputs = [
      ncurses
      isl_0_23
      mpfr
      libmpc
      xz
      zstd
      expat
    ];

    src =
      {
        aarch64-linux = fetchurl {
          url = "https://github.com/tenstorrent/sfpi/releases/download/${version}/sfpi_${version}_aarch64_debian.txz";
          hash = "sha256-UuxrmFQr1e3whI+dxNg0bvRFg5Yyu1+VFWb3754IHRI=";
        };
        x86_64-linux = fetchurl {
          url = "https://github.com/tenstorrent/sfpi/releases/download/${version}/sfpi_${version}_x86_64_debian.txz";
          hash = "sha256-VQMEwGexmKviACCBDw850zD3SrKMkqIUQcwapx7vSws=";
        };
      }
      ."${stdenv.hostPlatform.system}" or (throw "SFPI does not support ${stdenv.hostPlatform.system}");
  }
  ''
    runPhase unpackPhase
    cp -r ../"$sourceRoot" "$out"
    runPhase fixupPhase
  ''
