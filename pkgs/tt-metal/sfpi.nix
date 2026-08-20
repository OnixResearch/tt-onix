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
  version = "7.70.0";
  releaseBuild = "847";
  releaseHashes = {
    aarch64DebianTxz = "9081fe36a768abe47cbd3306b8d747f9f01ba1d524b1e4201989609dbcac9258";
    aarch64DebianDeb = "b63f3c7607dad59e406a4960877cf103d9bec9f1c8b1438e4889036b9f6ff902";
    x86_64DebianTxz = "06d1c43831c2ea42a062ad46d274cf7221a3c013c638d52da2de4660586e4b30";
    x86_64DebianDeb = "ce1fcbfde600ceaf133e575113ffe1904ad2f0e015ab680ba1d0d3bf727aa2ca";
    x86_64FedoraTxz = "c6903fc47f1d854954237c38a314d773731a240447f437034a8341cb38ab2812";
    x86_64FedoraRpm = "cccb4791e555aca12233466bf5e249bbf56c452d97d6afa495c8c6e589d60bc5";
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
          hash = "sha256-kIH+Nqdoq+R8vTMGuNdH+fAbodUkseQgGYlgnbysklg=";
        };
        x86_64-linux = fetchurl {
          url = "https://github.com/tenstorrent/sfpi/releases/download/${version}/sfpi_${version}_x86_64_debian.txz";
          hash = "sha256-BtHEODHC6kKgYq1G0nTPciGjwBPGONUtot5GYFhuSzA=";
        };
      }
      ."${stdenv.hostPlatform.system}" or (throw "SFPI does not support ${stdenv.hostPlatform.system}");
  }
  ''
    runPhase unpackPhase
    cp -r ../"$sourceRoot" "$out"
    runPhase fixupPhase
  ''
