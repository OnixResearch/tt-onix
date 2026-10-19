{
  lib,
  fetchFromGitHub,
  fetchurl,
  gzip,
  makeRustPlatform,
  runCommand,
  rustToolchain,
}:

let
  rustPlatform = makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };

  src = fetchFromGitHub {
    owner = "brittonr";
    repo = "metalium-rs";
    rev = "d8c6e2fc4579cf29aa7525ddd086a90f81da2260";
    hash = "sha256-izG12OWV37sIUiQFYam701M6FR5kUAvqYvvLeEEqT28=";
  };

  mnist = runCommand "mnist-idx-2017" { nativeBuildInputs = [ gzip ]; } ''
    mkdir -p $out
    gzip -dc ${
      fetchurl {
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
        sha256 = "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609";
      }
    } > $out/train-images-idx3-ubyte
    gzip -dc ${
      fetchurl {
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";
        sha256 = "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c";
      }
    } > $out/train-labels-idx1-ubyte
    gzip -dc ${
      fetchurl {
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz";
        sha256 = "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6";
      }
    } > $out/t10k-images-idx3-ubyte
    gzip -dc ${
      fetchurl {
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz";
        sha256 = "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6";
      }
    } > $out/t10k-labels-idx1-ubyte
  '';

  isaDocumentation = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-isa-documentation";
    rev = "f848eb668c2aeae742a88a49a86157e24a0a20c6";
    hash = "sha256-varzkuN6utL6xA3GDd3Fe9XnrcFlpansaDKSNoB+/OY=";
  };

  ttsimLibs = runCommand "ttsim-1.10.9" { } ''
    mkdir -p $out
    cp ${
      fetchurl {
        url = "https://github.com/tenstorrent/ttsim/releases/download/v1.10.9/libttsim_bh.so";
        sha256 = "e6ed2da11718683738d43f14a0bf4f13285b8621697b3165c1eaa240d36cfad5";
      }
    } $out/libttsim_bh.so
    cp ${
      fetchurl {
        url = "https://github.com/tenstorrent/ttsim/releases/download/v1.10.9/libttsim_bh_x2.so";
        sha256 = "e1ffbaf39c7d071a2f64d7072a586da5f9a7bef55907459edbdb185189d23108";
      }
    } $out/libttsim_bh_x2.so
    cp ${
      fetchurl {
        url = "https://github.com/tenstorrent/ttsim/releases/download/v1.10.9/libttsim_bh_x4.so";
        sha256 = "942a7a2a210259b77d1c44a7b13ac98fe672aec28d28e4385fcd30f6967756c1";
      }
    } $out/libttsim_bh_x4.so
  '';

  cfgDefines = fetchurl {
    url = "https://raw.githubusercontent.com/tenstorrent/tt-metal/81989dcdb8f9b340c932ae7a71a346f4f08703eb/tt_metal/hw/inc/blackhole/cfg_defines.h";
    sha256 = "bc2636abc3ea04e6ca322923e6f2713b242857d021d8ef39b9180a14bcb1a5b8";
  };

  ioctlHeader = fetchurl {
    url = "https://raw.githubusercontent.com/tenstorrent/tt-kmd/ttkmd-2.11.0/ioctl.h";
    sha256 = "831031cdb556f3f13597af838535d4ce594b47b63d685e87312db76087fbd4dc";
  };
in
rustPlatform.buildRustPackage {
  pname = "metalium-rs";
  version = "0.1.0-unstable-2026-10-19";

  inherit src;

  cargoLock.lockFile = "${src}/Cargo.lock";

  # Dev-only crates stay out of the installed package, but the check lane runs
  # the full workspace with the pinned specification and simulator payloads.
  cargoBuildFlags = [
    "--workspace"
    "--exclude"
    "tt-tests"
    "--exclude"
    "tt-ttsim"
    "--exclude"
    "tt-ttsim-sys"
    "--exclude"
    "burn-tt-conformance"
    "--exclude"
    "tt-models"
    "--exclude"
    "xtask"
  ];

  doCheck = true;
  checkType = "debug";

  cargoTestFlags = [
    "--workspace"
    "--no-fail-fast"
  ];

  # `tt-firmware-images` runs a nested cargo for the bare-metal target. Cargo
  # only needs a writable home for its package cache; the nested build has no
  # registry dependencies. `tt-mnist` embeds the pinned corpus at build time.
  preBuild = ''
    export HOME=$TMPDIR
    mkdir -p vendor
    cp -r ${isaDocumentation} vendor/tt-isa-documentation
    cp ${ttsimLibs}/libttsim_bh*.so vendor/
    cp ${cfgDefines} vendor/cfg_defines.h
    cp ${ioctlHeader} vendor/ioctl.h
    mkdir -p vendor/mnist
    cp ${mnist}/* vendor/mnist/
  '';

  meta = {
    description = "Native Rust stack for Tenstorrent Blackhole";
    homepage = "https://github.com/brittonr/metalium-rs";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
