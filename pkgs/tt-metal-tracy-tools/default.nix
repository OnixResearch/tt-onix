# Host-side Tracy CLI tools matching the packaged tt-metal runtime.
#
# The Nix tt-metal package disables Tracy's profiler CLI tools and WASM viewer
# because upstream builds them through a GUI/web CPM stack. The packaged Python
# `tracy` module still expects `tracy-capture` and `tracy-csvexport` under
# $TT_METAL_HOME/build/tools/profiler/bin, so this package rebuilds exactly
# those two binaries from the package-matched tenstorrent/tracy fork vendored
# in the tt-metal source tree, with system capstone/zstd instead of CPM
# downloads. See ./CMakeLists.txt for the dependency mapping.
{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  capstone,
  zstd,
  fetchFromGitHub,
  applyPatches,
  tt-metal,
}:
let
  tracyRoot = "${tt-metal.src}/tt_metal/third_party/tracy";
  # Same revision the fork's vendor.cmake pins; patched with the fork's own
  # patches so the header-only sort matches what tt-metal's build would use.
  ppqsortRoot = applyPatches {
    src = fetchFromGitHub {
      owner = "GabTux";
      repo = "PPQSort";
      tag = "v1.0.6";
      hash = "sha256-HgM+p2QGd9C8A8l/VaEB+cLFDrY2HU6mmXyTNh7xd0A=";
    };
    patches = [
      "${tracyRoot}/cmake/ppqsort-nodebug.patch"
      "${tracyRoot}/cmake/ppqsort-semaphore.patch"
    ];
  };
in
stdenv.mkDerivation {
  pname = "tt-metal-tracy-tools";
  version = "0.74.0";

  src = ./.;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ];
  buildInputs = [
    capstone
    zstd
  ];

  cmakeFlags = [
    (lib.cmakeFeature "ppqsort_root" "${ppqsortRoot}")
    (lib.cmakeFeature "git_ref" "package-matched")
  ];

  # The fork's TracyWorker.cpp uses capstone 6 (Alpha) names. The system
  # capstone 5 keeps the stable pre-rename API, so rewrite the three renamed
  # identifiers in a writable copy of the fork before configuring.
  preConfigure = ''
    cp -r --no-preserve=mode ${tracyRoot} ./tracy
    substituteInPlace tracy/server/TracyWorker.cpp \
      --replace-fail 'CS_ARCH_AARCH64' 'CS_ARCH_ARM64' \
      --replace-fail 'AARCH64_OP_IMM' 'ARM64_OP_IMM' \
      --replace-fail 'detail.aarch64' 'detail.arm64'
    cmakeFlagsArray+=("-Dtracy_root=$PWD/tracy")
  '';
}
