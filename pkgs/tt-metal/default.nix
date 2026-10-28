{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchurl,
  fetchzip,
  applyPatches,
  callPackage,
  pkg-config,
  cmake,
  ninja,
  boost,
  capstone,
  numactl,
  mpi,
  hwloc,
  python3,
  fmt,
  nlohmann_json,
  gbenchmark,
  capnproto,
  gtest,
  spdlog,
  yaml-cpp,
  tt-logger,
  enchantum,
  simde,
  xtl,
  xtensor,
  xtensor-blas,
  range-v3,
  libblake3,
  blas,
  lapack,
}:
stdenv.mkDerivation (
  finalAttrs:
  let
    deps = import ./deps.nix {
      inherit
        fetchFromGitHub
        fetchzip
        applyPatches
        ;
      tt-metal-src = finalAttrs.src;
    };
  in
  {
    pname = "tt-metal";
    version = "0.77.0";

    src = fetchFromGitHub {
      owner = "tenstorrent";
      repo = "tt-metal";
      tag = "v${finalAttrs.version}";
      fetchSubmodules = true;
      hash = "sha256-5NY5orbs7w9BTOUsylg8eAXIxJK+smthhkjd6HFePGA=";
    };

    cpm = fetchurl {
      url = "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake";
      hash = "sha256-yM3DLAOBZTjOInge1ylk3IZLKjSjENO3EEgSpcotg10=";
    };

    sfpi = callPackage ./sfpi.nix { };

    patches = [
      # TT-Metal 0.77 already accepts local cadical and header-only dependencies.
      ./umd-asio-local.patch
      # https://github.com/tenstorrent/tt-metal/pull/46224
      ./local-find-package.patch
      # https://github.com/tenstorrent/tt-metal/pull/46229
      ./patched-pins-local.patch
      # tt-umd PR 2187 (./umd-targets-local.patch) merged upstream as of the umd
      # submodule in tt-metal 0.74; dropped (patch now applies in reverse).
      # https://github.com/tenstorrent/tt-metal/issues/49701
      # Drop harvested/nonexistent eth dispatch cores so a 4x p150a mesh can open
      # (the eth dispatch YAML lists cores that don't exist on harvested Blackhole).
      ./bh-eth-dispatch-harvesting.patch
      # Fuse the fp32 single-token gated-delta recurrence into one device program.
      # The public path stays opt-in at the Qwen model layer until physical admission.
      ./gdn-decode-fp32.patch
      # Release the GIL in from_device and the host/device tensor copies. A blocking
      # read after a long prefill waited ~17 s holding it, so the Qwen service's HTTP
      # threads (health checks, /v1/models) stalled until the prefill finished.
      ./transfer-bindings-release-gil.patch
    ];

    patchFlags = [
      "-p1"
      "--fuzz=0"
    ];

    postUnpack = ''
      mkdir -p "$sourceRoot/runtime"
      ln -s "$sfpi" "$sourceRoot/runtime/sfpi"
    '';

    postPatch = ''
      cp $cpm cmake/CPM.cmake
      cp $cpm tt_metal/third_party/umd/cmake/CPM.cmake
      patchShebangs .
      substituteInPlace tt_metal/sfpi-info.sh --replace-fail "sfpi_dist=unknown" "sfpi_dist=debian"
      # Keep TT-Metal's compiler guard and installed metadata aligned with the
      # newer correctness-fix SFPI release carried by this package.
      cat > tt_metal/sfpi-version <<'EOF'
      # sfpi release ${finalAttrs.sfpi.version}

      sfpi_repo="https://github.com/tenstorrent/sfpi"
      sfpi_version="${finalAttrs.sfpi.version}"
      sfpi_base=""
      sfpi_build="${finalAttrs.sfpi.releaseBuild}"
      sfpi_hashtype="sha256"
      sfpi_aarch64_debian_txz_hash="${finalAttrs.sfpi.releaseHashes.aarch64DebianTxz}"
      sfpi_aarch64_debian_deb_hash="${finalAttrs.sfpi.releaseHashes.aarch64DebianDeb}"
      sfpi_x86_64_debian_txz_hash="${finalAttrs.sfpi.releaseHashes.x86_64DebianTxz}"
      sfpi_x86_64_debian_deb_hash="${finalAttrs.sfpi.releaseHashes.x86_64DebianDeb}"
      sfpi_x86_64_fedora_txz_hash="${finalAttrs.sfpi.releaseHashes.x86_64FedoraTxz}"
      sfpi_x86_64_fedora_rpm_hash="${finalAttrs.sfpi.releaseHashes.x86_64FedoraRpm}"
      EOF
      # Keep Cap'n Proto discovery offline. Its block moved independently of
      # the local-find-package patch. Upstream already names googletest GTest.
      substituteInPlace third_party/CMakeLists.txt --replace-fail "NAME capnproto" "NAME CapnProto"
      sed -i 's|^        capnproto_pthread.patch$|&\n    FIND_PACKAGE_ARGUMENTS GLOBAL|' third_party/CMakeLists.txt

      # Disable Tracy's profiler CLI tools + WASM viewer: they pull a GUI/web CPM
      # stack (imgui/glfw/emsdk) and run `emsdk install` at configure.
      # Keep TracyClient and its generated debug-category header enabled.
      sed -i '/^add_subdirectory(tracy\/csvexport)$/i if(FALSE) # nix: profiler tools + WASM viewer disabled; TracyClient is built above' tt_metal/third_party/CMakeLists.txt
      sed -i '/^set(_tt_tracy_categories_file /i endif() # nix: retain runtime debug-category generation' tt_metal/third_party/CMakeLists.txt
    '';

    cmakeFlags = [
      (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
      (lib.cmakeBool "CPM_USE_LOCAL_PACKAGES" true)
      (lib.cmakeBool "WITH_PYTHON_BINDINGS" true)
      (lib.cmakeBool "TT_INSTALL" true)
      (lib.cmakeFeature "VERSION_NUMERIC" finalAttrs.version)
      (lib.cmakeFeature "CMAKE_POLICY_VERSION_MINIMUM" "3.10")
      (lib.cmakeFeature "cadical_SOURCE_DIR" (builtins.toString deps.cadical))
      (lib.cmakeFeature "umd_asio_SOURCE_DIR" (builtins.toString deps.umd_asio))
      (lib.cmakeFeature "CAPNP_INCLUDE_DIRECTORY" "${lib.getDev capnproto}/include")
      (lib.cmakeFeature "reflect_SOURCE_DIR" (builtins.toString deps.reflect))
      (lib.cmakeFeature "simd-everywhere_SOURCE_DIR" "${simde}/include")
      # ELFIO is fetched by tt-exalens's nested CPM, which ignores build/_deps;
      # point CPM straight at the source.
      (lib.cmakeFeature "CPM_ELFIO_SOURCE" (builtins.toString deps.elfio))
    ];

    preConfigure = ''
      mkdir -p build/_deps
      ${lib.concatMapAttrsStringSep "\n" (
        name: src: "cp -r --no-preserve=ownership,mode ${src} build/_deps/${name}-src"
      ) deps}
    '';

    env.NIX_CFLAGS_COMPILE = "-Wno-error=unused-but-set-variable";

    enableParallelBuilding = true;

    nativeBuildInputs = [
      cmake
      ninja
      pkg-config
      python3
    ];

    buildInputs = [
      numactl
      boost
      capstone
      mpi
      hwloc
      fmt
      nlohmann_json
      gbenchmark
      capnproto
      gtest
      spdlog
      yaml-cpp
      tt-logger
      enchantum
      xtl
      xtensor
      xtensor-blas
      range-v3
      libblake3
      blas
      lapack
    ];

    postInstall = ''
      # The default install ships only a subset of ttnn C++ op headers; the
      # Metalium backend includes many more via two forms (<ttnn/...> from
      # $out/include and <ttnn/cpp/ttnn/...> from libexec). Install the full tree
      # to both roots.
      ( cd ../ttnn/cpp && find ttnn -name '*.hpp' -print0 | while IFS= read -r -d "" h; do
          install -Dm444 "$h" "$out/include/$h"
          install -Dm444 "$h" "$out/libexec/tt-metalium/ttnn/cpp/$h"
        done )

      mkdir -p $out/${python3.sitePackages}
      cp -r ../ttnn/ttnn $out/${python3.sitePackages}/ttnn
      cp -r ../ttnn/tt_lib $out/${python3.sitePackages}/tt_lib
      cp -r ../tools/tracy $out/${python3.sitePackages}/tracy
      cp $out/lib/_ttnn.so $out/${python3.sitePackages}/ttnn/_ttnn.so

      mkdir -p $out/${python3.sitePackages}/ttnn-${finalAttrs.version}.dist-info
      cat > $out/${python3.sitePackages}/ttnn-${finalAttrs.version}.dist-info/METADATA <<EOF
      Metadata-Version: 2.1
      Name: ttnn
      Version: ${finalAttrs.version}
      EOF
    '';

    # Fixes the parallel hook crashing in the fixupPhase with no error.
    noAuditTmpdir = true;

    installTargets = [
      "install"
      "ttnn"
    ];

    passthru = {
      sfpiVersion = finalAttrs.sfpi.version;
    };

    meta = {
      description = "TT-NN operator library, and TT-Metalium low level kernel programming model";
      homepage = "https://github.com/tenstorrent/tt-metal";
      maintainers = with lib.maintainers; [ RossComputerGuy ];
      license = with lib.licenses; [ asl20 ];
      platforms = lib.platforms.linux;
    };
  }
)
