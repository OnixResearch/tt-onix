{
  lib,
  fetchFromGitHub,
  callPackage,
  cmake,
  ninja,
  pkg-config,
  makeWrapper,
  patchelf,
  llvmPackages_20,
  boost,
  fmt_11,
  magic-enum,
  nlohmann_json,
  simdjson,
  yaml-cpp,
  zstd,
  gtest,
  python3,
}:
let
  # Upstream publishes no tags, so the commit is the only stable identity.
  # This is the tip of `main` observed on 2026-10-28 (committed 2026-10-08);
  # the version string tracks the observation date, not the commit date.
  sourceRevision = "42cf88f8c3af30c8fa3e1abf5fe460a9fc38be7f";

  inherit (llvmPackages_20) stdenv;

  # `npe_analyze_noc_trace_dir` is the entrypoint tt-metal's profiler imports;
  # it needs orjson at runtime.
  python = python3.withPackages (ps: [ ps.orjson ]);
  sitePackages = python3.sitePackages;
  deviceModelDir = "share/tt_npe/device/models";
in
stdenv.mkDerivation (finalAttrs: {
  pname = "tt-npe";
  version = "0-unstable-2026-10-28";
  __structuredAttrs = true;
  strictDeps = true;

  src = fetchFromGitHub {
    owner = "tenstorrent";
    repo = "tt-npe";
    rev = sourceRevision;
    hash = "sha256-KN0a44LQwU2ek71RXjDRd6hR9PylvIfHHzAaz/WXes4=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    makeWrapper
    patchelf
    python3
  ];

  buildInputs = [
    boost
    fmt_11
    magic-enum
    nlohmann_json
    simdjson
    yaml-cpp
    zstd
    gtest
    python3
    python3.pkgs.pybind11
  ];

  postPatch = ''
    # Dependency mapping (upstream CPM/FetchContent -> nixpkgs):
    #   fmt 11.1.4          -> fmt_11 11.2.0
    #   magic_enum 0.9.6    -> magic-enum 0.9.8
    #   nlohmann/json 3.12  -> nlohmann_json 3.12.0
    #   simdjson 3.12.3     -> simdjson 4.6.0 (DOM API still present)
    #   yaml-cpp 0.8.0      -> yaml-cpp 0.8.0
    #   zstd 1.5.5 (static) -> zstd 1.5.7 (shared; exposes zstd::libzstd)
    #   googletest 1.16.0   -> gtest 1.17.0
    #   pybind11 2.13.6     -> python3Packages.pybind11 3.0.4
    #   Boost 1.87 (git)    -> boost 1.89 headers (nixpkgs ships no
    #                          BoostConfig.cmake, so the interface targets are
    #                          defined by hand below)
    # Fetches dropped because no translation unit uses them: boost-ext/reflect,
    # Boost.tokenizer and Boost.ProgramOptions (the latter is linked upstream
    # but never included).
    #
    # Upstream resolves every dependency through CPM/FetchContent, which needs
    # network access at configure time. Replace that file with system packages
    # from nixpkgs; nothing else in the tree includes CPM.
    cat > cmake/dependencies.cmake <<'EOF'
    # SPDX-License-Identifier: Apache-2.0
    # Nixpkgs-provided dependencies, see pkgs/tt-npe/default.nix.
    find_package(Threads REQUIRED)
    find_package(fmt CONFIG REQUIRED)
    find_package(magic_enum CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(simdjson CONFIG REQUIRED)
    find_package(yaml-cpp CONFIG REQUIRED)
    find_package(zstd CONFIG REQUIRED)
    find_package(GTest CONFIG REQUIRED)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    # Upstream CMakeLists predates namespaced imported targets. magic_enum
    # 0.9.8 moved its headers into include/magic_enum/, while the sources use
    # the historical <magic_enum.hpp> spelling.
    if(NOT TARGET magic_enum)
        set_property(TARGET magic_enum::magic_enum APPEND PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES "@magicEnumIncludeDir@")
        add_library(magic_enum ALIAS magic_enum::magic_enum)
    endif()
    if(NOT TARGET simdjson)
        add_library(simdjson ALIAS simdjson::simdjson)
    endif()
    if(NOT TARGET yaml-cpp)
        add_library(yaml-cpp ALIAS yaml-cpp::yaml-cpp)
    endif()
    if(NOT TARGET libzstd_static)
        add_library(libzstd_static ALIAS zstd::libzstd)
    endif()
    if(NOT TARGET gtest)
        add_library(gtest ALIAS GTest::gtest)
    endif()
    if(NOT TARGET gtest_main)
        add_library(gtest_main ALIAS GTest::gtest_main)
    endif()

    # nixpkgs ships Boost headers and libraries but no BoostConfig.cmake.
    # Only headers are required: Boost.Unordered, Boost.Container and
    # Boost.Core are header-only here, and no translation unit includes
    # Boost.ProgramOptions (the upstream link is vestigial).
    if(NOT TARGET Boost::core)
        add_library(npe_boost_headers INTERFACE IMPORTED)
        set_target_properties(npe_boost_headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "@boostIncludeDir@")
        add_library(Boost::core ALIAS npe_boost_headers)
        add_library(Boost::container ALIAS npe_boost_headers)
        add_library(Boost::unordered ALIAS npe_boost_headers)
        add_library(Boost::program_options ALIAS npe_boost_headers)
    endif()

    # Upstream aliases Boost.Core as `span` for the unit tests. Nothing in the
    # tree includes boost/span or span-lite; keep the target for compatibility.
    if(NOT TARGET span)
        add_library(span INTERFACE)
        target_link_libraries(span INTERFACE Boost::core)
    endif()
    EOF
    substituteInPlace cmake/dependencies.cmake \
      --replace-fail "@boostIncludeDir@" "${boost.dev}/include" \
      --replace-fail "@magicEnumIncludeDir@" "${magic-enum}/include/magic_enum"

    # zstd_SOURCE_DIR only exists for the FetchContent build; the imported
    # zstd::libzstd target already carries the include directory.
    sed -i '/zstd_SOURCE_DIR/d' tt_npe/CMakeLists.txt

    # Bake the installed model-config location instead of the build tree.
    substituteInPlace tt_npe/CMakeLists.txt \
      --replace-fail \
        'TT_NPE_SOURCE_MODEL_CONFIG_DIR="''${CMAKE_CURRENT_SOURCE_DIR}/data/device/models"' \
        'TT_NPE_SOURCE_MODEL_CONFIG_DIR="''${CMAKE_INSTALL_PREFIX}/${deviceModelDir}"'
  '';

  cmakeFlags = [
    (lib.cmakeBool "ENABLE_LIBCXX" false)
    (lib.cmakeBool "ENABLE_CCACHE" false)
    (lib.cmakeBool "TT_UNITY_BUILDS" false)
    "-DCMAKE_C_COMPILER=${stdenv.cc}/bin/clang"
    "-DCMAKE_CXX_COMPILER=${stdenv.cc}/bin/clang++"
    "-DPython_EXECUTABLE=${python3}/bin/python3"
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
  ];

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    export PYTHONDONTWRITEBYTECODE=1
    build_dir="$PWD"
    source_root="$(cd "$NIX_BUILD_TOP" && cd "$sourceRoot" && pwd)"
    # Upstream tests resolve some data files relative to the `tt_npe/`
    # subdirectory and expect the compiled-in model-config directory to exist,
    # which is only true once everything is installed.
    cd "$source_root/tt_npe"
    unset TT_NPE_DEVICE_MODEL_CONFIG_DIR
    "$build_dir/tt_npe/tt_npe_ut" --gtest_filter='-npePerfTest.*'

    export HOME="$TMPDIR"
    log="$("$out/bin/tt-npe" -d blackhole \
      -w "$out/share/tt-npe/examples/example_wl.json" 2>&1)"
    printf '%s\n' "$log"
    case "$log" in
      *"tt-npe simulation finished successfully"*) ;;
      *)
        echo "tt-npe did not report a successful blackhole simulation" >&2
        exit 1
        ;;
    esac
    runHook postInstallCheck
  '';

  postInstall = ''
    source_root="$(cd "$NIX_BUILD_TOP" && cd "$sourceRoot" && pwd)"

    # CMake only installs the top-level public headers; the device models under
    # cpp/include/device_models are part of the public API too.
    cp -r "$source_root"/tt_npe/cpp/include/. "$out/include/"

    # Python bindings and CLIs live in site-packages, not in lib/bin.
    install -d "$out/${sitePackages}"
    for bindings in "$out"/lib/tt_npe_pybind*.so; do
      mv "$bindings" "$out/${sitePackages}/"
    done
    for module in tt_npe.py npe_analyze_noc_trace_dir.py fabric_post_process.py \
                  programmatic_workload_generation.py; do
      mv "$out/bin/$module" "$out/${sitePackages}/$module"
    done
    # `mv` preserves the build-tree 0700 mode; installed modules must be readable.
    chmod 644 "$out/${sitePackages}"/*.py

    # Bundled example workload used by `passthru.tests.example-blackhole`.
    install -Dm644 "$source_root"/tt_npe/workload/example_wl.json \
      "$out/share/tt-npe/examples/example_wl.json"
    install -Dm644 "$source_root"/LICENSE "$out/share/licenses/tt-npe/LICENSE"

    makeWrapper ${python}/bin/python "$out/bin/tt-npe" \
      --prefix PYTHONPATH : "$out/${sitePackages}" \
      --set TT_NPE_DEVICE_MODEL_CONFIG_DIR "$out/${deviceModelDir}" \
      --add-flags "-m tt_npe"
    makeWrapper ${python}/bin/python "$out/bin/npe_analyze_noc_trace_dir" \
      --prefix PYTHONPATH : "$out/${sitePackages}" \
      --set TT_NPE_DEVICE_MODEL_CONFIG_DIR "$out/${deviceModelDir}" \
      --add-flags "-m npe_analyze_noc_trace_dir"

    # The bindings are outside lib/, so $ORIGIN/../lib from the build tree no
    # longer resolves. Append the real library directory to the fixup RPATH.
    for bindings in "$out/${sitePackages}"/tt_npe_pybind*.so; do
      patchelf --add-rpath "$out/lib" "$bindings"
    done
  '';

  passthru = {
    inherit sourceRevision;
    pythonPath = "${finalAttrs.finalPackage}/${sitePackages}";
    deviceModelConfigDir = "${finalAttrs.finalPackage}/${deviceModelDir}";
    tests.example-blackhole = callPackage ./check.nix {
      tt-npe = finalAttrs.finalPackage;
    };
  };

  meta = {
    description = "Lightweight Network-on-Chip performance estimator for Tenstorrent devices";
    homepage = "https://github.com/tenstorrent/tt-npe";
    license = lib.licenses.asl20;
    mainProgram = "tt-npe";
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
})
