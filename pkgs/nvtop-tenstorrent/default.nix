{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  gtest,
  libdrm,
  ncurses,
  pkg-config,
  udev,
  addDriverRunpath,
}:
# Why this is a separate derivation rather than plain `prev.nvtop`:
#
# nvtop's Tenstorrent vendor backend (`src/extract_gpuinfo_tenstorrent.c`, added
# by commit 3982204 "Add support for Tenstorrent accelerators", 2026-03-22) is
# master-only: the newest release tag, 3.3.2 (2026-02-08), has no
# TENSTORRENT_SUPPORT option and none of the vendor code. So this package pins
# the master commit that carries the backend, observed as master HEAD on
# 2026-10-28, and enables TENSTORRENT_SUPPORT (upstream already defaults it ON
# for Linux; passing it explicitly keeps the build honest if that changes).
#
# Everything else mirrors `nvtopPackages.full` in nixpkgs: the same GPU families
# stay enabled so the binary is a drop-in replacement for the distro nvtop on a
# machine that also has an AMD/Intel/NVIDIA device, and the same build-time
# dependency set is needed for them.
#
# The Tenstorrent backend itself is sysfs-only (`/sys/class/tenstorrent/*` plus
# the per-device hwmon node and `/proc/driver/tenstorrent/*/pids`), so it needs
# no extra library beyond what the other families already pull in.
stdenv.mkDerivation {
  pname = "nvtop";
  # Master pin, not a release tag: `<upstream release>-unstable-<commit date>`.
  version = "3.3.2-unstable-2026-10-04";

  strictDeps = true;
  __structuredAttrs = true;

  src = fetchFromGitHub {
    owner = "Syllo";
    repo = "nvtop";
    rev = "71094a9f7dfbfe7cbe9faebfdf055e580a034283";
    hash = "sha256-yrB8Z097kqX8NeEMzw0FGlF/zc+Vg1W+4qVub8vsj2k=";
  };

  nativeBuildInputs = [
    cmake
    pkg-config
    addDriverRunpath
  ];

  buildInputs = [
    ncurses
    udev
    libdrm
    gtest
  ];

  cmakeFlags = [
    (lib.cmakeBool "BUILD_TESTING" true)
    (lib.cmakeBool "USE_LIBUDEV_OVER_LIBSYSTEMD" true)
    (lib.cmakeBool "AMDGPU_SUPPORT" true)
    (lib.cmakeBool "NVIDIA_SUPPORT" true)
    (lib.cmakeBool "INTEL_SUPPORT" true)
    (lib.cmakeBool "MSM_SUPPORT" true)
    (lib.cmakeBool "PANFROST_SUPPORT" true)
    (lib.cmakeBool "PANTHOR_SUPPORT" true)
    (lib.cmakeBool "V3D_SUPPORT" true)
    (lib.cmakeBool "ROCKCHIP_SUPPORT" true)
    (lib.cmakeBool "METAX_SUPPORT" true)
    (lib.cmakeBool "ENFLAME_SUPPORT" true)
    # The reason this package exists.
    (lib.cmakeBool "TENSTORRENT_SUPPORT" true)
    # Families with no packaged library; explicit OFF so a stray one in the
    # build environment cannot flip them on.
    (lib.cmakeBool "APPLE_SUPPORT" false)
    (lib.cmakeBool "ASCEND_SUPPORT" false)
    (lib.cmakeBool "TPU_SUPPORT" false)
    (lib.cmakeBool "IXML_SUPPORT" false)
  ];

  # libdrm installs <drm.h> under an include subdirectory that CMake's Libdrm
  # module does not add for every family.
  env.NIX_CFLAGS_COMPILE = "-isystem ${lib.getDev libdrm}/include/libdrm";

  # nvtop's RPATH-only install paths drop the ncurses/libdrm/udev closure, and
  # NVML is dlopen'd from the driver runpath rather than linked.
  postFixup = ''
    patchelf \
      --set-rpath "${
        lib.makeLibraryPath [
          libdrm
          ncurses
          udev
        ]
      }" \
      $out/bin/nvtop
    addDriverRunpath $out/bin/nvtop
  '';

  doCheck = true;
  doInstallCheck = true;

  installCheckPhase = ''
    runHook preInstallCheck
    $out/bin/nvtop --version | grep -F "nvtop version 3.3.2"
    runHook postInstallCheck
  '';

  meta = {
    description = "htop-like task monitor for AMD, Intel, NVIDIA and Tenstorrent accelerators";
    longDescription = ''
      Nvtop stands for Neat Videocard TOP, a (h)top like task monitor for GPUs
      and accelerators. This build tracks nvtop's master branch to include the
      Tenstorrent backend, which reads device telemetry from
      /sys/class/tenstorrent and its hwmon attributes.
    '';
    homepage = "https://github.com/Syllo/nvtop";
    license = lib.licenses.gpl3Only;
    platforms = lib.platforms.linux;
    maintainers = with lib.maintainers; [ gbtb ];
    mainProgram = "nvtop";
  };
}
