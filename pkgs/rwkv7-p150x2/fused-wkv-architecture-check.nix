{
  lib,
  runCommand,
  writeText,
  tt-metal,
}:
let
  metaliumRoot = "${tt-metal}/libexec/tt-metalium";
  compiler = "${metaliumRoot}/runtime/sfpi/compiler/bin/riscv-tt-elf-g++";
  firmwareSource = "${metaliumRoot}/tt_metal/hw/firmware/src/tt-1xx/trisck.cc";
  kernelSource = writeText "rwkv7-fused-wkv-compute.cpp" (
    builtins.readFile ./kernel/fused_wkv_compute.cpp
  );
  descriptors = writeText "rwkv7-fused-wkv-architecture-check-descriptors.h" (
    builtins.readFile ./fused-wkv-architecture-check-descriptors.h
  );
  buildDirectory = "$PWD/build/blackhole/fused-wkv-compute";
  negativeBuildDirectory = "$PWD/build/unsupported/fused-wkv-compute";
  pcieNocX = 0;
  pcieNocY = 3;
  dispatchMessageAddress = 4290184248;
  mathProcessorIndex = 3;
  blackholeDramBankCount = 8;
  blackholeL1BankCount = 130;
  compileTimeArgs = lib.concatStringsSep "," (lib.replicate 20 "1");
  commonFlags = [
    "-O3"
    "-std=c++17"
    "-ftt-nttp"
    "-ftt-constinit"
    "-ftt-consteval"
    "-ftt-no-dyninit"
    "-flto=auto"
    "-ffast-math"
    "-fno-exceptions"
    "-fno-rtti"
    "-fno-use-cxa-atexit"
    "-g"
    "-MMD"
    "-Wall"
    "-Werror"
    "-Wno-error=deprecated-declarations"
    "-Wno-error=multistatement-macros"
    "-Wno-error=parentheses"
    "-Wno-error=unused-but-set-variable"
    "-Wno-unused-variable"
    "-Wno-unused-function"
  ];
  includeDirectories = [
    metaliumRoot
    "${metaliumRoot}/ttnn"
    "${metaliumRoot}/ttnn/cpp"
    "${metaliumRoot}/tt_metal"
    "${metaliumRoot}/tt_metal/hw/inc"
    "${metaliumRoot}/tt_metal/tt-llk/common"
    "${metaliumRoot}/tt_metal/hostdevcommon/api"
    "${metaliumRoot}/tt_metal/api"
    "${metaliumRoot}/runtime/sfpi/include"
    "${metaliumRoot}/tt_metal/hw/firmware/src/tt-1xx"
    "${metaliumRoot}/tt_metal/hw/ckernels/blackhole/metal/common"
    "${metaliumRoot}/tt_metal/hw/ckernels/blackhole/metal/llk_io"
    "${metaliumRoot}/tt_metal/hw/inc/internal/tt-1xx"
    "${metaliumRoot}/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    "${metaliumRoot}/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines"
    "${metaliumRoot}/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
    "${metaliumRoot}/tt_metal/tt-llk/tt_llk_blackhole/common/inc"
    "${metaliumRoot}/tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
    "${metaliumRoot}/tt_metal/hw/ckernels/blackhole/metal/llk_api"
    "${metaliumRoot}/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
  ];
  definitions = [
    "TENSIX_FIRMWARE"
    "LOCAL_MEM_EN=0"
    "PROCESSOR_INDEX=${toString mathProcessorIndex}"
    "UCK_CHLKC_MATH"
    "COMPILE_FOR_TRISC=1"
    "KERNEL_BUILD"
    "NOC_INDEX=0"
    "PCIE_NOC_X=${toString pcieNocX}"
    "PCIE_NOC_Y=${toString pcieNocY}"
    "DISPATCH_MESSAGE_ADDR=${toString dispatchMessageAddress}"
    ''FULL_KERNEL_NAME="rwkv7-fused-wkv-architecture-check/"''
    "KERNEL_COMPILE_TIME_ARGS=${compileTimeArgs}"
    "ARCH_BLACKHOLE"
    "NUM_DRAM_BANKS=${toString blackholeDramBankCount}"
    "NUM_L1_BANKS=${toString blackholeL1BankCount}"
    "IS_NOT_POW2_NUM_L1_BANKS=1"
  ];
  includeFlags = map (directory: "-I${directory}") includeDirectories;
  definitionFlags = map (definition: "-D${definition}") definitions;
  compileFlags = commonFlags ++ [ "-mcpu=tt-bh-tensix" ] ++ includeFlags ++ definitionFlags;
  scaffold = directory: ''
    mkdir -p "${directory}/trisc1"
    cp ${lib.escapeShellArg descriptors} "${directory}/chlkc_descriptors.h"
    printf '%s\n' '#define NOC_MODE 0' >"${directory}/defines_generated.h"
    printf '%s\n' \
      '#define TRISC_MATH' \
      '#include "defines_generated.h"' \
      '#include "${kernelSource}"' \
      >"${directory}/chlkc_math.cpp"
  '';
in
runCommand "rwkv7-fused-wkv-architecture-check" { } ''
  set -euo pipefail

  test -x ${lib.escapeShellArg compiler}
  test -f ${lib.escapeShellArg firmwareSource}
  test -f ${lib.escapeShellArg kernelSource}
  test -f ${lib.escapeShellArg descriptors}

  ${scaffold buildDirectory}
  mkdir -p "$out"
  (
    cd "${buildDirectory}/trisc1"
    ${lib.escapeShellArgs ([ compiler ] ++ compileFlags)} \
      -I. -I.. \
      -c ${lib.escapeShellArg firmwareSource} \
      -o "$out/fused-wkv-compute-blackhole.o"
  )
  test -s "$out/fused-wkv-compute-blackhole.o"

  invalid_kernel="$PWD/invalid-fused-wkv-compute.cpp"
  cp ${lib.escapeShellArg kernelSource} "$invalid_kernel"
  chmod u+w "$invalid_kernel"
  substituteInPlace "$invalid_kernel" \
    --replace-fail '#if !defined(ARCH_BLACKHOLE)' '#if defined(ARCH_BLACKHOLE)'

  mkdir -p "${negativeBuildDirectory}/trisc1"
  cp ${lib.escapeShellArg descriptors} "${negativeBuildDirectory}/chlkc_descriptors.h"
  printf '%s\n' '#define NOC_MODE 0' >"${negativeBuildDirectory}/defines_generated.h"
  printf '%s\n' \
    '#define TRISC_MATH' \
    '#include "defines_generated.h"' \
    "#include \"$invalid_kernel\"" \
    >"${negativeBuildDirectory}/chlkc_math.cpp"

  negative_log="$PWD/invalid-blackhole-fixture.log"
  if (
    cd "${negativeBuildDirectory}/trisc1"
    ${lib.escapeShellArgs ([ compiler ] ++ compileFlags)} \
      -I. -I.. \
      -c ${lib.escapeShellArg firmwareSource} \
      -o "$PWD/invalid-blackhole-fixture.o"
  ) >"$negative_log" 2>&1; then
    echo "fused WKV architecture check accepted its invalid Blackhole fixture" >&2
    exit 1
  fi
  grep -Fq 'fused RWKV-7 WKV compute supports Blackhole only' "$negative_log"
  test ! -e "${negativeBuildDirectory}/trisc1/invalid-blackhole-fixture.o"
''
