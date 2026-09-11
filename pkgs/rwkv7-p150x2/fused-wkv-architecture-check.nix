{
  lib,
  runCommand,
  writeText,
  tt-metal,
}:
let
  metaliumRoot = "${tt-metal}/libexec/tt-metalium";
  compiler = "${metaliumRoot}/runtime/sfpi/compiler/bin/riscv-tt-elf-g++";
  computeFirmware = "${metaliumRoot}/tt_metal/hw/firmware/src/tt-1xx/trisck.cc";
  readerFirmware = "${metaliumRoot}/tt_metal/hw/firmware/src/tt-1xx/brisck.cc";
  writerFirmware = "${metaliumRoot}/tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc";
  computeSource = writeText "rwkv7-fused-wkv-compute.cpp" (
    builtins.readFile ./kernel/fused_wkv_compute.cpp
  );
  readerSource = writeText "rwkv7-fused-wkv-reader.cpp" (
    builtins.readFile ./kernel/fused_wkv_reader.cpp
  );
  writerSource = writeText "rwkv7-fused-wkv-writer.cpp" (
    builtins.readFile ./kernel/fused_wkv_writer.cpp
  );
  descriptors = writeText "rwkv7-fused-wkv-architecture-check-descriptors.h" (
    builtins.readFile ./fused-wkv-architecture-check-descriptors.h
  );
  computeBuildDirectory = "$PWD/build/blackhole/fused-wkv-compute";
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
  architectureDefinitions = [
    "ARCH_BLACKHOLE"
    "NUM_DRAM_BANKS=${toString blackholeDramBankCount}"
    "NUM_L1_BANKS=${toString blackholeL1BankCount}"
    "IS_NOT_POW2_NUM_L1_BANKS=1"
  ];
  computeDefinitions = [
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
  ]
  ++ architectureDefinitions;
  includeFlags = map (directory: "-I${directory}") includeDirectories;
  definitionFlags = definitions: map (definition: "-D${definition}") definitions;
  computeFlags =
    commonFlags ++ [ "-mcpu=tt-bh-tensix" ] ++ includeFlags ++ definitionFlags computeDefinitions;
  computeScaffold =
    {
      directory,
      source,
    }:
    ''
      mkdir -p "${directory}/trisc1"
      cp ${lib.escapeShellArg descriptors} "${directory}/chlkc_descriptors.h"
      printf '%s\n' '#define NOC_MODE 0' >"${directory}/defines_generated.h"
      printf '%s\n' \
        '#define TRISC_MATH' \
        '#include "defines_generated.h"' \
        '#include "${source}"' \
        >"${directory}/chlkc_math.cpp"
    '';
  dataMovementSpecs = [
    {
      name = "reader";
      source = readerSource;
      firmware = readerFirmware;
      processorIndex = 0;
      nocIndex = 0;
      processorDefinition = "COMPILE_FOR_BRISC";
    }
    {
      name = "writer";
      source = writerSource;
      firmware = writerFirmware;
      processorIndex = 1;
      nocIndex = 1;
      processorDefinition = "COMPILE_FOR_NCRISC";
    }
  ];
  dataMovementFlags =
    spec:
    commonFlags
    ++ [ "-mcpu=tt-bh-tensix" ]
    ++ includeFlags
    ++ definitionFlags (
      architectureDefinitions
      ++ [
        "TENSIX_FIRMWARE"
        "LOCAL_MEM_EN=0"
        "PCIE_NOC_X=${toString pcieNocX}"
        "PCIE_NOC_Y=${toString pcieNocY}"
        "DISPATCH_MESSAGE_ADDR=${toString dispatchMessageAddress}"
        "PROCESSOR_INDEX=${toString spec.processorIndex}"
        spec.processorDefinition
        "NOC_INDEX=${toString spec.nocIndex}"
        "NOC_MODE=0"
        "KERNEL_BUILD"
        ''FULL_KERNEL_NAME="rwkv7-fused-wkv-${spec.name}-architecture-check/"''
        "KERNEL_COMPILE_TIME_ARGS=${compileTimeArgs}"
        "LOG_BASE_2_OF_NUM_DRAM_BANKS=3"
      ]
    );
  compileDataMovement =
    spec:
    let
      directory = "$PWD/build/blackhole/fused-wkv-${spec.name}";
    in
    ''
      mkdir -p "${directory}"
      cp ${lib.escapeShellArg descriptors} "${directory}/chlkc_descriptors.h"
      printf '%s\n' '#define NOC_MODE ${toString spec.nocIndex}' >"${directory}/defines_generated.h"
      printf '%s\n' '#include "${spec.source}"' >"${directory}/kernel_includes.hpp"
      (
        cd "${directory}"
        ${lib.escapeShellArgs ([ compiler ] ++ dataMovementFlags spec)} \
          -I. \
          -c ${lib.escapeShellArg spec.firmware} \
          -o "$out/fused-wkv-${spec.name}-blackhole.o"
      )
      test -s "$out/fused-wkv-${spec.name}-blackhole.o"
    '';
in
runCommand "rwkv7-fused-wkv-architecture-check" { } ''
  set -euo pipefail

  test -x ${lib.escapeShellArg compiler}
  test -f ${lib.escapeShellArg computeFirmware}
  test -f ${lib.escapeShellArg readerFirmware}
  test -f ${lib.escapeShellArg writerFirmware}
  test -f ${lib.escapeShellArg computeSource}
  test -f ${lib.escapeShellArg readerSource}
  test -f ${lib.escapeShellArg writerSource}
  test -f ${lib.escapeShellArg descriptors}

  ${computeScaffold {
    directory = computeBuildDirectory;
    source = computeSource;
  }}
  mkdir -p "$out"
  (
    cd "${computeBuildDirectory}/trisc1"
    ${lib.escapeShellArgs ([ compiler ] ++ computeFlags)} \
      -I. -I.. \
      -c ${lib.escapeShellArg computeFirmware} \
      -o "$out/fused-wkv-compute-blackhole.o"
  )
  test -s "$out/fused-wkv-compute-blackhole.o"

  ${lib.concatMapStringsSep "\n" compileDataMovement dataMovementSpecs}

  invalid_kernel="$PWD/invalid-fused-wkv-compute.cpp"
  cp ${lib.escapeShellArg computeSource} "$invalid_kernel"
  chmod u+w "$invalid_kernel"
  substituteInPlace "$invalid_kernel" \
    --replace-fail '#if !defined(ARCH_BLACKHOLE)' '#if defined(ARCH_BLACKHOLE)'

  ${computeScaffold {
    directory = negativeBuildDirectory;
    source = "$invalid_kernel";
  }}
  substituteInPlace "${negativeBuildDirectory}/chlkc_math.cpp" \
    --replace-fail '#include "$invalid_kernel"' "#include \"$invalid_kernel\""

  negative_log="$PWD/invalid-blackhole-fixture.log"
  if (
    cd "${negativeBuildDirectory}/trisc1"
    ${lib.escapeShellArgs ([ compiler ] ++ computeFlags)} \
      -I. -I.. \
      -c ${lib.escapeShellArg computeFirmware} \
      -o "$PWD/invalid-blackhole-fixture.o"
  ) >"$negative_log" 2>&1; then
    echo "fused WKV architecture check accepted its invalid Blackhole fixture" >&2
    exit 1
  fi
  grep -Fq 'fused RWKV-7 WKV compute supports Blackhole only' "$negative_log"
  test ! -e "${negativeBuildDirectory}/trisc1/invalid-blackhole-fixture.o"
''
