cmake_minimum_required(VERSION 3.22)
project(rwkv7_p150x2 LANGUAGES CXX)

option(
  RWKV7_FETCH_QUEUE_DIAGNOSTICS
  "Build with diagnostic TT-Metal fetch-queue counters"
  OFF
)
option(
  RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY
  "Build a diagnostic fused WKV with the generic BF16 readout boundary"
  OFF
)

find_package(PkgConfig REQUIRED)
find_package(enchantum REQUIRED)
find_package(TT-NN REQUIRED)
find_package(nlohmann_json REQUIRED)
pkg_check_modules(BLAKE3 REQUIRED IMPORTED_TARGET libblake3)

foreach(required_path IN ITEMS RWKV7_CHECKPOINT_PATH RWKV7_MESH_DESCRIPTOR_PATH RWKV7_SINGLE_CARD_MESH_DESCRIPTOR_PATH RWKV7_METALIUM_ROOT RWKV7_PERSISTENT_DECODE_PROFILE_PATH RWKV7_MAXIMUM_GENERATED_TOKENS)
  if(NOT DEFINED ${required_path})
    message(FATAL_ERROR "${required_path} is required")
  endif()
endforeach()
configure_file(config.hpp.in generated/config.hpp @ONLY)

add_executable(
  rwkv7-p150x2-runtime
  src/fused_wkv_program.cpp
  src/main.cpp
  src/persistent_decode.cpp
  src/persistent_decode_self_test.cpp
)
if(RWKV7_FETCH_QUEUE_DIAGNOSTICS)
  target_sources(
    rwkv7-p150x2-runtime
    PRIVATE
      src/hardware_counter_diagnostics.cpp
  )
endif()
target_compile_features(rwkv7-p150x2-runtime PRIVATE cxx_std_20)
target_compile_definitions(
  rwkv7-p150x2-runtime
  PRIVATE
    $<$<BOOL:${RWKV7_FETCH_QUEUE_DIAGNOSTICS}>:RWKV7_FETCH_QUEUE_DIAGNOSTICS_AVAILABLE=1>
)
target_include_directories(
  rwkv7-p150x2-runtime
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_options(
  rwkv7-p150x2-runtime
  PRIVATE
    -Wall
    -Wextra
    -Wconversion
    -Werror
)
target_link_libraries(
  rwkv7-p150x2-runtime
  PRIVATE
    TTNN::TTNN
    nlohmann_json::nlohmann_json
    PkgConfig::BLAKE3
)

install(TARGETS rwkv7-p150x2-runtime RUNTIME DESTINATION libexec/rwkv7-p150x2)
install(
  DIRECTORY kernel/
  DESTINATION share/rwkv7-p150x2/kernel
  FILES_MATCHING PATTERN "*.cpp"
)
