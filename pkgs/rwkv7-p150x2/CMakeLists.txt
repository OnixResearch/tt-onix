cmake_minimum_required(VERSION 3.22)
project(rwkv7_p150x2 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
find_package(enchantum REQUIRED)
find_package(TT-NN REQUIRED)
find_package(nlohmann_json REQUIRED)
pkg_check_modules(BLAKE3 REQUIRED IMPORTED_TARGET libblake3)

foreach(required_path IN ITEMS RWKV7_CHECKPOINT_PATH RWKV7_MESH_DESCRIPTOR_PATH RWKV7_SINGLE_CARD_MESH_DESCRIPTOR_PATH RWKV7_METALIUM_ROOT)
  if(NOT DEFINED ${required_path})
    message(FATAL_ERROR "${required_path} is required")
  endif()
endforeach()
configure_file(config.hpp.in generated/config.hpp @ONLY)

add_executable(rwkv7-p150x2-runtime src/main.cpp)
target_compile_features(rwkv7-p150x2-runtime PRIVATE cxx_std_20)
target_include_directories(rwkv7-p150x2-runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
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
