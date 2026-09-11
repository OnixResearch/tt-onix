# Minimal host-side Tracy CLI tools (tracy-capture, tracy-csvexport) built from
# the package-matched tenstorrent/tracy fork vendored inside the tt-metal source
# tree. This replaces tt-metal's own csvexport/capture subprojects, whose
# vendor.cmake pulls a GUI/web dependency stack (ImGui, GLFW, emsdk) that the
# Nix tt-metal package intentionally disables.
#
# Only the dependencies the two CLI tools actually need are wired here:
#   - TracyGetOpt / TracyDtl: vendored in the fork itself
#   - TracyCapstone: system capstone via pkg-config (API-compatible with the
#     pinned 6.0.0-Alpha1 download; only stable cs_open/cs_disasm/detail APIs)
#   - libzstd: system zstd via pkg-config (server.cmake links target "libzstd")
#   - PPQSort: header-only, patched with the fork's own nodebug/semaphore patches
cmake_minimum_required(VERSION 3.20)
project(tt-tracy-cli-tools LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(tracy_root "" CACHE PATH "tenstorrent/tracy fork root inside the tt-metal source tree")
set(ppqsort_root "" CACHE PATH "patched PPQSort header root")
set(git_ref "package-matched" CACHE STRING "GitRef reported by the tools")
set(TRACY_ROOT ${tracy_root})
set(PPQSORT_ROOT ${ppqsort_root})
set(GIT_REF ${git_ref})

foreach(_required_var TRACY_ROOT PPQSORT_ROOT)
    if("${${_required_var}}" STREQUAL "")
        message(FATAL_ERROR "${_required_var} must be provided")
    endif()
endforeach()

include(${TRACY_ROOT}/cmake/version.cmake)

add_library(TracyGetOpt STATIC ${TRACY_ROOT}/getopt/getopt.c)
target_include_directories(TracyGetOpt PUBLIC ${TRACY_ROOT}/getopt)

add_library(TracyDtl INTERFACE)
target_include_directories(TracyDtl INTERFACE ${TRACY_ROOT}/dtl)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CAPSTONE REQUIRED IMPORTED_TARGET capstone)
add_library(TracyCapstone ALIAS PkgConfig::CAPSTONE)

pkg_check_modules(LIBZSTD REQUIRED IMPORTED_TARGET libzstd)
add_library(libzstd ALIAS PkgConfig::LIBZSTD)

add_library(PPQSort INTERFACE)
target_include_directories(PPQSort INTERFACE ${PPQSORT_ROOT}/include)
add_library(PPQSort::PPQSort ALIAS PPQSort)

set(TRACY_COMMON_DIR ${TRACY_ROOT}/public/common)
set(TRACY_SERVER_DIR ${TRACY_ROOT}/server)

set(TRACY_COMMON_SOURCES
    tracy_lz4.cpp
    tracy_lz4hc.cpp
    TracySocket.cpp
    TracyStackFrames.cpp
    TracySystem.cpp
)
list(TRANSFORM TRACY_COMMON_SOURCES PREPEND "${TRACY_COMMON_DIR}/")

set(TRACY_SERVER_SOURCES
    TracyBroadcast.cpp
    TracyMemory.cpp
    TracyMmap.cpp
    TracyPrint.cpp
    TracySysUtil.cpp
    TracyTaskDispatch.cpp
    TracyTextureCompression.cpp
    TracyThreadCompress.cpp
    TracyWorker.cpp
)
list(TRANSFORM TRACY_SERVER_SOURCES PREPEND "${TRACY_SERVER_DIR}/")

add_library(TracyServer STATIC ${TRACY_COMMON_SOURCES} ${TRACY_SERVER_SOURCES})
target_include_directories(TracyServer PUBLIC
    ${TRACY_COMMON_DIR}
    ${TRACY_SERVER_DIR}
)
target_link_libraries(TracyServer PUBLIC TracyCapstone libzstd PPQSort::PPQSort)

set(GITREF_DIR ${CMAKE_BINARY_DIR}/gitref)
file(MAKE_DIRECTORY ${GITREF_DIR})
file(WRITE ${GITREF_DIR}/GitRef.hpp
    "#pragma once\nnamespace tracy { static inline const char* GitRef = \"${GIT_REF}\"; }\n"
)

add_executable(tracy-csvexport ${TRACY_ROOT}/csvexport/src/csvexport.cpp)
target_include_directories(tracy-csvexport PRIVATE ${GITREF_DIR})
target_link_libraries(tracy-csvexport PRIVATE TracyServer TracyGetOpt)

add_executable(tracy-capture
    ${TRACY_ROOT}/capture/src/capture.cpp
    ${TRACY_ROOT}/capture/src/CaptureOutput.cpp
)
target_include_directories(tracy-capture PRIVATE
    ${TRACY_ROOT}/capture/src
    ${GITREF_DIR}
)
target_link_libraries(tracy-capture PRIVATE TracyServer TracyGetOpt)

install(TARGETS tracy-csvexport tracy-capture RUNTIME DESTINATION bin)
