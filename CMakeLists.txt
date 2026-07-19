cmake_minimum_required(VERSION 3.20)
project(vox2tet
    VERSION 0.1.0
    DESCRIPTION "Voxel-to-tetrahedron meshing pipeline"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(VOX2TET_BUILD_TESTS "Build unit tests" ON)
option(VOX2TET_USE_OPENMP "Use OpenMP for parallelism" ON)

# Debug-only scipy/numpy reference bridge. When ON, the library exposes
# `vox2tet::debug_bridge::run(...)` that shells out to
# `tools/scipy_bridge.py` for cross-comparison against scipy. Keep OFF
# in release builds — this adds an external Python dependency.
option(VOX2TET_DEBUG_BRIDGE "Enable scipy/numpy reference bridge" OFF)
set(VOX2TET_PYTHON_BIN "" CACHE PATH "Python interpreter with numpy+scipy (for VOX2TET_DEBUG_BRIDGE)")

include(FetchContent)
include(cmake/dependencies.cmake)

# ------------------------------------------- third-party volume meshers
# CDT (Diazzi/Panozzo/Vaxman/Attene, SIGGRAPH Asia 2023): in-process
# Steiner constrained Delaunay tetrahedrization. Compiled as its own
# static library because it requires C++20, strict FP rounding
# semantics (-frounding-math) and its own SIMD flags — none of which
# should leak into the main C++17 library. src/tetmesh/steiner_cdt.cpp
# is our wrapper: the only TU that includes CDT headers; it exposes a
# plain-STL interface consumed by the C++17 runner.
#
# License: with VOX2TET_CDT_MAROT=OFF (default) CDT builds in its LGPL
# configuration, matching this project's LGPL-3.0-or-later. Switching
# the option ON enables the (faster) GPL-licensed Marot face-recovery
# method and makes the combined binary GPL — leave OFF unless that is
# acceptable.
option(VOX2TET_CDT_MAROT "Build CDT with the GPL-only Marot face-recovery method" OFF)
option(VOX2TET_CDT_AVX2 "Compile CDT with AVX2+FMA (build host must support them)" ON)
add_library(vox2tet_cdt STATIC
    third-party/CDT/src/giftWrap.cpp
    third-party/CDT/src/PLC.cpp
    third-party/CDT/src/delaunay.cpp
    src/tetmesh/steiner_cdt.cpp)
target_include_directories(vox2tet_cdt SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/CDT/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third-party/CDT/include)
target_include_directories(vox2tet_cdt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(vox2tet_cdt PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(vox2tet_cdt PRIVATE /fp:strict /Oi /D_CRT_SECURE_NO_WARNINGS)
    if(VOX2TET_CDT_AVX2)
        target_compile_options(vox2tet_cdt PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(vox2tet_cdt PRIVATE -O2 -frounding-math)
    if(VOX2TET_CDT_AVX2)
        target_compile_options(vox2tet_cdt PRIVATE -mavx2 -mfma)
    endif()
endif()
if(VOX2TET_CDT_MAROT)
    target_compile_definitions(vox2tet_cdt PRIVATE USE_MAROTS_METHOD)
endif()

# MMG3D (LGPL): in-process tetrahedral mesh optimizer, used to improve
# the quality of the CDT output while preserving material interfaces.
# Built as a subproject; only the static library is compiled.
set(BUILD "MMG3D" CACHE STRING "mmg: build only mmg3d" FORCE)
set(LIBMMG3D_STATIC ON  CACHE BOOL "" FORCE)
set(LIBMMG3D_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/mmg EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------- core library
file(GLOB_RECURSE VOX2TET_SOURCES CONFIGURE_DEPENDS
    src/core/*.cpp
    src/io/*.cpp
    src/image/*.cpp
    src/mesh/*.cpp
    src/marching_cubes/*.cpp
    src/brep/*.cpp
    src/remesh/*.cpp
    src/tetgen/*.cpp
    src/tetmesh/cdt_mmg_runner.cpp)

add_library(vox2tet ${VOX2TET_SOURCES})
target_include_directories(vox2tet
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vox2tet
    PUBLIC  Eigen3::Eigen nlohmann_json::nlohmann_json
    PRIVATE ${TIFF_LIBRARIES} vox2tet_cdt libmmg3d_a)
target_include_directories(vox2tet PRIVATE ${TIFF_INCLUDE_DIRS})

if(VOX2TET_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vox2tet PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(vox2tet PUBLIC VOX2TET_HAS_OPENMP)
    endif()
endif()

if(VOX2TET_DEBUG_BRIDGE)
    if(NOT VOX2TET_PYTHON_BIN)
        message(WARNING
            "VOX2TET_DEBUG_BRIDGE=ON but VOX2TET_PYTHON_BIN is unset — "
            "the bridge will compile but always report unavailable. "
            "Configure with -DVOX2TET_PYTHON_BIN=/path/to/python (numpy+scipy).")
    endif()
    # PUBLIC: consumers must see VOX2TET_DEBUG_BRIDGE too, otherwise the
    # header dispatches to the no-op inline stubs and available()/run()
    # silently report unavailable. The string paths can stay PRIVATE.
    target_compile_definitions(vox2tet PUBLIC  VOX2TET_DEBUG_BRIDGE)
    target_compile_definitions(vox2tet PRIVATE
        VOX2TET_DEBUG_PYTHON_BIN="${VOX2TET_PYTHON_BIN}"
        VOX2TET_DEBUG_BRIDGE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/tools/scipy_bridge.py")
endif()

# Pass install-prefix for shipped LUT binaries
target_compile_definitions(vox2tet PRIVATE
    VOX2TET_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

if(MSVC)
    target_compile_options(vox2tet PRIVATE /W4 /permissive-)
else()
    target_compile_options(vox2tet PRIVATE -Wall -Wextra -Wpedantic
        -Wno-unused-parameter -Wno-unused-function)
endif()

# ------------------------------------------------------------------- entry app
# Thin CLI wrapper over the library: parses args into Settings, calls
# vox2tet::generate(). Also serves as the canonical "how to embed
# libvox2tet" example — see src/app/run_vox2tet.cpp.
add_executable(run_vox2tet src/app/run_vox2tet.cpp)
target_link_libraries(run_vox2tet PRIVATE vox2tet)

# ------------------------------------------------------------------- install
# `cmake --install build/` lays the library, headers, and executables out
# under <prefix>/{lib,include,bin}. A full `find_package(vox2tet)`
# integration is intentionally omitted — Eigen and nlohmann_json arrive
# via FetchContent and would need to be re-exported, which complicates
# downstream consumers. Instead the README documents the in-tree build
# pattern (cmake/add_subdirectory or direct linking against libvox2tet).
include(GNUInstallDirs)

install(TARGETS vox2tet
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS run_vox2tet
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/vox2tet
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ----------------------------------------------------------------------- tests
if(VOX2TET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
