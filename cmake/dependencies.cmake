# External dependencies. Header-only ones are pulled via FetchContent so
# the project builds out of the box on a minimal Ubuntu 22.04 image.

# Eigen — dense linear algebra, header-only.
FetchContent_Declare(eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen3)

# nlohmann_json — drop-in JSON for the settings file (Cglobal).
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# libtiff — TIFF stack I/O. Prefer the system package; fall back to vendored
# if missing.
find_package(TIFF QUIET)
if(NOT TIFF_FOUND)
    message(STATUS "System libtiff not found — attempting FetchContent build")
    FetchContent_Declare(libtiff
        GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
        GIT_TAG        v4.6.0
        GIT_SHALLOW    TRUE)
    set(tiff-tools OFF CACHE BOOL "" FORCE)
    set(tiff-tests OFF CACHE BOOL "" FORCE)
    set(tiff-contrib OFF CACHE BOOL "" FORCE)
    set(tiff-docs OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libtiff)
    set(TIFF_LIBRARIES tiff)
    set(TIFF_INCLUDE_DIRS ${libtiff_SOURCE_DIR}/libtiff)
endif()

# GoogleTest — only fetched when tests are enabled.
if(VOX2TET_BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
