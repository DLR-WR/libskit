cmake_minimum_required(VERSION 3.15)

# ── Auto-select Clang before project() locks the compiler choice ─────────────
# Enzyme AD requires Clang.  Priority order:
#   1. $ENV{CXX} / $ENV{CC}  (set by CI env.sh or user)
#   2. Already-cached value if it points to clang
#   3. First versioned clang found on PATH
# We FORCE the override when the cached value is not Clang so that a stale
# CMakeCache.txt pointing at g++ is corrected automatically.
macro(_libskit_prefer_clang lang names cache_var)
    if(NOT "$ENV{${lang}}" STREQUAL "")
        set(${cache_var} "$ENV{${lang}}" CACHE FILEPATH "${lang} compiler" FORCE)
    elseif(NOT "${${cache_var}}" MATCHES "[Cc][Ll][Aa][Nn][Gg]")
        find_program(_found_clang NAMES ${names})
        if(_found_clang)
            set(${cache_var} "${_found_clang}" CACHE FILEPATH "${lang} compiler" FORCE)
        endif()
        unset(_found_clang CACHE)
    endif()
endmacro()

_libskit_prefer_clang(CXX "clang++-19;clang++-18;clang++-17;clang++-16;clang++-15;clang++" CMAKE_CXX_COMPILER)
_libskit_prefer_clang(CC  "clang-19;clang-18;clang-17;clang-16;clang-15;clang"             CMAKE_C_COMPILER)

# ── Enzyme plugin path ────────────────────────────────────────────────────────
# Read from environment first; fall back to the well-known symlink written by
# scripts/setup_enzyme_toolchain.sh.
if(NOT "$ENV{ENZYME_PLUGIN_PATH}" STREQUAL "")
    set(ENZYME_PLUGIN_PATH "$ENV{ENZYME_PLUGIN_PATH}" CACHE FILEPATH "Path to the ClangEnzyme plugin shared library" FORCE)
else()
    set(ENZYME_PLUGIN_PATH "" CACHE FILEPATH "Path to the ClangEnzyme plugin shared library")
endif()

if(NOT ENZYME_PLUGIN_PATH)
    if(EXISTS "/tmp/libskit-enzyme/ClangEnzyme.so")
        set(ENZYME_PLUGIN_PATH "/tmp/libskit-enzyme/ClangEnzyme.so" CACHE FILEPATH "Path to the ClangEnzyme plugin shared library" FORCE)
        message(STATUS "Auto-detected Enzyme plugin: ${ENZYME_PLUGIN_PATH}")
    endif()
endif()

############################
# project name and version #
############################
project( ${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "libskit requires a Clang-based C++ compiler for Enzyme AD. "
        "Found: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})\n"
        "Set CXX=clang++ (or a versioned clang++ binary) before running pip install, "
        "or run scripts/setup_enzyme_toolchain.sh which writes an env.sh you can source."
    )
endif()

if(NOT ENZYME_PLUGIN_PATH)
    message(FATAL_ERROR
        "ENZYME_PLUGIN_PATH is required. Build Enzyme and point this cache variable "
        "to ClangEnzyme-<llvm-major>.so."
    )
endif()

if(NOT EXISTS "${ENZYME_PLUGIN_PATH}")
    message(FATAL_ERROR "ENZYME_PLUGIN_PATH does not exist: ${ENZYME_PLUGIN_PATH}")
endif()

set(ENZYME_PLUGIN_COMPILE_FLAGS
    "-fplugin=${ENZYME_PLUGIN_PATH} -O2 -mllvm -enzyme-globals-default-inactive=1"
)
set(ENZYME_PLUGIN_COMPILE_OPTIONS
    "SHELL:-fplugin=${ENZYME_PLUGIN_PATH}"
    -O2
    "SHELL:-mllvm -enzyme-globals-default-inactive=1"
)
message(STATUS "Enzyme plugin: ${ENZYME_PLUGIN_PATH}")

##############################
# set c++ flags and standard #
##############################

# Set compiler-specific flags
if(MSVC)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    add_compile_options(/W0)
    add_compile_options(/O2)
    add_definitions(-D_USE_MATH_DEFINES)  # Enable M_PI on Windows
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    # GCC/Clang compiler flags
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic -DNDEBUG")
    #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -flto=auto")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -pg -O0 -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O2 -pg -g -DNDEBUG")
endif()




##########################
# find required packages #
##########################

# Try to find Eigen3 on the system first, otherwise download it
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system, downloading via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    # Eigen3 is header-only, just make it available
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
# When built through scikit-build-core, Python3_EXECUTABLE points to the isolated
# PEP 517 build environment. Otherwise it falls back to the system python3 on PATH.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ── Optional: source-based coverage instrumentation (Clang/GCC) ──────────────
# Enable with -DLIBSKIT_ENABLE_COVERAGE=ON.  Adds `--coverage` to libskit and
# the gtest executables so running ctest emits .gcda/.gcno files which
# `gcovr` (or `lcov --gcov-tool llvm-cov-wrapper`) can turn into a report.
# Must be set BEFORE any target is created so the flags propagate.
option(LIBSKIT_ENABLE_COVERAGE "Instrument C++ sources for gcov-style coverage" OFF)
if(LIBSKIT_ENABLE_COVERAGE)
    message(STATUS "C++ coverage instrumentation: ENABLED (--coverage)")
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

######################
# add python library #
######################
pybind11_add_module(_libskit src/_libskit/_libskit.cpp)
target_link_libraries(_libskit PRIVATE libskit Eigen3::Eigen)
install(TARGETS _libskit DESTINATION libskit)

# ── Generate pybind11 type stubs from the compiled extension ──────────────────
# pybind11-stubgen imports the module by name, so we set PYTHONPATH to the build
# dir and pass the bare module name (not the .so path).
#
# Stubgen flags explained:
#   --numpy-array-wrap-with-annotated  : Reformats numpy.ndarray[numpy.float64[m, n]]
#                                         to typing.Annotated format, fixing "Can't
#                                         find/import 'm'" errors from Eigen types
#   --enum-class-locations             : Maps C++ enum to Python path for proper
#                                         enum value resolution (fixes NormalizationType)
#   --ignore-invalid-expressions       : Silences C++ types like "libskit::PlasmaModel<double>"
#                                         that appear in auto-generated docstrings
#   --print-invalid-expressions-as-is  : Renders valid Python exprs that happen to look
#                                         like invalid ones instead of replacing with ...
# NOTE: --exit-code is intentionally omitted. Inside cibuildwheel's isolated
# build env, pybind11-stubgen can fail to resolve `numpy.typing.NDArray`
# annotations even though numpy is installed; with --exit-code that turns
# stubgen warnings into wheel-build failures. Without it, the .pyi still
# generates (with `...` fallbacks for unresolved annotations) and the wheel
# builds successfully.
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_libskit.pyi"
    COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH="${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
        "${Python3_EXECUTABLE}" -m pybind11_stubgen
        _libskit
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}"
        --numpy-array-wrap-with-annotated
        --enum-class-locations "NormalizationType:_libskit.NormalizationType"
        --ignore-invalid-expressions "libskit::|\\b[mMnN]\\b"
        --print-invalid-expressions-as-is
    DEPENDS _libskit
    COMMENT "Generating pybind11 stubs for _libskit"
    VERBATIM
)

# Make stub generation part of the default build so it ships in the wheel.
add_custom_target(_libskit_stubs ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/_libskit.pyi")

# Install the stub alongside the .so so type checkers find it.
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/_libskit.pyi"
    DESTINATION libskit
)

# Bundle atomic databases into the wheel so users don't need to manage them
install(FILES
    databases/lines_processed_all.csv
    databases/ionization_energies_processed.csv
    databases/levels_processed.csv
    databases/nist_asd_lines_vacuum.csv
    DESTINATION libskit/data
)


# include directories
include_directories(src/_libskit/include src/_libskit/)
add_subdirectory(src/_libskit)

# Testing
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()