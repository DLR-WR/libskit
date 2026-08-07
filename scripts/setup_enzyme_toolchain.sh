#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENZYME_PREFIX="${TMPDIR:-/tmp}/libskit-enzyme-${USER:-user}"
ENZYME_PREFIX="${ENZYME_PREFIX:-$DEFAULT_ENZYME_PREFIX}"
ENZYME_REPO="${ENZYME_REPO:-https://github.com/EnzymeAD/Enzyme.git}"
ENZYME_GIT_REF="${ENZYME_GIT_REF:-v0.0.256}"

install_with_apt() {
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        clang \
        llvm-dev \
        libclang-dev \
        cmake \
        ninja-build \
        git \
        ca-certificates
}

install_with_dnf() {
    dnf install -y \
        clang \
        clang-devel \
        llvm-devel \
        cmake \
        ninja-build \
        git \
        ca-certificates
}

install_with_yum() {
    yum install -y \
        clang \
        clang-devel \
        llvm-devel \
        cmake \
        ninja-build \
        git \
        ca-certificates
}

find_llvm_config() {
    if command -v llvm-config >/dev/null 2>&1; then
        command -v llvm-config
        return
    fi

    # Collect versioned binaries, prefer the highest version number so that
    # the selected llvm-config matches the clang/libclangBasic.a that is installed.
    local best=""
    local best_ver=0
    for candidate in /usr/bin/llvm-config-* /usr/local/bin/llvm-config-*; do
        if [[ -x "$candidate" ]]; then
            local ver
            ver=$(echo "$candidate" | grep -o '[0-9]*$')
            if [[ "$ver" -gt "$best_ver" ]]; then
                best_ver="$ver"
                best="$candidate"
            fi
        fi
    done
    if [[ -n "$best" ]]; then
        echo "$best"
        return
    fi

    return 1
}

clang_dev_present() {
    # Check that Clang static development libs are installed (needed by ClangConfig.cmake)
    local llvm_cfg
    llvm_cfg="$(find_llvm_config 2>/dev/null)" || return 1
    local prefix
    prefix="$("$llvm_cfg" --prefix 2>/dev/null)" || return 1
    [[ -f "$prefix/lib/libclangBasic.a" ]]
}

ensure_system_dependencies() {
    if command -v clang >/dev/null 2>&1 && find_llvm_config >/dev/null 2>&1 && clang_dev_present; then
        return
    fi

    if command -v apt-get >/dev/null 2>&1; then
        install_with_apt
        return
    fi

    if command -v dnf >/dev/null 2>&1; then
        install_with_dnf
        return
    fi

    if command -v yum >/dev/null 2>&1; then
        install_with_yum
        return
    fi

    echo "Unsupported package manager. Install clang, llvm-config, cmake, ninja, and git manually." >&2
    exit 1
}

ensure_system_dependencies

LLVM_CONFIG="$(find_llvm_config)"
LLVM_CMAKE_DIR="$($LLVM_CONFIG --cmakedir)"
LLVM_MAJOR="$($LLVM_CONFIG --version | cut -d. -f1)"
# Derive Clang cmake dir from LLVM cmake dir (sibling 'clang' directory)
CLANG_CMAKE_DIR="${LLVM_CMAKE_DIR%/llvm}/clang"
PLUGIN_DIR="$ENZYME_PREFIX/build/Enzyme"
# macOS emits .dylib, Linux emits .so; the symlink used by callers is always .so.
case "$(uname -s)" in
    Darwin) PLUGIN_EXT="dylib" ;;
    *)      PLUGIN_EXT="so" ;;
esac
PLUGIN_REAL="$PLUGIN_DIR/ClangEnzyme-${LLVM_MAJOR}.${PLUGIN_EXT}"
PLUGIN_LINK="$ENZYME_PREFIX/ClangEnzyme.so"

mkdir -p "$ENZYME_PREFIX"

if [[ ! -f "$PLUGIN_REAL" ]]; then
    rm -rf "$ENZYME_PREFIX/src" "$ENZYME_PREFIX/build"
    git clone --depth 1 --branch "$ENZYME_GIT_REF" "$ENZYME_REPO" "$ENZYME_PREFIX/src"
    cmake_args=(
        -S "$ENZYME_PREFIX/src/enzyme"
        -B "$ENZYME_PREFIX/build"
        -G Ninja
        -DLLVM_DIR="$LLVM_CMAKE_DIR"
    )
    if [[ -d "$CLANG_CMAKE_DIR" ]]; then
        cmake_args+=(-DClang_DIR="$CLANG_CMAKE_DIR")
    else
        echo "Warning: Clang cmake dir not found at $CLANG_CMAKE_DIR; ClangEnzyme plugin may not build" >&2
    fi
    cmake "${cmake_args[@]}"
    cmake --build "$ENZYME_PREFIX/build" --target "ClangEnzyme-${LLVM_MAJOR}" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

if [[ ! -f "$PLUGIN_REAL" ]]; then
    shopt -s nullglob
    plugin_candidates=("$PLUGIN_DIR"/ClangEnzyme-*.so "$PLUGIN_DIR"/ClangEnzyme-*.dylib)
    shopt -u nullglob
    if [[ ${#plugin_candidates[@]} -eq 0 ]]; then
        echo "Enzyme build completed, but no ClangEnzyme plugin was produced in $PLUGIN_DIR" >&2
        exit 1
    fi
    PLUGIN_REAL="${plugin_candidates[0]}"
fi

ln -sf "$PLUGIN_REAL" "$PLUGIN_LINK"

# Resolve versioned clang/clang++ binaries matching the detected LLVM version
CLANG_BIN="$(command -v "clang-${LLVM_MAJOR}" 2>/dev/null || command -v clang)"
CLANGXX_BIN="$(command -v "clang++-${LLVM_MAJOR}" 2>/dev/null || command -v clang++)"

# Write a sourceable env file so callers can activate the toolchain with one line:
#   source "$ENZYME_PREFIX/env.sh"
cat > "$ENZYME_PREFIX/env.sh" <<EOF
export CC=${CLANG_BIN}
export CXX=${CLANGXX_BIN}
export ENZYME_PLUGIN_PATH=${PLUGIN_LINK}
EOF
