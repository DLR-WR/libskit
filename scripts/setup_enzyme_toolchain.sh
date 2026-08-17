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
    # Check that Clang development files are installed (needed by ClangConfig.cmake)
    local prefix
    local llvm_cfg
    llvm_cfg="$(find_llvm_config 2>/dev/null)"
    if [[ -n "$llvm_cfg" ]]; then
        prefix="$("$llvm_cfg" --prefix 2>/dev/null)" || return 1
    else
        # Windows LLVM ships no llvm-config; derive prefix from clang location.
        local clang_bin
        clang_bin="$(command -v clang 2>/dev/null)" || return 1
        prefix="$(dirname "$(dirname "$clang_bin")")" 
    fi
    # Linux/macOS: static .a libs; Windows: .lib import libs or cmake configs.
    [[ -f "$prefix/lib/libclangBasic.a" || \
       -f "$prefix/lib/clangBasic.lib" || \
       -f "$prefix/lib/cmake/clang/ClangConfig.cmake" ]]
}

ensure_system_dependencies() {
    # git-bash's $PATH does not always inherit the LLVM bin dir added via
    # $GITHUB_PATH on Windows runners; probe the standard install location.
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            # install-llvm-action provides a valid toolchain; cygpath converts the
            # Windows path so bash can use it, then trust it without further checks.
            if [[ -n "${LLVM_PATH:-}" ]]; then
                _llvm_posix=$(cygpath -u "${LLVM_PATH}" 2>/dev/null || echo "${LLVM_PATH}")
                [[ ":$PATH:" != *":${_llvm_posix}/bin:"* ]] && export PATH="${_llvm_posix}/bin:$PATH"
                return
            fi
            for llvm_bin in "/c/Program Files/LLVM/bin" "/c/Program Files (x86)/LLVM/bin"; do
                if [[ -x "$llvm_bin/clang.exe" && ":$PATH:" != *":$llvm_bin:"* ]]; then
                    export PATH="$llvm_bin:$PATH"
                fi
            done
            # The official LLVM Windows binary release does NOT ship
            # llvm-config.exe. Skip that specific check; the rest of the script
            # has a Windows branch that derives LLVM_MAJOR/LLVM_CMAKE_DIR
            # directly from clang.
            if command -v clang >/dev/null 2>&1 && clang_dev_present; then
                return
            fi
            ;;
        *)
            if command -v clang >/dev/null 2>&1 && find_llvm_config >/dev/null 2>&1 && clang_dev_present; then
                return
            fi
            ;;
    esac

    # Debug hint: dump what we found so far to help diagnose CI failures.
    echo "system-deps check failed:" >&2
    echo "  clang         = $(command -v clang || echo '<not found>')" >&2
    echo "  llvm-config   = $(find_llvm_config 2>/dev/null || echo '<not found>')" >&2
    echo "  PATH          = $PATH" >&2

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
echo "[toolchain] uname=$(uname -s) clang=$(command -v clang 2>/dev/null || echo '<missing>')"

# The official LLVM Windows binary release lacks llvm-config.exe, so on
# Windows we derive the same values (major version + cmake module dir) from
# clang's install prefix and `clang --version`.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        if [[ -n "${LLVM_PATH:-}" ]]; then
            LLVM_PREFIX="$(cygpath -u "${LLVM_PATH}" 2>/dev/null || { CLANG_BIN_PATH="$(command -v clang)"; dirname "$(dirname "$CLANG_BIN_PATH")"; })"
        else
            CLANG_BIN_PATH="$(command -v clang)"
            LLVM_PREFIX="$(dirname "$(dirname "$CLANG_BIN_PATH")")"
        fi
        LLVM_CMAKE_DIR="$LLVM_PREFIX/lib/cmake/llvm"
        LLVM_MAJOR="$(clang --version | head -n1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1 | cut -d. -f1)"
        ;;
    *)
        LLVM_CONFIG="$(find_llvm_config)"
        LLVM_CMAKE_DIR="$($LLVM_CONFIG --cmakedir)"
        LLVM_MAJOR="$($LLVM_CONFIG --version | cut -d. -f1)"
        ;;
esac
# Derive Clang cmake dir from LLVM cmake dir (sibling 'clang' directory)
CLANG_CMAKE_DIR="${LLVM_CMAKE_DIR%/llvm}/clang"
echo "[toolchain] LLVM_PREFIX=$LLVM_PREFIX  LLVM_MAJOR=$LLVM_MAJOR"
echo "[toolchain] LLVM_CMAKE_DIR=$LLVM_CMAKE_DIR  exists=$([ -d "$LLVM_CMAKE_DIR" ] && echo yes || echo NO)"
echo "[toolchain] CLANG_CMAKE_DIR=$CLANG_CMAKE_DIR  exists=$([ -d "$CLANG_CMAKE_DIR" ] && echo yes || echo NO)"
PLUGIN_DIR="$ENZYME_PREFIX/build/Enzyme"
# macOS emits .dylib, Windows emits .dll, Linux emits .so; the stable alias
# used by callers is always ClangEnzyme.so (extension is irrelevant to dlopen).
case "$(uname -s)" in
    Darwin)             PLUGIN_EXT="dylib" ;;
    MINGW*|MSYS*|CYGWIN*) PLUGIN_EXT="dll" ;;
    *)                  PLUGIN_EXT="so" ;;
esac
PLUGIN_REAL="$PLUGIN_DIR/ClangEnzyme-${LLVM_MAJOR}.${PLUGIN_EXT}"
PLUGIN_LINK="$ENZYME_PREFIX/ClangEnzyme.so"

mkdir -p "$ENZYME_PREFIX"

echo "[toolchain] PLUGIN_REAL=$PLUGIN_REAL"
if [[ ! -f "$PLUGIN_REAL" ]]; then
    rm -rf "$ENZYME_PREFIX/src" "$ENZYME_PREFIX/build"
    echo "[toolchain] cloning Enzyme $ENZYME_GIT_REF"
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
    echo "[toolchain] cmake configure: ${cmake_args[*]}"
    cmake "${cmake_args[@]}"
    cmake --build "$ENZYME_PREFIX/build" --target "ClangEnzyme-${LLVM_MAJOR}" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

if [[ ! -f "$PLUGIN_REAL" ]]; then
    shopt -s nullglob
    plugin_candidates=("$PLUGIN_DIR"/ClangEnzyme-*.so "$PLUGIN_DIR"/ClangEnzyme-*.dylib "$PLUGIN_DIR"/ClangEnzyme-*.dll)
    shopt -u nullglob
    if [[ ${#plugin_candidates[@]} -eq 0 ]]; then
        echo "Enzyme build completed, but no ClangEnzyme plugin was produced in $PLUGIN_DIR" >&2
        exit 1
    fi
    PLUGIN_REAL="${plugin_candidates[0]}"
fi

# Windows lacks unprivileged symlinks in git-bash; a plain copy is fine because
# clang's -fpass-plugin loads by path regardless of extension.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) cp -f "$PLUGIN_REAL" "$PLUGIN_LINK" ;;
    *)                    ln -sf "$PLUGIN_REAL" "$PLUGIN_LINK" ;;
esac

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
