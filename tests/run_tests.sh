#!/bin/bash
# Simplified test runner for libskit
# Uses ctest for C++ tests and pytest for Python tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "============================================"
echo "LibsKit Test Suite"
echo "============================================"

# Check build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found. Run 'cmake -B build && cmake --build build' first."
    exit 1
fi

# Run C++ tests via ctest
echo ""
echo "Running C++ tests..."
echo "--------------------------------------------"
cd "$PROJECT_ROOT"
ctest --test-dir "$BUILD_DIR" --output-on-failure --output-junit test-results-cpp.xml

# Run Python tests via pytest
echo ""
echo "Running Python tests..."
echo "--------------------------------------------"
export PYTHONPATH="$BUILD_DIR:$PYTHONPATH"
pytest "$SCRIPT_DIR" --junitxml=test-results-python.xml -v --ignore="$SCRIPT_DIR/test_utils.py"

echo ""
echo "============================================"
echo "All tests passed!"
echo "============================================"
