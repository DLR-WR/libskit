#!/bin/bash
# Notebook tests for libskit - executes all tutorial notebooks
# Note: tutorial_4_gradient_descent is skipped in CI as it's computationally expensive

set -e

cd "$(dirname "$0")/.."

if [ ! -d "notebooks" ]; then
    echo "ERROR: notebooks directory not found"
    exit 1
fi

export PYTHONPATH="${PWD}/build:${PYTHONPATH}"

echo "Testing tutorial notebooks..."
failed=0

for notebook in notebooks/tutorial_*.ipynb; do
    # Skip computationally expensive notebooks in CI
    if [[ "$notebook" == *"tutorial_4_gradient_descent"* ]]; then
        echo "Skipping: $notebook (computationally expensive)"
        continue
    fi

    if [[ "$notebook" == *"tutorial_7"* ]]; then
        echo "Skipping: $notebook (requires measurement data file)"
        continue
    fi

    echo "Testing: $notebook"
    if jupyter nbconvert --to notebook --execute "$notebook" --stdout --ExecutePreprocessor.timeout=600 > /dev/null 2>&1; then
        echo "  ✓ passed"
    else
        echo "  ✗ failed"
        failed=1
    fi
done

exit $failed

echo ""
echo "✓ All tutorial notebooks executed successfully!"
exit 0
