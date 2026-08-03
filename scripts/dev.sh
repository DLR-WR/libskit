#!/bin/bash
export ENZYME_PREFIX="${ENZYME_PREFIX:-${TMPDIR:-/tmp}/libskit-enzyme-${USER:-user}}"
bash ./scripts/setup_enzyme_toolchain.sh
source "$ENZYME_PREFIX/env.sh"
pip install -e .