#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Shared pytest configuration for libskit tests.

The repository ships two equivalent copies of the compiled C extension
(``build/_libskit.so`` and the editable-install copy under
``site-packages/libskit/_libskit*.so``), and historical tests import
``_libskit`` directly from ``build/`` while modern tests import the
top-level ``libskit`` package. Pybind11 raises
``generic_type: type "Spectrum" is already registered`` if both
extensions are loaded as separate modules in the same process.

We unify them up-front: prepend ``build/`` to ``sys.path``, import
``_libskit`` once, and alias it as ``libskit._libskit`` so subsequent
relative imports inside the Python package resolve to the same module
object.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "build"
_SRC_DIR = _REPO_ROOT / "src"

# If an editable install of libskit from a different worktree is registered as
# a meta-path finder (scikit-build-core's ScikitBuildRedirectingFinder), it
# will intercept ``import libskit`` and load the wrong package. Drop it so the
# standard PathFinder picks up this worktree's ``src/libskit``.
_kept_finders = []
for _f in sys.meta_path:
    _name = type(_f).__name__
    if _name == "ScikitBuildRedirectingFinder":
        continue
    _kept_finders.append(_f)
sys.meta_path[:] = _kept_finders
for _stale in ("libskit", "_libskit"):
    sys.modules.pop(_stale, None)

if _BUILD_DIR.is_dir() and str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

# scikit-build-core places the .so in a wheel-tag subdirectory like
# build/cp312-cp312-linux_x86_64/.  Search one level deep so that
# ``import _libskit`` succeeds when the extension lives there.
for _subdir in sorted(_BUILD_DIR.glob("*")):
    if _subdir.is_dir() and any(_subdir.glob("_libskit*.so")):
        if str(_subdir) not in sys.path:
            sys.path.insert(0, str(_subdir))
        break

# Expose the in-tree Python package (src/libskit/) so 'import libskit' works
# without requiring a pip install. CI invokes pytest with PYTHONPATH=build:...
# only, so without this addition tests that import the top-level package fail
# at collection time.
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    import _libskit as _ext  # noqa: F401  — load from build/ first
    # Guard against picking up the C++ source directory ``src/_libskit/`` as
    # an empty namespace package (which would shadow the real extension).
    if hasattr(_ext, "Experiment"):
        sys.modules.setdefault("libskit._libskit", _ext)
except ImportError:
    # _libskit not present (e.g. pure-Python CI lint job); let individual
    # tests skip via their own import errors.
    pass
