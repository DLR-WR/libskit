Installation
============

From a Wheel (Recommended)
--------------------------

The easiest way to install LIBSkit is from a pre-built wheel::

    pip install libskit

The wheel includes the compiled C++ extension and all required atomic
databases — no compiler or external libraries needed.

From Source
-----------

If you need to build from source (e.g., for development), you need:

* **Clang-based C++ compiler** with C++20 support
* **CMake** (version 3.15 or higher)
* **Python** (3.8 or higher)
* **LLVM + Enzyme** for automatic differentiation support

Eigen3 and pybind11 are downloaded automatically during the build if not
found on your system.

1. **Clone and enter the repository**::

    git clone https://codebase.helmholtz.cloud/christoph.egerland/libskit.git
    cd libskit

2. **Create and activate a virtual environment** (recommended)::

    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install LIBSkit from source**::

    ./scripts/dev.sh

   This builds the Enzyme toolchain and installs the package in one step.


Verifying the Installation
--------------------------

To verify that LIBSkit is installed correctly, run::

    python -c "import libskit; print('LIBSkit', libskit.__version__, 'installed successfully!')"

The atomic databases are bundled inside the package and loaded
automatically on ``import libskit``.
