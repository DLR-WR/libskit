# LIBSkit
A software tool for the simulation and analysis of laser-induced breakdown spectroscopy (LIBS) spectra.

The simulation core is written in C++ with Python bindings via pybind11. Atomic databases (NIST ASD lines, ionization energies, energy levels) are bundled with the package and loaded automatically on import.

## Quick Start

```python
import libskit
import numpy as np

# Create an LTE plasma model
model = libskit.LTEModel(
    {"Si": 0.72, "Na": 0.09, "Ca": 0.19},   # %
    temperature=17300.0,                    # K
    electron_density=2.53e23,               # m^-3
    length=2.4e-3                           # m
)

# Define an instrument and calculate a spectrum
wavelengths = np.arange(250, 900, 0.05)                       # nm
instrument = libskit.GenericSpectrometer(0.05, 0.001)         # nm
spectrum = instrument.calculate_spectrum(wavelengths, model)
```

## Installation

This package is only available as a source build — it is not published on PyPI or any package registry.

Source builds need a Clang toolchain with the Enzyme automatic-differentiation plugin.

- **Linux**: ``sudo apt install cmake ninja-build git clang llvm-dev``
- **macOS**: install LLVM/Clang and CMake, then build Enzyme against that LLVM
- **Windows**: use an LLVM/Clang toolchain with Enzyme available

Then:
```bash
git clone <repository-url>
cd libskit

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
./scripts/dev.sh
```

This builds the Enzyme plugin and installs the package in one step.

> **Note:** Eigen3 is downloaded automatically during build if not found on your system.

## Prerequisites (for development)
To build the code from source you need:
- Clang-compatible C++20 compiler
- CMake 3.15+
- Python 3.8+

## Tutorials 

The `notebooks/` directory in this repository contains a curated set of tutorials (`tutorial_1_…` through `tutorial_7_…`).


## Contributors
Christoph H. Egerland

## License
MIT
