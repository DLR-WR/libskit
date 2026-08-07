# LIBSkit

[![CI](https://github.com/DLR-WR/libskit/actions/workflows/ci.yml/badge.svg)](https://github.com/DLR-WR/libskit/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/libskit/badge/?version=latest)](https://libskit.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/libskit.svg)](https://pypi.org/project/libskit/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21788186.svg)](https://doi.org/10.5281/zenodo.21788186)

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

Pre-built wheels are published on PyPI for **Linux (x86_64)** and **macOS
(x86_64 and Apple Silicon)**:

```bash
pip install libskit
```

**Windows is not supported** as a binary target: the official LLVM Windows
distribution does not ship the static Clang development libraries that Enzyme
needs to build against, so we cannot produce Windows wheels in CI. Windows
users must build from source (see below) with a full LLVM install that
includes the Clang static libs (e.g. a self-built LLVM or conda-forge's
`llvmdev` package).

### Building from source

Source builds need a Clang toolchain with the Enzyme automatic-differentiation
plugin.

- **Linux**: `sudo apt install cmake ninja-build git clang llvm-dev`
- **macOS**: `brew install llvm cmake ninja`
- **Windows**: install a full LLVM toolchain (including static Clang dev libs)
  and Ninja; the CI-provided setup script is not sufficient on Windows

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

## Prerequisites (for development)
To build the code from source you need:
- Clang-compatible C++20 compiler
- CMake 3.15+
- Python 3.8+

## Tutorials 

The `notebooks/` directory in this repository contains a curated set of tutorials (`tutorial_1_…` through `tutorial_7_…`).



## Citation

If you use LIBSkit in your research, please cite it:

```bibtex
@software{libskit,
  author  = {Egerland, Christoph H.},
  title   = {LIBSkit: A software tool for the simulation and analysis of
             laser-induced breakdown spectroscopy (LIBS) spectra},
  year    = {2026},
  version = {0.1},
  doi     = {10.5281/zenodo.21788186},
  url     = {https://doi.org/10.5281/zenodo.21788186},
}
```