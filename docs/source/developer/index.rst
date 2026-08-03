Developer Documentation
========================

Welcome to the LIBSkit developer documentation. This section provides detailed information
for developers who want to contribute to LIBSkit or understand its internal architecture.

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   release_workflow

Overview
--------

LIBSkit is structured as a hybrid C++/Python project:

* **C++ Core**: plasma model calculations, gradients, everything numeric
* **Python Interface**: bindings to C++ code for ease of use

Project Structure
-----------------

::

    libskit/
    ├── src/
    │   ├── _libskit/          # C++ source code
    │   └── libskit/           # Python package
    ├── databases/             # NIST ASD atomic data
    ├── tests/                 # Unit and integration tests
    ├── docs/                  # Documentation source
    └── notebooks/             # User-facing tutorial notebooks

Deeper analysis notebooks (parameter sensitivity, multiplet Boltzmann,
ChemCam reanalysis, autodiff benchmarks, …) live in the companion
`libskit_experiments`_ repository.

.. _libskit_experiments: https://codebase.helmholtz.cloud/christoph.egerland/libskit-experiments
