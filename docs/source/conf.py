# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LIBSkit"
copyright = "Christoph H. Egerland"
author = "Christoph H. Egerland"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "breathe"
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


# Breathe
breathe_projects = {"libskit": os.path.join("..", "build", "doxygen", "xml")}
breathe_default_project = "libskit"

# Autodoc
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Mock the compiled C++ extension so RTD (and docs-smoke CI) can import
# `libskit` without building Enzyme/pybind11.
autodoc_mock_imports = ["libskit._libskit"]

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
