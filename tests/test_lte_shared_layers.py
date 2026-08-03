#
# Copyright (c) 2026 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Tests for libskit.LTESharedLayers (native C++ shared-composition model)."""

from __future__ import annotations

import numpy as np
import pytest

import libskit


def _wavelengths() -> np.ndarray:
    return np.arange(380.0, 410.0, 0.05)


def test_parameter_names_layout() -> None:
    m = libskit.LTESharedLayers(
        {"Fe": 0.7, "Si": 0.3},
        [10000.0, 8000.0],
        [1e23, 5e22],
        [1e-3, 1e-3],
    )
    names = m.get_parameter_names()
    assert names[:6] == [
        "layer_0_temperature",
        "layer_0_electron_density",
        "layer_0_length",
        "layer_1_temperature",
        "layer_1_electron_density",
        "layer_1_length",
    ]
    # element order follows std::map alphabetical iteration
    assert set(names[6:]) == {"Fe", "Si"}
    assert len(m) == 2


def test_set_parameter_shared_element_affects_all_layers() -> None:
    m = libskit.LTESharedLayers(
        {"Fe": 0.5, "Si": 0.5},
        [10000.0, 8000.0],
        [1e23, 5e22],
        [1e-3, 1e-3],
    )
    wls = _wavelengths()
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    irr0 = np.asarray(spec.calculate_spectrum(wls, m))

    m.set_parameter("Fe", 0.4)
    m.set_parameter("Si", 0.6)
    irr1 = np.asarray(spec.calculate_spectrum(wls, m))

    # Two distinct spectra
    assert not np.allclose(irr0, irr1)
    # Constructing a fresh model with the new ratios must match
    m2 = libskit.LTESharedLayers(
        {"Fe": 0.4, "Si": 0.6},
        [10000.0, 8000.0],
        [1e23, 5e22],
        [1e-3, 1e-3],
    )
    irr2 = np.asarray(spec.calculate_spectrum(wls, m2))
    np.testing.assert_allclose(irr1, irr2, rtol=1e-10, atol=0.0)


def test_irradiance_matches_lte_layers_with_duplicated_composition() -> None:
    er = {"Fe": 0.7, "Si": 0.3}
    temps = [10000.0, 8000.0, 6000.0]
    dens = [1e23, 5e22, 1e22]
    lens = [1e-3, 1e-3, 1e-3]

    shared = libskit.LTESharedLayers(er, temps, dens, lens)
    layered = libskit.LTELayers([er, er, er], temps, dens, lens)

    wls = _wavelengths()
    a = np.asarray(shared.calculate_irradiance(wls))
    b = np.asarray(layered.calculate_irradiance(wls))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=0.0)


def test_gradient_consolidates_shared_element() -> None:
    """Element gradient is a single entry, not per-layer."""
    er = {"Fe": 0.7, "Si": 0.3}
    model = libskit.LTESharedLayers(
        er, [10000.0, 8000.0], [1e23, 5e22], [1e-3, 1e-3]
    )
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    wls = _wavelengths()
    true_irr = spec.calculate_spectrum(wls, model)
    true_spectrum = libskit.Spectrum(wls, true_irr)

    init = libskit.LTESharedLayers(
        er, [10500.0, 8500.0], [1.1e23, 5.5e22], [1e-3, 1e-3]
    )
    exp = libskit.Experiment(
        libskit.LossType.L2Loss, init, spec, true_spectrum,
        libskit.NormalizationType.AnalyticNorm,
    )
    grads = libskit.get_gradients(exp)

    # One gradient per parameter as returned by get_parameter_names()
    assert set(grads.keys()) == set(init.get_parameter_names())
    # Shared elements present unprefixed (not duplicated per layer)
    assert "Fe" in grads
    assert "Si" in grads
    assert "layer_0_Fe" not in grads
    assert "layer_1_Fe" not in grads


def test_optimizer_runs_on_shared_layers() -> None:
    er = {"Fe": 1.0}
    truth = libskit.LTESharedLayers(
        er, [11000.0, 9000.0], [5e23, 2e23], [0.5e-3, 0.5e-3]
    )
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    wls = _wavelengths()
    true_spectrum = libskit.Spectrum(wls, spec.calculate_spectrum(wls, truth))

    init = libskit.LTESharedLayers(
        er, [10500.0, 8500.0], [4e23, 1.5e23], [0.4e-3, 0.4e-3]
    )
    exp = libskit.Experiment(
        libskit.LossType.L2Loss, init, spec, true_spectrum,
        libskit.NormalizationType.AnalyticNorm,
    )
    opt = libskit.Optimizer(exp)
    # Sanity: the optimizer discovers per-layer plasma params + one Fe
    assert "layer_0_temperature" in opt.param_names
    assert "layer_1_temperature" in opt.param_names
    assert "Fe" in opt.param_names
    # Loss decreases at least a little after a few iterations
    initial_loss = libskit.loss(exp)
    opt.optimize(max_eval=20)
    assert libskit.loss(exp) <= initial_loss
