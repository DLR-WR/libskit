# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Verification of ``libskit.stark_covariance(experiment, sigma_stark, ...)``.

Tests:
  1. shape, symmetry, PSD properties of every Sigma it returns.
  2. quadratic scaling: sigma_stark -> alpha * sigma_stark scales
     Sigma_stark by alpha^2 exactly.
  3. closure constraint: dependent-element row/column consistent with
     -sum(other elements).
  4. dict sigma_stark broadcasts correctly per ion.
  5. graceful failure on invalid inputs.
"""

import os
import sys

_build_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "build", "cp312-cp312-linux_x86_64"
)
sys.path.insert(0, _build_dir)

import numpy as np
import pytest

import libskit


def _make_experiment(elements=None, wide=False):
    if elements is None:
        elements = {"Na": 0.4, "Mg": 0.3, "Ca": 0.3}
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    if wide:
        wls = np.arange(380.0, 620.0, 0.05)
    else:
        wls = np.arange(580.0, 620.0, 0.05)
    true_model = libskit.LTEModel(elements, 11000.0, 5e23, 0.5e-3)
    init_model = libskit.LTEModel(elements, 9500.0, 4e23, 0.4e-3)
    true_irr = spec.calculate_spectrum(wls, true_model)
    true_spectrum = libskit.Spectrum(wls, true_irr)
    return libskit.Experiment(
        libskit.LossType.L2Loss, init_model, spec, true_spectrum
    )


def test_stark_covariance_shape_symmetry_psd():
    exp = _make_experiment()
    res = libskit.stark_covariance(
        exp, sigma_stark=0.05, include_noise=True, sigma_n=1e-8
    )
    p = len(res["parameter_names"])
    assert p > 0, "empty parameter list"

    for key in ("Sigma_stark", "Sigma_noise", "Sigma_total"):
        S = res[key]
        assert S.shape == (p, p), f"{key} shape {S.shape} != ({p},{p})"
        asym = np.max(np.abs(S - S.T))
        assert asym < 1e-12, f"{key} not symmetric: |asym|={asym:.2e}"
        eigvals = np.linalg.eigvalsh(S)
        tol = max(abs(eigvals).max() * 1e-10, 1e-30)
        assert eigvals.min() > -tol, (
            f"{key} not PSD: min eig = {eigvals.min():.3e}, tol = {tol:.3e}"
        )


def test_stark_covariance_quadratic_scaling():
    """Increasing sigma_stark from 0.05 to 0.25 (factor 5) should scale
    Sigma_stark by exactly 25 — Sigma_stark = M diag(sigma_m^2) M^T is
    quadratic in sigma_m.

    Equivalent: sqrt(diag(Sigma)) at sigma=0.25 divided by sqrt(diag(Sigma))
    at sigma=0.05 should equal 5.0 for every parameter.

    Uses the wide window to avoid the near-singular H (cond ~ 1e17) of
    the narrow window; there M diag(sigma^2) M^T scales exactly by 25 but
    the closure-expansion arithmetic amplifies float noise to O(1) in
    entries that are already at the noise floor.
    """
    exp = _make_experiment(wide=True)
    res_low = libskit.stark_covariance(
        exp, sigma_stark=0.05, include_noise=False, sigma_n=0.0
    )
    res_high = libskit.stark_covariance(
        exp, sigma_stark=0.25, include_noise=False, sigma_n=0.0
    )

    S_low = res_low["Sigma_stark"]
    S_high = res_high["Sigma_stark"]
    # Relative floor: entries below 1e-10 of the max are noise-dominated.
    floor = max(np.max(np.abs(S_low)), 1e-30) * 1e-10
    nonzero = np.abs(S_low) > floor
    if not np.any(nonzero):
        pytest.skip("Sigma_stark identically zero (no stark-sensitive lines)")
    ratio = S_high[nonzero] / S_low[nonzero]
    assert np.allclose(ratio, 25.0, rtol=1e-6), (
        f"Sigma_stark does not scale quadratically: ratio range "
        f"[{ratio.min():.3e}, {ratio.max():.3e}] (expected 25.0)"
    )

    diag_low = np.sqrt(np.maximum(np.diag(S_low), 0))
    diag_high = np.sqrt(np.maximum(np.diag(S_high), 0))
    diag_floor = max(diag_low.max(), 1e-30) * 1e-6
    mask = diag_low > diag_floor
    if np.any(mask):
        sd_ratio = diag_high[mask] / diag_low[mask]
        rel = np.max(np.abs(sd_ratio - 5.0) / 5.0)
        assert rel < 1e-2, (
            f"sigma_theta(0.25)/sigma_theta(0.05) = {sd_ratio}, "
            f"deviation from 5.0: {rel:.3e}"
        )


def test_stark_covariance_closure_consistency():
    exp = _make_experiment({"Na": 0.4, "Mg": 0.3, "Ca": 0.3})
    res = libskit.stark_covariance(
        exp, sigma_stark=0.05, include_noise=False, sigma_n=0.0
    )
    pn = res["parameter_names"]
    elements = list(exp.get_plasma_model().element_ratio.keys())
    dep = elements[-1]
    others = [e for e in elements if e != dep]

    dep_i = pn.index(dep)
    other_i = [pn.index(e) for e in others]

    S = res["Sigma_stark"]
    for k in range(S.shape[1]):
        expected = -sum(S[oj, k] for oj in other_i)
        diff = abs(S[dep_i, k] - expected)
        denom = max(abs(S[dep_i, k]), abs(expected), 1e-30)
        rel = diff / denom
        assert rel < 1e-8, (
            f"closure broken at row dep={dep}, col={pn[k]}: "
            f"S[dep,k]={S[dep_i, k]:.3e}, expected={expected:.3e}, rel={rel:.3e}"
        )


def test_stark_covariance_dict_sigma_broadcasts_per_ion():
    """Passing sigma_stark as a dict should apply the given sigma to
    every line of that ion, with missing ions taking the mean fallback.
    """
    exp = _make_experiment({"Na": 0.4, "Mg": 0.3, "Ca": 0.3})
    # Scalar equivalent: put same value for every ion in the model
    sigma_dict = {"Na I": 0.10, "Mg I": 0.10, "Ca I": 0.10,
                  "Na II": 0.10, "Mg II": 0.10, "Ca II": 0.10}
    res_scalar = libskit.stark_covariance(
        exp, sigma_stark=0.10, include_noise=False, sigma_n=0.0
    )
    res_dict = libskit.stark_covariance(
        exp, sigma_stark=sigma_dict, include_noise=False, sigma_n=0.0
    )
    S_s = res_scalar["Sigma_stark"]
    S_d = res_dict["Sigma_stark"]
    nonzero = np.abs(S_s) > 1e-25
    if not np.any(nonzero):
        pytest.skip("Sigma_stark identically zero")
    rel = np.max(np.abs(S_s[nonzero] - S_d[nonzero]) /
                 np.maximum(np.abs(S_s[nonzero]), 1e-30))
    assert rel < 1e-10, (
        f"Dict sigma_stark does not match scalar equivalent: rel={rel:.3e}"
    )


def test_stark_covariance_invalid_inputs():
    exp = _make_experiment()
    with pytest.raises(ValueError):
        libskit.stark_covariance(exp, sigma_stark=0.05, include_noise=True)
    n_out = exp.true_spectrum.wavelengths.size
    with pytest.raises(ValueError):
        libskit.stark_covariance(exp, sigma_stark=0.05,
                                 weights=np.ones(n_out + 1),
                                 include_noise=False, sigma_n=0.0)
    with pytest.raises(ValueError):
        libskit.stark_covariance(exp, sigma_stark=0.05,
                                 weights=-np.ones(n_out),
                                 include_noise=False, sigma_n=0.0)
    with pytest.raises(ValueError):
        libskit.stark_covariance(exp, sigma_stark=-0.05,
                                 include_noise=False, sigma_n=0.0)
    with pytest.raises(ValueError):
        libskit.stark_covariance(exp, sigma_stark={},
                                 include_noise=False, sigma_n=0.0)
    with pytest.raises(ValueError):
        libskit.stark_covariance(exp, sigma_stark=0.05,
                                 include_noise=False, sigma_n=0.0,
                                 dependent_element="Fe")
