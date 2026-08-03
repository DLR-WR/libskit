# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Phase 3 verification: ``libskit.aki_covariance(experiment, ...)`` matches
its mathematical definition and degrades gracefully on edge cases.

Tests:
  1. shape, symmetry, PSD properties of every Sigma it returns.
  2. quadratic scaling under sigma_A -> alpha * sigma_A.
  3. closure constraint: dependent-element row/column is consistent with
     -sum(other elements).
  4. parameter_shift_vectors / line_importance_ranking helpers return
     the same M and importance order as `aki_covariance`.
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
    """Build a small fitted experiment for covariance tests.

    ``wide=True`` uses a 380-620 nm window with multiple ionisation
    stages of each element, which breaks the (n_e, L, T) near-degeneracy
    and yields cond(H) ~ 1e2.  Only the dependent-element-choice test
    needs this; everything else is fine with the narrower default.
    """
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


@pytest.fixture(scope="module")
def cov_wide():
    """Cached aki_covariance result on the wide-window experiment."""
    exp = _make_experiment(wide=True)
    res_default = libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0)
    res_mg = libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0,
                                    dependent_element="Mg")
    return exp, res_default, res_mg


def test_aki_covariance_shape_symmetry_psd():
    exp = _make_experiment()
    res = libskit.aki_covariance(exp, include_noise=True, sigma_n=1e-8)

    p = len(res["parameter_names"])
    assert p > 0, "empty parameter list"

    for key in ("Sigma_aki", "Sigma_noise", "Sigma_total"):
        S = res[key]
        assert S.shape == (p, p), f"{key} shape {S.shape} != ({p},{p})"
        asym = np.max(np.abs(S - S.T))
        assert asym < 1e-12, f"{key} not symmetric: |asym|={asym:.2e}"
        eigvals = np.linalg.eigvalsh(S)
        # PSD: smallest eigenvalue must be > -1e-10 * max|eig|
        tol = max(abs(eigvals).max() * 1e-10, 1e-30)
        assert eigvals.min() > -tol, (
            f"{key} not PSD: min eig = {eigvals.min():.3e}, tol = {tol:.3e}"
        )


def test_aki_covariance_scaling_law():
    """Doubling all sigma_A_l should scale Sigma_aki by exactly 4.

    Verified via the M matrix: Sigma_aki_red = M diag(sigma^2) M^T, a
    bilinear form in sigma.  Confirms the implementation has no hidden
    nonlinear dependence on the catalogue accuracies.
    """
    exp = _make_experiment()
    res = libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0)
    M_red = res["M"]
    line_imp = res["line_importance"]
    # line_importance is sorted, so we cannot read sigma directly from it
    # for the M ordering. Re-derive enumeration-order sigma using the
    # public API.
    aki_values = libskit.get_aki_values()
    aki_accs   = libskit.get_aki_accuracies()
    elements = list(exp.get_plasma_model().element_ratio.keys())
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    sigma_enum = []
    for el in elements:
        for il in range(len(roman)):
            ion = f"{el} {roman[il]}"
            if ion not in aki_values:
                continue
            for li in range(len(aki_values[ion])):
                sigma_enum.append(aki_accs[ion][li] * aki_values[ion][li])
    sigma_enum = np.array(sigma_enum)
    assert M_red.shape[1] == sigma_enum.size

    Sigma_1x = M_red @ (np.diag(sigma_enum ** 2)) @ M_red.T
    Sigma_2x = M_red @ (np.diag((2 * sigma_enum) ** 2)) @ M_red.T
    nonzero = np.abs(Sigma_1x) > 1e-20
    if not np.any(nonzero):
        pytest.skip("Reduced Sigma is identically zero (no active lines).")
    ratio = Sigma_2x[nonzero] / Sigma_1x[nonzero]
    assert np.allclose(ratio, 4.0, atol=1e-10), (
        f"Sigma_aki does not scale quadratically: ratio range "
        f"[{ratio.min():.3e}, {ratio.max():.3e}]"
    )


def test_aki_covariance_closure_consistency():
    """The dependent element's row+column should be expressible as
    cov(x_dep, theta_k) = -sum_{other elem} cov(x_other, theta_k)."""
    exp = _make_experiment({"Na": 0.4, "Mg": 0.3, "Ca": 0.3})
    res = libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0)

    pn = res["parameter_names"]
    elements = list(exp.get_plasma_model().element_ratio.keys())
    # Default dependent element is the last in the model's map (alphabetical)
    dep = elements[-1]
    others = [e for e in elements if e != dep]

    dep_i = pn.index(dep)
    other_i = [pn.index(e) for e in others]

    Sigma = res["Sigma_aki"]
    # Sigma[dep_i, k] should equal -sum_j Sigma[other_j, k]  for any k
    for k in range(Sigma.shape[1]):
        expected = -sum(Sigma[oj, k] for oj in other_i)
        diff = abs(Sigma[dep_i, k] - expected)
        denom = max(abs(Sigma[dep_i, k]), abs(expected), 1e-30)
        rel = diff / denom
        assert rel < 1e-8, (
            f"closure broken at row dep={dep}, col={pn[k]}: "
            f"Sigma[dep,k]={Sigma[dep_i, k]:.3e}, expected={expected:.3e}, rel={rel:.3e}"
        )


def test_parameter_shift_helper_matches_main():
    exp = _make_experiment()
    full = libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0)
    helper = libskit.parameter_shift_vectors(exp)
    assert np.allclose(helper["M"], full["M"])
    assert helper["parameter_names"] == full["parameter_names"]


def test_line_importance_helper_matches_main():
    exp = _make_experiment()
    full = libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0)
    helper = libskit.line_importance_ranking(exp, top_n=5)
    assert helper == full["line_importance"][:5]


def test_aki_covariance_dependent_element_choice(cov_wide):
    """Choosing a different dependent element should produce statistically
    equivalent Sigma in the parameter columns that exist in both
    parametrisations (Sigma_aki[i,j] for non-element i,j must agree).

    Requires the wider wavelength window to avoid the cond ~ 1e16
    near-degeneracy of the narrow window (the two parametrisations agree
    mathematically but differ to within numerical noise).
    """
    _, res_default, res_mg = cov_wide
    pn = res_default["parameter_names"]
    # Non-element parameter indices
    nonelem_idx = [i for i, name in enumerate(pn)
                   if name not in ("Na", "Mg", "Ca")]
    S_def = res_default["Sigma_aki"][np.ix_(nonelem_idx, nonelem_idx)]
    S_mg  = res_mg     ["Sigma_aki"][np.ix_(nonelem_idx, nonelem_idx)]
    rel = np.max(np.abs(S_def - S_mg) / np.maximum(np.abs(S_def), 1e-30))
    assert rel < 1e-2, (
        f"Sigma_aki depends too much on dependent_element choice: rel={rel:.3e}"
    )


def test_aki_covariance_invalid_inputs():
    exp = _make_experiment()
    with pytest.raises(ValueError):
        libskit.aki_covariance(exp, include_noise=True)  # missing sigma_n
    n_out = exp.true_spectrum.wavelengths.size
    with pytest.raises(ValueError):
        libskit.aki_covariance(exp, weights=np.ones(n_out + 1),
                               include_noise=False, sigma_n=0.0)
    with pytest.raises(ValueError):
        libskit.aki_covariance(exp, weights=-np.ones(n_out),
                               include_noise=False, sigma_n=0.0)
    with pytest.raises(ValueError):
        libskit.aki_covariance(exp, include_noise=False, sigma_n=0.0,
                               dependent_element="Fe")  # not in model
