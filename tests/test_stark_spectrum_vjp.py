# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Stark spectrum-VJP verification: ``spectrum_vjp_stark(experiment, adjoint)``
returns (J_m)^T @ adjoint in one Enzyme reverse pass, where
J_m(i, l) = ∂y_sim_i / ∂m_l and m_l is the per-line Stark FWHM
multiplier (baseline 1.0).

Reference: brute-force Jacobian assembled column-by-column via central
finite differences directly in multiplier space (no A_l scaling since
Stark multipliers *are* the perturbation parameters).
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


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _make_simple_experiment():
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    wls = np.arange(380.0, 410.0, 0.05)
    true_model = libskit.LTEModel({"Fe": 1.0}, 11000.0, 5e23, 0.5e-3)
    init_model = libskit.LTEModel({"Fe": 1.0}, 9500.0, 4e23, 0.4e-3)
    true_irr = spec.calculate_spectrum(wls, true_model)
    true_spectrum = libskit.Spectrum(wls, true_irr)
    return libskit.Experiment(
        libskit.LossType.L2Loss, init_model, spec, true_spectrum
    )


def _flat_layout(values, elements):
    """Return [(ion, line_index, flat_index)] in build_const_data's order."""
    layout = []
    flat = 0
    for elem in elements:
        for il in range(len(ROMAN)):
            ion = f"{elem} {ROMAN[il]}"
            if ion not in values:
                continue
            n = len(values[ion])
            for li in range(n):
                layout.append((ion, li, flat + li))
            flat += n
    return layout


def _spectrum_under_stark(exp, mults):
    """Re-evaluate y_sim with the given per-ion Stark multipliers active."""
    try:
        libskit.set_stark_multipliers(mults)
        return np.asarray(
            exp.get_instrument().calculate_spectrum(
                exp.true_spectrum.wavelengths, exp.get_plasma_model()
            )
        )
    finally:
        libskit.reset_stark()


def test_spectrum_vjp_stark_shape():
    exp = _make_simple_experiment()
    n_out = exp.true_spectrum.wavelengths.size
    n_stark = libskit.get_stark_gradient(exp).size

    adjoint = np.ones(n_out)
    out = np.asarray(libskit.spectrum_vjp_stark(exp, adjoint))
    assert out.shape == (n_stark,)


def test_spectrum_vjp_stark_matches_jacobian_column_fd():
    """For each strongly-coupled line l, assemble J_m[:, l] via
    multiplier-space central FD and check its contraction with a random
    adjoint agrees with spectrum_vjp_stark(exp, adjoint)[l].
    """
    exp = _make_simple_experiment()
    values = libskit.get_aki_values()  # reused for line layout only
    layout = _flat_layout(values, ["Fe"])

    n_out = exp.true_spectrum.wavelengths.size
    rng = np.random.default_rng(7)
    adjoint = rng.standard_normal(n_out)

    out = np.asarray(libskit.spectrum_vjp_stark(exp, adjoint))
    g = np.asarray(libskit.get_stark_gradient(exp))
    flat_to_ionline = {flat: (ion, li) for ion, li, flat in layout}

    g_max = max(np.max(np.abs(g)), 1e-30)
    active = [k for k in np.argsort(np.abs(g))[::-1] if abs(g[k]) > 1e-3 * g_max]
    picked = active[:5]
    assert len(picked) >= 1, "No active Stark-sensitive line found via Enzyme-VJP"

    h = 1e-3
    n_match = 0
    diagnostics = []
    for flat in picked:
        ion, li = flat_to_ionline[flat]

        plus = {i: [1.0] * len(values[i]) for i in values}
        minus = {i: [1.0] * len(values[i]) for i in values}
        plus[ion][li] = 1.0 + h
        minus[ion][li] = 1.0 - h

        y_plus = _spectrum_under_stark(exp, plus)
        y_minus = _spectrum_under_stark(exp, minus)

        col = (y_plus - y_minus) / (2.0 * h)
        fd_dot = float(adjoint @ col)
        enzyme_dot = float(out[flat])

        denom = max(abs(fd_dot), abs(enzyme_dot), 1e-30)
        rel = abs(fd_dot - enzyme_dot) / denom
        diagnostics.append((ion, li, enzyme_dot, fd_dot, rel))
        if rel < 5e-2:
            n_match += 1

    match_rate = n_match / len(picked)
    if match_rate < 0.8:
        worst = sorted(diagnostics, key=lambda r: -r[4])[:5]
        msg = "\n".join(
            f"  {ion} line {li}: enzyme={e:.3e}, fd={f:.3e}, rel={r:.2e}"
            for ion, li, e, f, r in worst
        )
        pytest.fail(
            f"Stark VJP/FD agreement {n_match}/{len(picked)} = {match_rate:.1%} "
            f"below 80% threshold:\n{msg}"
        )


def test_spectrum_vjp_stark_zero_adjoint_returns_zero():
    exp = _make_simple_experiment()
    n_out = exp.true_spectrum.wavelengths.size
    out = np.asarray(libskit.spectrum_vjp_stark(exp, np.zeros(n_out)))
    assert np.all(out == 0.0)


def test_spectrum_vjp_stark_is_linear_in_adjoint():
    """spectrum_vjp_stark is linear in its adjoint argument:
    VJP(v + α w) = VJP(v) + α · VJP(w)."""
    exp = _make_simple_experiment()
    n_out = exp.true_spectrum.wavelengths.size
    rng = np.random.default_rng(123)
    v = rng.standard_normal(n_out)
    w = rng.standard_normal(n_out)
    alpha = 0.37

    out_v = np.asarray(libskit.spectrum_vjp_stark(exp, v))
    out_w = np.asarray(libskit.spectrum_vjp_stark(exp, w))
    out_combo = np.asarray(libskit.spectrum_vjp_stark(exp, v + alpha * w))
    expected = out_v + alpha * out_w

    if np.any(np.abs(expected) > 0):
        rel = np.max(np.abs(out_combo - expected) / np.maximum(np.abs(expected), 1e-30))
    else:
        rel = 0.0
    assert rel < 1e-10, f"Linearity violated by rel={rel:.2e}"
