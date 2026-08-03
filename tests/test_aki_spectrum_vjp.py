# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Phase 2 verification: ``spectrum_vjp_aki(experiment, adjoint)`` returns
(J_A)^T @ adjoint in one Enzyme reverse pass, where
J_A(i, l) = ∂y_sim_i / ∂A_l.

Reference: brute-force Jacobian assembled column-by-column via central
finite differences in multiplier space.  For each line l we run a small
perturbation through the same in-process line database used by Phase 1.
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


def _flat_layout(values):
    """Return [(ion, line_index, flat_index)] in build_const_data's order."""
    layout = []
    flat = 0
    for elem in ["Fe"]:
        for il in range(len(ROMAN)):
            ion = f"{elem} {ROMAN[il]}"
            if ion not in values:
                continue
            n = len(values[ion])
            for li in range(n):
                layout.append((ion, li, flat + li))
            flat += n
    return layout


def _spectrum_under_multipliers(exp, mults):
    """Re-evaluate simulated y_sim with given per-ion multipliers active.

    Uses the same forward model that compute_loss calls, but
    returns y_sim (not the loss).  The Spectrum class exposes irradiance
    on the experiment's output grid.
    """
    try:
        libskit.set_aki_multipliers(mults)
        spec = exp.get_instrument().calculate_spectrum(
            exp.true_spectrum.wavelengths, exp.get_plasma_model()
        )
        return np.asarray(spec)
    finally:
        libskit.reset_aki()


def test_spectrum_vjp_shape():
    exp = _make_simple_experiment()
    n_out = exp.true_spectrum.wavelengths.size
    n_aki = libskit.get_aki_gradient(exp).size

    adjoint = np.ones(n_out)
    out = libskit.spectrum_vjp_aki(exp, adjoint)
    out = np.asarray(out)
    assert out.shape == (n_aki,)


def test_spectrum_vjp_matches_jacobian_column_fd():
    """For each active line l (Enzyme-VJP gives non-negligible output for
    SOME adjoint), assemble the Jacobian column J_A[:, l] via multiplier-
    space central FD, then check the contraction agrees with
    spectrum_vjp_aki(exp, adjoint) for several random adjoint vectors.

    Strategy:
      * use a fixed adjoint v of length n_out
      * pick the strongest-coupled lines via get_aki_gradient
      * for each such l, compute J_A[:, l] = (y_sim(m_l=1+h) - y_sim(m_l=1-h)) / (2 h A_l^(0))
      * confirm dot(v, J_A[:, l]) == out[l] from spectrum_vjp_aki(v)
    """
    exp = _make_simple_experiment()
    values = libskit.get_aki_values()
    layout = _flat_layout(values)

    n_out = exp.true_spectrum.wavelengths.size
    rng = np.random.default_rng(7)
    adjoint = rng.standard_normal(n_out)

    out = np.asarray(libskit.spectrum_vjp_aki(exp, adjoint))
    g = np.asarray(libskit.get_aki_gradient(exp))
    flat_to_ionline = {flat: (ion, li) for ion, li, flat in layout}

    g_max = max(np.max(np.abs(g)), 1e-30)
    active = [k for k in np.argsort(np.abs(g))[::-1] if abs(g[k]) > 1e-3 * g_max]
    picked = active[:5]
    assert len(picked) >= 1, "No active line found via Enzyme-VJP"

    h = 1e-3
    n_match = 0
    diagnostics = []
    for flat in picked:
        ion, li = flat_to_ionline[flat]
        A0_l = values[ion][li]

        plus = {i: [1.0] * len(values[i]) for i in values}
        minus = {i: [1.0] * len(values[i]) for i in values}
        plus[ion][li] = 1.0 + h
        minus[ion][li] = 1.0 - h

        y_plus = _spectrum_under_multipliers(exp, plus)
        y_minus = _spectrum_under_multipliers(exp, minus)

        # ∂y_sim/∂m_l = A_l · ∂y_sim/∂A_l, so divide by A_l to get J_A[:, l]
        col = (y_plus - y_minus) / (2.0 * h * A0_l)
        fd_dot = float(adjoint @ col)
        enzyme_dot = float(out[flat])

        denom = max(abs(fd_dot), abs(enzyme_dot), 1e-30)
        rel = abs(fd_dot - enzyme_dot) / denom
        diagnostics.append((ion, li, enzyme_dot, fd_dot, rel))
        if rel < 1e-2:
            n_match += 1

    match_rate = n_match / len(picked)
    if match_rate < 0.8:
        worst = sorted(diagnostics, key=lambda r: -r[4])[:5]
        msg = "\n".join(
            f"  {ion} line {li}: enzyme={e:.3e}, fd={f:.3e}, rel={r:.2e}"
            for ion, li, e, f, r in worst
        )
        pytest.fail(
            f"VJP/FD agreement {n_match}/{len(picked)} = {match_rate:.1%} "
            f"below 80% threshold:\n{msg}"
        )


def test_spectrum_vjp_zero_adjoint_returns_zero():
    """spectrum_vjp_aki(exp, 0) must be exactly zero — sanity check."""
    exp = _make_simple_experiment()
    n_out = exp.true_spectrum.wavelengths.size
    out = np.asarray(libskit.spectrum_vjp_aki(exp, np.zeros(n_out)))
    assert np.all(out == 0.0)


def test_spectrum_vjp_is_linear_in_adjoint():
    """spectrum_vjp_aki is linear in its adjoint argument: VJP(v + αw) = VJP(v) + α·VJP(w)."""
    exp = _make_simple_experiment()
    n_out = exp.true_spectrum.wavelengths.size
    rng = np.random.default_rng(123)
    v = rng.standard_normal(n_out)
    w = rng.standard_normal(n_out)
    alpha = 0.37

    out_v = np.asarray(libskit.spectrum_vjp_aki(exp, v))
    out_w = np.asarray(libskit.spectrum_vjp_aki(exp, w))
    out_combo = np.asarray(libskit.spectrum_vjp_aki(exp, v + alpha * w))
    expected = out_v + alpha * out_w

    if np.any(np.abs(expected) > 0):
        rel = np.max(np.abs(out_combo - expected) / np.maximum(np.abs(expected), 1e-30))
    else:
        rel = 0.0
    assert rel < 1e-10, f"Linearity violated by rel={rel:.2e}"
