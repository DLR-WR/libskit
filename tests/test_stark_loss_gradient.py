# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Stark loss-gradient verification: ``get_stark_gradient`` returns
∂L/∂m_l for every active emission line in one Enzyme reverse pass,
where m_l is the per-line Stark FWHM multiplier (baseline 1.0).

Reference: central finite differences in multiplier space.  Unlike the
Aki gradient (where the multiplier scales an underlying catalogue value
A_l^(0), so ∂L/∂m_l = A_l^(0) · ∂L/∂A_l), the Stark multiplier IS the
perturbation variable, so ``get_stark_gradient`` may be checked
directly against the multiplier-space FD.
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


def _flat_index_layout(elements):
    """Walk get_aki_values() in build_const_data's iteration order
    (elements -> ions ascending) and return [(ion, line_index, flat_index)]
    plus the values dict. Same layout is used for stark multipliers."""
    values = libskit.get_aki_values()
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
    return layout, values


def _loss_with_stark(exp, mults):
    try:
        libskit.set_stark_multipliers(mults)
        return libskit.compute_loss(exp)
    finally:
        libskit.reset_stark()


def test_stark_gradient_shape_nonzero():
    exp = _make_simple_experiment()
    g = np.asarray(libskit.get_stark_gradient(exp))
    assert g.ndim == 1
    assert g.size > 0
    assert np.count_nonzero(g) > 0


def test_stark_gradient_total_directional_fd():
    """Compare the Enzyme-gradient contraction ⟨g, 1⟩ against central FD
    of the loss under a uniform multiplier perturbation m_l = 1+h applied
    to ALL active lines simultaneously.

        L(m=1+h) − L(m=1−h)
        ─────────────────── ≈ Σ_l g_l
                2h
    """
    exp = _make_simple_experiment()

    g = np.asarray(libskit.get_stark_gradient(exp))
    layout, values = _flat_index_layout(["Fe"])
    assert len(layout) == g.size, (
        f"layout size {len(layout)} != gradient size {g.size}; "
        "iteration order mismatch with build_const_data"
    )
    enzyme_dir = float(np.sum(g))

    h = 1e-3
    plus = {ion: [1.0 + h] * len(values[ion]) for ion in values}
    minus = {ion: [1.0 - h] * len(values[ion]) for ion in values}
    L_plus = _loss_with_stark(exp, plus)
    L_minus = _loss_with_stark(exp, minus)
    fd_dir = (L_plus - L_minus) / (2.0 * h)

    rel = abs(enzyme_dir - fd_dir) / max(abs(fd_dir), 1e-30)
    assert rel < 5e-3, (
        f"Total directional derivative mismatch: "
        f"Enzyme Σ g_l = {enzyme_dir:.6e}, FD dL/dh = {fd_dir:.6e}, "
        f"rel = {rel:.2e}"
    )


def test_stark_gradient_per_line_fd():
    """Per-line check: Enzyme g_l vs. multiplier-space central FD,
    restricted to the lines whose Enzyme gradient is non-negligible.
    """
    exp = _make_simple_experiment()
    g = np.asarray(libskit.get_stark_gradient(exp))
    layout, values = _flat_index_layout(["Fe"])
    flat_to_ionline = {flat: (ion, li) for ion, li, flat in layout}

    active_flat = np.argsort(np.abs(g))[::-1]
    g_max = np.max(np.abs(g))
    threshold = 1e-2 * g_max
    picked = [k for k in active_flat if abs(g[k]) >= threshold][:5]
    assert len(picked) >= 1, (
        f"No stark-sensitive lines found (max|g|={g_max:.3e})."
    )

    h = 1e-3
    n_match = 0
    diagnostics = []
    for flat in picked:
        ion, li = flat_to_ionline[flat]
        enzyme_val = g[flat]

        plus = {i: [1.0] * len(values[i]) for i in values}
        minus = {i: [1.0] * len(values[i]) for i in values}
        plus[ion][li] = 1.0 + h
        minus[ion][li] = 1.0 - h
        L_plus = _loss_with_stark(exp, plus)
        L_minus = _loss_with_stark(exp, minus)
        fd_val = (L_plus - L_minus) / (2.0 * h)

        denom = max(abs(fd_val), abs(enzyme_val), 1e-30)
        rel = abs(enzyme_val - fd_val) / denom
        diagnostics.append((ion, li, enzyme_val, fd_val, rel))
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
            f"Per-line match rate {n_match}/{len(picked)} = {match_rate:.1%} "
            f"below 80% threshold. Picked lines:\n{msg}"
        )
