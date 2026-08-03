#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Tests for libskit.boltzmann_plot helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from libskit.boltzmann_plot import (
    _group_multiplets,
    _multiplet_correction_term,
    _planck_function,
    _saha_factor,
    _self_absorption_correction,
    saha_factor,
)
from libskit.boltzmann_utils import kB_J, me


class TestGroupMultiplets:
    def test_empty_input(self) -> None:
        assert _group_multiplets([]) == []

    def test_single_line(self) -> None:
        assert _group_multiplets([400.0]) == [[400.0]]

    def test_close_lines_grouped(self) -> None:
        groups = _group_multiplets([400.0, 400.5, 401.0], threshold=2.0)
        assert groups == [[400.0, 400.5, 401.0]]

    def test_distant_lines_separated(self) -> None:
        groups = _group_multiplets([400.0, 405.0, 410.0], threshold=2.0)
        assert groups == [[400.0], [405.0], [410.0]]

    def test_mixed_grouping(self) -> None:
        groups = _group_multiplets([400.0, 400.5, 405.0, 405.3, 410.0],
                                    threshold=2.0)
        assert groups == [[400.0, 400.5], [405.0, 405.3], [410.0]]

    def test_unsorted_input_is_sorted(self) -> None:
        groups = _group_multiplets([405.0, 400.0, 400.5, 410.0],
                                    threshold=2.0)
        assert groups == [[400.0, 400.5], [405.0], [410.0]]


class TestMultipletCorrection:
    def test_single_line_returns_zero(self) -> None:
        params = [{"gk": 1.0, "Aki": 1e8, "wavelength": 400.0, "Ek": 5.0}]
        assert _multiplet_correction_term(params, 10000.0) == 0.0

    def test_two_identical_lines_gives_log2(self) -> None:
        """Two identical-strength lines: correction = ln(1 + 1) = ln(2)."""
        line = {"gk": 4.0, "Aki": 5e7, "wavelength": 400.0, "Ek": 4.0}
        params = [dict(line), dict(line)]
        assert _multiplet_correction_term(params, 10000.0) == pytest.approx(np.log(2.0))

    def test_correction_positive_for_real_multiplet(self) -> None:
        params = [
            {"gk": 4.0, "Aki": 5e7, "wavelength": 400.0, "Ek": 4.0},
            {"gk": 6.0, "Aki": 3e7, "wavelength": 400.5, "Ek": 4.1},
        ]
        assert _multiplet_correction_term(params, 10000.0) > 0


class TestSahaFactorPlot:
    """`_saha_factor` returns ln(prefactor); public `saha_factor` returns prefactor."""

    def test_log_relationship(self) -> None:
        T, ne = 12000.0, 5e23
        assert _saha_factor(T, ne) == pytest.approx(np.log(saha_factor(T, ne)))

    def test_known_formula(self) -> None:
        from scipy import constants as const
        T, ne = 10000.0, 1e23
        expected = 2 * (2 * np.pi * me * kB_J * T) ** 1.5 / (ne * const.h ** 3)
        assert saha_factor(T, ne) == pytest.approx(expected, rel=1e-12)


class TestPlanckFunction:
    def test_returns_positive_finite(self) -> None:
        wl = np.linspace(300.0, 1000.0, 50)
        B = _planck_function(wl, 5800.0)
        assert np.all(np.isfinite(B))
        assert np.all(B > 0)

    def test_peak_near_wien_displacement(self) -> None:
        """For T=5800 K, Planck peak at λ ≈ 500 nm (Wien: 2.898e-3/T)."""
        wl = np.linspace(200.0, 2000.0, 1000)
        B = _planck_function(wl, 5800.0)
        peak_wl = wl[np.argmax(B)]
        # Wien displacement: λ_max·T = 2.898e-3 m·K → 500 nm at 5800 K
        assert 450.0 < peak_wl < 550.0


class TestSelfAbsorptionCorrection:
    def test_unsaturated_signal_unchanged(self) -> None:
        """Far-from-blackbody signal: correction factor ≈ 1 (no change)."""
        wl = np.linspace(400.0, 410.0, 100)
        # Tiny intensity compared to blackbody at the same T
        I = 1e-30 * np.ones_like(wl)
        I_corr = _self_absorption_correction(wl, I, 10000.0)
        assert np.allclose(I_corr, I, rtol=1e-6)
