# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""Tests for libskit.boltzmann_utils — physics utilities for Boltzmann/Saha plots."""

import numpy as np
import pytest

import libskit  # noqa: F401  — triggers database autoload
from libskit.boltzmann_utils import (
    get_atomic_mass,
    int_to_roman,
    kB_eV,
    kB_J,
    me,
    partition_function,
    saha_factor,
    saha_function,
)


class TestRomanNumerals:
    @pytest.mark.parametrize(
        "value,expected",
        [(1, "I"), (2, "II"), (4, "IV"), (5, "V"), (9, "IX"),
         (10, "X"), (40, "XL"), (90, "XC"), (1994, "MCMXCIV")],
    )
    def test_known_values(self, value: int, expected: str) -> None:
        assert int_to_roman(value) == expected

    @pytest.mark.parametrize("bad", [0, -1, 4000, 5000])
    def test_out_of_range_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            int_to_roman(bad)


class TestAtomicMass:
    @pytest.mark.parametrize(
        "element,expected_amu,tol",
        [("H", 1.008, 0.01),
         ("Fe", 55.845, 0.01),
         ("Ca", 40.078, 0.01),
         ("U", 238.029, 0.01)],
    )
    def test_known_atomic_masses(self, element: str, expected_amu: float, tol: float) -> None:
        mass = get_atomic_mass(element)
        assert abs(mass - expected_amu) < tol


class TestSahaFactor:
    """saha_factor(T, ne) = 2(2π me kB T)^(3/2) / (ne h^3) — pure formula check."""

    def test_against_explicit_formula(self) -> None:
        from scipy import constants as const
        T, ne = 12000.0, 5e23
        expected = (
            2 * (2 * np.pi * me * kB_J * T) ** 1.5
            / (ne * const.h**3)
        )
        assert abs(saha_factor(T, ne) - expected) / expected < 1e-12

    def test_inverse_proportional_to_ne(self) -> None:
        T = 10000.0
        s1 = saha_factor(T, 1e23)
        s2 = saha_factor(T, 2e23)
        assert abs(s1 / s2 - 2.0) < 1e-12

    def test_temperature_scaling(self) -> None:
        """saha_factor ∝ T^(3/2) at fixed ne."""
        ne = 1e23
        s_low = saha_factor(5000.0, ne)
        s_high = saha_factor(20000.0, ne)
        # Ratio = (20000/5000)^1.5 = 4^1.5 = 8
        assert abs(s_high / s_low - 8.0) < 1e-12


class TestPartitionFunction:
    """C++-backed partition function from NIST levels."""

    def test_returns_positive_finite(self) -> None:
        for element, sp_num in [("Fe", 1), ("Ca", 1), ("Ca", 2), ("Si", 1)]:
            U = partition_function(element, sp_num, 10000.0)
            assert np.isfinite(U)
            assert U > 0

    def test_monotonic_in_temperature(self) -> None:
        """U(T) must be non-decreasing in T (more states accessible)."""
        Us = [partition_function("Fe", 1, T) for T in (5000, 8000, 12000, 20000)]
        for prev, curr in zip(Us, Us[1:]):
            assert curr >= prev - 1e-9


class TestSahaFunction:
    """Composite Saha function used in CF-LIBS."""

    def test_returns_positive(self) -> None:
        ion_e = {"Ca": 6.11, "Fe": 7.90}
        for element in ("Ca", "Fe"):
            S = saha_function(element, 10000.0, 1e23, ion_e)
            assert np.isfinite(S)
            assert S > 0

    def test_decreases_with_ne(self) -> None:
        """S(T)/ne is divided by ne, so doubling ne halves it."""
        ion_e = {"Ca": 6.11}
        s_low = saha_function("Ca", 10000.0, 1e23, ion_e)
        s_high = saha_function("Ca", 10000.0, 2e23, ion_e)
        assert abs(s_low / s_high - 2.0) < 1e-9

    def test_unknown_element_uses_default_ion_energy(self) -> None:
        """A missing entry falls back to 6.0 eV without raising."""
        S = saha_function("Xe", 10000.0, 1e23, ionization_energies={})
        assert np.isfinite(S)
        assert S > 0
