# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""Tests for libskit.wavelengths (vacuum/air conversion)."""

import numpy as np
import pytest

from libskit.wavelengths import air_to_vacuum, vacuum_to_air


class TestVacuumAirConversion:
    """Vacuum ↔ air wavelength conversion (Ciddor 1996)."""

    def test_round_trip_scalar(self) -> None:
        """vacuum_to_air ∘ air_to_vacuum must be the identity to ~1 pm."""
        # Single Newton step in air_to_vacuum; valid range is the visible/IR
        # where the initial guess is accurate.
        for lam in [393.37, 500.0, 656.28, 900.0, 1100.0]:
            recovered = vacuum_to_air(air_to_vacuum(lam))
            assert abs(float(recovered) - lam) < 1e-3, (
                f"round-trip failed at {lam} nm: got {float(recovered)}"
            )

    def test_round_trip_array(self) -> None:
        """Round-trip works element-wise on arrays."""
        wls = np.linspace(380.0, 1100.0, 200)
        recovered = vacuum_to_air(air_to_vacuum(wls))
        assert np.allclose(recovered, wls, atol=1e-3)

    def test_air_is_shorter_than_vacuum(self) -> None:
        """Air wavelengths are always smaller than vacuum (n_air > 1)."""
        wls_vac = np.linspace(300.0, 1000.0, 50)
        wls_air = vacuum_to_air(wls_vac)
        assert np.all(wls_air < wls_vac), "air should be shorter than vacuum"

    def test_known_iau_value_500nm(self) -> None:
        """Vacuum 500.000 nm ≈ air 499.86 nm (Ciddor 1996, ±0.01 nm)."""
        # IAU/Ciddor reference: 500 nm vac → ~499.86 nm air
        air = float(vacuum_to_air(500.0))
        assert 499.83 < air < 499.89, f"expected ~499.86 nm, got {air}"

    def test_refractive_index_decreases_with_wavelength(self) -> None:
        """(n - 1) must decrease toward longer wavelengths (normal dispersion)."""
        n_uv = 300.0 / float(vacuum_to_air(300.0))
        n_ir = 900.0 / float(vacuum_to_air(900.0))
        assert (n_uv - 1.0) > (n_ir - 1.0)
