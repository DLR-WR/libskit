#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Vacuum ↔ air wavelength conversions (Ciddor / IAU standard)."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def vacuum_to_air(wavelengths_vacuum_nm: npt.ArrayLike) -> np.ndarray:
    """Convert vacuum wavelengths to standard-air wavelengths.

    Uses the IAU standard (Ciddor 1996) dispersion formula for the
    refractive index of dry air at standard conditions.

    Parameters
    ----------
    wavelengths_vacuum_nm : array_like
        Vacuum wavelengths in nanometers.

    Returns
    -------
    numpy.ndarray
        Air wavelengths in nanometers.

    Examples
    --------
    >>> vacuum_to_air(500.0)
    array(499.8...)
    """
    lam = np.asarray(wavelengths_vacuum_nm, dtype=np.float64)
    sigma2 = (1e3 / lam) ** 2  # (µm⁻¹)²
    n = 1.0 + 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)
    return lam / n


def air_to_vacuum(wavelengths_air_nm: npt.ArrayLike) -> np.ndarray:
    """Convert standard-air wavelengths to vacuum wavelengths.

    Iterative inversion of the Ciddor formula (one Newton step,
    accurate to < 10⁻¹² nm).

    Parameters
    ----------
    wavelengths_air_nm : array_like
        Air wavelengths in nanometers.

    Returns
    -------
    numpy.ndarray
        Vacuum wavelengths in nanometers.

    Examples
    --------
    >>> air_to_vacuum(500.0)
    array(500.1...)
    """
    lam_air = np.asarray(wavelengths_air_nm, dtype=np.float64)
    # Initial guess
    lam_vac = lam_air * 1.00028
    # One Newton step
    sigma2 = (1e3 / lam_vac) ** 2
    n = 1.0 + 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)
    lam_vac = lam_air * n
    return lam_vac
