#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Extended Spectrum class with convenience methods."""
from __future__ import annotations

import numpy as np

from ._libskit import Spectrum as _CppSpectrum
from .wavelengths import vacuum_to_air, air_to_vacuum


class Spectrum(_CppSpectrum):
    """Spectrum with wavelength-intensity pairs and convenience methods.

    Subclasses the C++ Spectrum to add Python-level utilities for
    wavelength conversion, cropping, and normalisation. All methods
    return new Spectrum objects (immutable style).

    Parameters
    ----------
    wavelengths : numpy.ndarray
        1D array of wavelength values in nanometers.
    irradiance : numpy.ndarray
        1D array of intensity/irradiance values.

    Examples
    --------
    >>> import numpy as np
    >>> from libskit import Spectrum
    >>> wl = np.linspace(300, 800, 5000)
    >>> intensity = np.exp(-0.5 * ((wl - 550) / 10) ** 2)
    >>> s = Spectrum(wl, intensity)
    >>> cropped = s.crop(500, 600)
    >>> normed = s.normalize('max')
    """

    def to_air(self) -> Spectrum:
        """Return a new Spectrum with wavelengths converted from vacuum to air."""
        return Spectrum(vacuum_to_air(self.wavelengths), self.irradiance)

    def to_vacuum(self) -> Spectrum:
        """Return a new Spectrum with wavelengths converted from air to vacuum."""
        return Spectrum(air_to_vacuum(self.wavelengths), self.irradiance)

    def crop(self, wl_min: float, wl_max: float) -> Spectrum:
        """Return a new Spectrum restricted to [wl_min, wl_max]."""
        mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
        return Spectrum(self.wavelengths[mask], self.irradiance[mask])

    def normalize(self, method: str = "max") -> Spectrum:
        """Return a new Spectrum with normalised irradiance.

        Parameters
        ----------
        method : {'max', 'area'}
            ``'max'`` divides by the peak value, ``'area'`` divides by
            the integrated area (trapezoid rule).
        """
        if method == "max":
            scale = np.max(np.abs(self.irradiance))
        elif method == "area":
            scale = np.trapz(self.irradiance, self.wavelengths)
        else:
            raise ValueError(f"Unknown normalisation method: {method!r}")
        if scale == 0:
            return Spectrum(self.wavelengths.copy(), self.irradiance.copy())
        return Spectrum(self.wavelengths.copy(), self.irradiance / scale)
