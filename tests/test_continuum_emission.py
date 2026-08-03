#!/usr/bin/env python3
#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Tests for continuum emission (bremsstrahlung + recombination radiation).

Verifies the implementation of Noll (2012), Eqs. 10.14–10.18 in LTEModel.
"""

import unittest
import sys
import os
import numpy as np

# Add the build directory to the path to import the compiled module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
build_dir = os.path.join(project_root, "build")
sys.path.insert(0, build_dir)

import _libskit as libskit


class TestContinuumEmission(unittest.TestCase):
    """Test continuum emission feature of LTEModel."""

    def setUp(self):
        self.elements = {"Fe": 0.5, "Si": 0.3, "Ca": 0.2}
        self.temperature = 10000.0
        self.electron_density = 1e23
        self.length = 1e-3
        self.wavelengths = np.arange(250.0, 900.0, 0.1)

        self.model = libskit.LTEModel(
            self.elements, self.temperature,
            self.electron_density, self.length
        )

    def test_default_flag_is_false(self):
        """include_continuum should default to False."""
        self.assertFalse(self.model.include_continuum)

    def test_backward_compatibility(self):
        """Spectra with include_continuum=False must be unchanged."""
        irr_default = self.model.calculate_irradiance(self.wavelengths)
        self.model.include_continuum = False
        irr_explicit = self.model.calculate_irradiance(self.wavelengths)
        np.testing.assert_array_equal(irr_default, irr_explicit)

    def test_continuum_adds_positive_contribution(self):
        """Continuum emission should be >= 0 everywhere."""
        irr_no = self.model.calculate_irradiance(self.wavelengths)
        self.model.include_continuum = True
        irr_yes = self.model.calculate_irradiance(self.wavelengths)
        diff = irr_yes - irr_no
        self.assertTrue(np.all(diff >= -1e-30),
                        "Continuum contribution must be non-negative")

    def test_continuum_magnitude_reasonable(self):
        """Continuum should be much weaker than the strongest lines."""
        irr_no = self.model.calculate_irradiance(self.wavelengths)
        self.model.include_continuum = True
        irr_yes = self.model.calculate_irradiance(self.wavelengths)
        diff = irr_yes - irr_no
        ratio = np.max(diff) / np.max(irr_no)
        # Continuum peak should be < 10% of max line peak
        self.assertLess(ratio, 0.1,
                        f"Continuum-to-line ratio {ratio:.4f} unexpectedly large")
        # But should be non-trivial (> 1e-6)
        self.assertGreater(ratio, 1e-6,
                           f"Continuum-to-line ratio {ratio:.2e} unexpectedly small")

    def test_continuum_increases_toward_red(self):
        """Bremsstrahlung continuum should generally increase with wavelength."""
        irr_no = self.model.calculate_irradiance(self.wavelengths)
        self.model.include_continuum = True
        irr_yes = self.model.calculate_irradiance(self.wavelengths)
        diff = irr_yes - irr_no

        # Compare mean continuum in blue (250-400 nm) vs red (600-850 nm)
        blue_mask = self.wavelengths < 400
        red_mask = self.wavelengths > 600
        mean_blue = np.mean(diff[blue_mask])
        mean_red = np.mean(diff[red_mask])
        self.assertGreater(mean_red, mean_blue,
                           "Continuum should be stronger in the red than blue")

    def test_continuum_monotonic_trend(self):
        """Continuum contribution should vary smoothly across broad regions."""
        # Use a coarse grid (10 nm steps) to average over line features.
        # The RTE is nonlinear so irradiance difference has small modulations
        # near strong lines — that's physically correct, not a bug.
        wl_coarse = np.arange(300.0, 850.0, 10.0)
        self.model.include_continuum = False
        irr_lines = self.model.calculate_irradiance(wl_coarse)
        self.model.include_continuum = True
        irr_total = self.model.calculate_irradiance(wl_coarse)
        continuum = irr_total - irr_lines

        # On a 10 nm grid the continuum should be nearly monotonic
        # (increasing toward red). Allow a few dips from line-coupling.
        diffs = np.diff(continuum)
        frac_increasing = np.sum(diffs > 0) / len(diffs)
        self.assertGreater(frac_increasing, 0.7,
                           f"Continuum should mostly increase with wavelength, "
                           f"but only {frac_increasing:.0%} of steps were positive")

    def test_single_element(self):
        """Continuum should work with a single element."""
        model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
        model.include_continuum = True
        irr = model.calculate_irradiance(self.wavelengths)
        self.assertTrue(np.all(np.isfinite(irr)), "Irradiance should be finite")
        self.assertTrue(np.any(irr > 0), "Irradiance should have positive values")

    def test_high_temperature(self):
        """Continuum should work and be stronger at higher temperatures."""
        wl = np.arange(300.0, 800.0, 1.0)

        model_low = libskit.LTEModel({"Fe": 1.0}, 8000.0, 1e23, 1e-3)
        model_high = libskit.LTEModel({"Fe": 1.0}, 15000.0, 1e23, 1e-3)

        # Get continuum-only contributions via irradiance difference
        irr_low_no = model_low.calculate_irradiance(wl)
        model_low.include_continuum = True
        irr_low_yes = model_low.calculate_irradiance(wl)
        cont_low = np.mean(irr_low_yes - irr_low_no)

        irr_high_no = model_high.calculate_irradiance(wl)
        model_high.include_continuum = True
        irr_high_yes = model_high.calculate_irradiance(wl)
        cont_high = np.mean(irr_high_yes - irr_high_no)

        self.assertGreater(cont_high, cont_low,
                           "Continuum should be stronger at higher temperature")


if __name__ == "__main__":
    unittest.main()
