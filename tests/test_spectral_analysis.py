# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""Tests for libskit.spectral_analysis core utilities."""

import numpy as np
import pytest

from libskit.spectral_analysis import (
    estimate_baseline_noise,
    find_spectrum_peaks,
    fit_voigt_profile,
)


class TestVoigtFitting:
    def test_recovers_centre_of_clean_gaussian(self) -> None:
        """Pure Gaussian (γ→0): fit must recover the centre to <1 pm."""
        rng = np.random.default_rng(0)
        wl = np.linspace(399.5, 400.5, 400)
        true_centre, sigma = 400.0, 0.05
        spectrum = (
            100.0 * np.exp(-0.5 * ((wl - true_centre) / sigma) ** 2)
            + 5.0
            + rng.normal(0, 0.1, wl.size)
        )
        result = fit_voigt_profile(wl, spectrum, true_centre, window=0.5)
        assert result["success"] is True
        assert abs(result["center"] - true_centre) < 1e-3

    def test_returns_failure_on_too_few_points(self) -> None:
        wl = np.linspace(399.999, 400.001, 5)
        spectrum = np.zeros_like(wl)
        result = fit_voigt_profile(wl, spectrum, 400.0, window=0.001)
        assert result["success"] is False
        assert "error" in result

    def test_handles_offset_baseline(self) -> None:
        """Fit should still recover centre on top of a non-zero baseline."""
        wl = np.linspace(499.5, 500.5, 400)
        spectrum = (
            50.0 * np.exp(-0.5 * ((wl - 500.0) / 0.05) ** 2)
            + 1000.0
        )
        result = fit_voigt_profile(wl, spectrum, 500.0, window=0.4)
        assert result["success"] is True
        assert abs(result["center"] - 500.0) < 1e-3


class TestBaselineNoise:
    def test_recovers_known_gaussian_noise(self) -> None:
        rng = np.random.default_rng(42)
        sigma_true = 0.05
        spectrum = rng.normal(0.0, sigma_true, 5000)
        sigma_est = estimate_baseline_noise(spectrum)
        # MAD-based estimator on differences should land within 30%
        assert 0.7 * sigma_true < sigma_est < 1.3 * sigma_true

    def test_returns_positive_for_clipped_spectrum(self) -> None:
        """Heavily clipped (mostly-zero) spectrum still returns a positive number."""
        spectrum = np.zeros(1000)
        spectrum[::100] = 1.0  # a few isolated peaks
        sigma = estimate_baseline_noise(spectrum)
        assert sigma > 0.0


class TestPeakDetection:
    def test_finds_synthetic_peaks(self) -> None:
        rng = np.random.default_rng(7)
        wl = np.arange(395.0, 405.0, 0.01)
        true_peaks = [397.5, 400.0, 402.7]
        spectrum = np.full_like(wl, 0.5)
        for c in true_peaks:
            spectrum += 50.0 * np.exp(-0.5 * ((wl - c) / 0.04) ** 2)
        spectrum += rng.normal(0, 0.05, wl.size)

        result = find_spectrum_peaks(wl, spectrum, height_fraction=0.05,
                                     prominence_fraction=0.02)
        peaks = result["wavelengths"]
        # Each true peak must have at least one detection within 0.05 nm
        for c in true_peaks:
            assert np.min(np.abs(peaks - c)) < 0.05, (
                f"missed peak near {c} nm; got {peaks}"
            )

    def test_returns_empty_for_flat_spectrum(self) -> None:
        wl = np.arange(400.0, 410.0, 0.05)
        flat = np.zeros_like(wl)
        result = find_spectrum_peaks(wl, flat, height_fraction=0.1,
                                     prominence_fraction=0.05)
        assert len(result["wavelengths"]) == 0
