#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Common utilities for libskit Python tests.

This module provides shared functionality for testing the Python bindings,
including test data generation, assertion helpers, and performance utilities.
"""

import numpy as np
import time
from contextlib import contextmanager
from typing import Dict, Tuple, Any


def create_standard_test_plasma() -> Tuple[Dict[str, float], float, float, float]:
    """
    Create standard test plasma parameters.

    Returns:
        Tuple of (elements, temperature, electron_density, length)
    """
    elements = {"Fe": 0.5, "Si": 0.3, "Ca": 0.2}
    temperature = 10000.0  # K
    electron_density = 1e23  # m^-3
    length = 0.001  # m

    return elements, temperature, electron_density, length


def create_test_wavelengths(start: float = 300.0, end: float = 800.0,
                          num_points: int = 100) -> np.ndarray:
    """
    Create standard test wavelength array.

    Args:
        start: Starting wavelength in nm
        end: Ending wavelength in nm
        num_points: Number of wavelength points

    Returns:
        Numpy array of wavelengths
    """
    return np.linspace(start, end, num_points)


def assert_spectrum_validity(spectrum: np.ndarray, wavelengths: np.ndarray) -> None:
    """
    Assert that a spectrum meets basic validity requirements.

    Args:
        spectrum: Spectrum intensity array
        wavelengths: Corresponding wavelength array
    """
    assert isinstance(spectrum, np.ndarray), "Spectrum must be numpy array"
    assert len(spectrum) == len(wavelengths), "Spectrum and wavelength lengths must match"
    assert np.all(spectrum >= 0), "Spectrum values must be non-negative"
    assert np.all(np.isfinite(spectrum)), "Spectrum must contain only finite values"


def assert_physical_consistency(model, wavelengths: np.ndarray) -> None:
    """
    Assert basic physical consistency of a plasma model.

    Args:
        model: LTEModel instance
        wavelengths: Wavelength array for calculations
    """
    # Temperature scaling test
    original_temp = model.temperature
    spectrum_low = model.calculate_irradiance(wavelengths)

    model.temperature = original_temp * 1.2
    spectrum_high = model.calculate_irradiance(wavelengths)

    # Restore original temperature
    model.temperature = original_temp

    # Higher temperature should generally increase emission
    total_low = np.sum(spectrum_low)
    total_high = np.sum(spectrum_high)

    assert total_high > total_low, "Higher temperature should increase total emission"


@contextmanager
def timing_context(test_name: str):
    """
    Context manager for timing test operations.

    Args:
        test_name: Name of the test being timed
    """
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{test_name} completed in {end_time - start_time:.3f} seconds")


def generate_noisy_spectrum(clean_spectrum: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add realistic noise to a clean spectrum.

    Args:
        clean_spectrum: Clean spectrum array
        noise_level: Relative noise level (0.01 = 1%)

    Returns:
        Noisy spectrum array
    """
    max_intensity = np.max(clean_spectrum)
    noise = np.random.normal(0, noise_level * max_intensity, len(clean_spectrum))
    return clean_spectrum + noise


class ParameterSweep:
    """Helper class for testing parameter sensitivity."""

    def __init__(self, model_class):
        self.model_class = model_class
        self.base_params = create_standard_test_plasma()

    def temperature_sweep(self, temp_range: Tuple[float, float],
                         num_points: int = 5) -> Dict[float, Any]:
        """
        Sweep temperature and return results.

        Args:
            temp_range: (min_temp, max_temp) in Kelvin
            num_points: Number of temperature points

        Returns:
            Dictionary mapping temperature to model results
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], num_points)
        results = {}

        elements, _, electron_density, length = self.base_params

        for temp in temperatures:
            model = self.model_class(elements, temp, electron_density, length)
            results[temp] = model

        return results

    def density_sweep(self, density_range: Tuple[float, float],
                     num_points: int = 5) -> Dict[float, Any]:
        """
        Sweep electron density and return results.

        Args:
            density_range: (min_density, max_density) in m^-3
            num_points: Number of density points

        Returns:
            Dictionary mapping electron density to model results
        """
        densities = np.linspace(density_range[0], density_range[1], num_points)
        results = {}

        elements, temperature, _, length = self.base_params

        for density in densities:
            model = self.model_class(elements, temperature, density, length)
            results[density] = model

        return results


def compare_spectra(spectrum1: np.ndarray, spectrum2: np.ndarray,
                   tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two spectra and return comparison metrics.

    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        tolerance: Tolerance for similarity testing

    Returns:
        Dictionary with comparison metrics
    """
    assert len(spectrum1) == len(spectrum2), "Spectra must have same length"

    # Mean squared error
    mse = np.mean((spectrum1 - spectrum2)**2)

    # Relative error
    max_val = max(np.max(spectrum1), np.max(spectrum2))
    relative_error = mse / (max_val**2) if max_val > 0 else 0

    # Correlation coefficient
    correlation = np.corrcoef(spectrum1, spectrum2)[0, 1] if len(spectrum1) > 1 else 1.0

    # Are they similar within tolerance?
    are_similar = relative_error < tolerance

    return {
        'mse': mse,
        'relative_error': relative_error,
        'correlation': correlation,
        'are_similar': are_similar,
        'max_difference': np.max(np.abs(spectrum1 - spectrum2))
    }


def stress_test_model(model, num_iterations: int = 100) -> Dict[str, Any]:
    """
    Perform stress testing on a model.

    Args:
        model: Model to test
        num_iterations: Number of iterations to run

    Returns:
        Dictionary with stress test results
    """
    wavelengths = create_test_wavelengths()

    times = []
    memory_usage = []

    for i in range(num_iterations):
        start_time = time.time()

        # Perform calculation
        spectrum = model.calculate_irradiance(wavelengths)

        end_time = time.time()
        times.append(end_time - start_time)

        # Basic validation
        assert_spectrum_validity(spectrum, wavelengths)

    return {
        'num_iterations': num_iterations,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times)
    }


# Constants for testing
TYPICAL_LIBS_CONDITIONS = {
    'temperature_range': (5000, 20000),  # K
    'electron_density_range': (1e22, 1e24),  # m^-3
    'length_range': (1e-4, 1e-2),  # m
    'wavelength_range': (200, 1000)  # nm
}
