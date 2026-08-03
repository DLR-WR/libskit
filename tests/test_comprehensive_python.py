#!/usr/bin/env python3
#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Comprehensive Python integration tests for libskit components.

This script tests the Python bridge functionality for:
- Spectrum class (wavelength/irradiance data structure)
- GenericSpectrometer class (instrument simulation)
- LTEModel class (plasma physics calculations)
- Utility functions (blackbody radiation)
- Integration between components

Note: These tests may segfault in some local development environments
due to compiler/library version differences, but should pass in CI.
The import structure prioritizes the build directory to avoid conflicts
with any pip-installed versions of the package.

These tests verify that the C++/Python binding works correctly and that
data can be passed between Python and C++ components without issues.
"""

import os
import sys
import unittest

import numpy as np

# Add the build directory to the path FIRST to ensure we use the latest build
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
build_dir = os.path.join(project_root, "build")
src_dir = os.path.join(project_root, "src")

# Insert build directory at the beginning of sys.path to prioritize it
sys.path.insert(0, build_dir)
# Also expose the in-tree Python package (src/libskit/) so `import libskit` works
# without requiring a pip install. ctest invokes this script directly with python,
# so conftest.py is not consulted -- we replicate its aliasing here.
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Strip any scikit-build editable-install finder that would redirect `libskit`
# to a different worktree (e.g. /main/src) installed in the active venv.
sys.meta_path[:] = [
    f for f in sys.meta_path
    if type(f).__name__ != "ScikitBuildRedirectingFinder"
]

# Pre-load the compiled extension from build/ and alias it as libskit._libskit
# so that libskit/__init__.py's `from ._libskit import ...` resolves to the
# build-tree binary (avoiding pybind11 "type already registered" double-loads
# and side-stepping the need for an installed copy of the package).
try:
    import _libskit as _ext

    sys.modules.setdefault("libskit._libskit", _ext)
except ImportError:
    pass  # fall through; the next try-block reports the real error

# Import the libskit module via its public package path
try:
    from libskit import _libskit

    print(
        f"Successfully imported libskit._libskit from: {getattr(_libskit, '__file__', 'unknown location')}"
    )
    # Load databases immediately after import to ensure they're available for tests
    _libskit.load_databases(0)
    print("Successfully loaded databases")
except ImportError as e:
    print(f"Failed to from libskit import _libskit module: {e}")
    print(f"Python path: {sys.path}")
    print(
        f"Build directory contents: {os.listdir(build_dir) if os.path.exists(build_dir) else 'Not found'}"
    )
    sys.exit(1)
except Exception as e:
    print(f"Failed to load databases: {e}")
    sys.exit(1)


class TestSpectrumPython(unittest.TestCase):
    """Test Spectrum class Python integration."""

    def setUp(self):
        """Set up test data."""
        self.wavelengths = np.linspace(400.0, 700.0, 100)
        self.irradiance = np.abs(np.random.randn(100))

    def test_spectrum_construction(self):
        """Test basic Spectrum construction."""
        spectrum = _libskit.Spectrum(self.wavelengths, self.irradiance)

        self.assertEqual(len(spectrum.wavelengths), 100)
        self.assertEqual(len(spectrum.irradiance), 100)
        np.testing.assert_array_almost_equal(spectrum.wavelengths, self.wavelengths)
        np.testing.assert_array_almost_equal(spectrum.irradiance, self.irradiance)

    def test_spectrum_data_access(self):
        """Test accessing and modifying Spectrum data."""
        spectrum = _libskit.Spectrum(self.wavelengths, self.irradiance)

        # Test direct access
        self.assertAlmostEqual(spectrum.wavelengths[0], self.wavelengths[0])
        self.assertAlmostEqual(spectrum.irradiance[0], self.irradiance[0])

        # Test array replacement (more realistic usage pattern)
        original_irradiance = spectrum.irradiance.copy()
        new_irradiance = np.ones_like(self.irradiance) * 999.0
        spectrum.irradiance = new_irradiance

        # Verify the replacement worked
        np.testing.assert_array_almost_equal(spectrum.irradiance, new_irradiance)
        self.assertAlmostEqual(spectrum.irradiance[0], 999.0)
        # Verify the original data is different
        self.assertNotAlmostEqual(spectrum.irradiance[0], original_irradiance[0])

    def test_spectrum_with_numpy_arrays(self):
        """Test Spectrum with various NumPy array types."""
        # Test with different dtypes
        wavelengths_float32 = self.wavelengths.astype(np.float32)
        irradiance_float32 = self.irradiance.astype(np.float32)

        spectrum = _libskit.Spectrum(wavelengths_float32, irradiance_float32)
        self.assertEqual(len(spectrum.wavelengths), 100)
        self.assertEqual(len(spectrum.irradiance), 100)

    def test_spectrum_empty(self):
        """Test Spectrum with empty arrays."""
        empty_wavelengths = np.array([])
        empty_irradiance = np.array([])

        spectrum = _libskit.Spectrum(empty_wavelengths, empty_irradiance)
        self.assertEqual(len(spectrum.wavelengths), 0)
        self.assertEqual(len(spectrum.irradiance), 0)


class TestGenericSpectrometerPython(unittest.TestCase):
    """Test GenericSpectrometer class Python integration."""

    def setUp(self):
        """Set up test data."""
        self.sigma = 0.01  # Gaussian broadening
        self.gamma = 0.005  # Lorentzian broadening
        self.wavelengths = np.linspace(400.0, 700.0, 50)

        # Create a simple LTE model for testing
        element_ratios = {"Fe": 1.0}
        self.lte_model = _libskit.LTEModel(element_ratios, 10000.0, 1e23, 1e-3)

    def test_spectrometer_construction(self):
        """Test basic GenericSpectrometer construction."""
        spectrometer = _libskit.GenericSpectrometer(self.sigma, self.gamma)

        self.assertAlmostEqual(spectrometer.sigma, self.sigma)
        self.assertAlmostEqual(spectrometer.gamma, self.gamma)

    def test_spectrometer_parameter_access(self):
        """Test accessing and modifying spectrometer parameters."""
        spectrometer = _libskit.GenericSpectrometer(self.sigma, self.gamma)

        # Test parameter access
        original_sigma = spectrometer.sigma
        original_gamma = spectrometer.gamma

        # Test parameter modification
        spectrometer.sigma = 0.02
        spectrometer.gamma = 0.01

        self.assertAlmostEqual(spectrometer.sigma, 0.02)
        self.assertAlmostEqual(spectrometer.gamma, 0.01)
        self.assertNotAlmostEqual(spectrometer.sigma, original_sigma)
        self.assertNotAlmostEqual(spectrometer.gamma, original_gamma)

    def test_spectrometer_spectrum_calculation(self):
        """Test spectrum calculation with GenericSpectrometer."""
        spectrometer = _libskit.GenericSpectrometer(self.sigma, self.gamma)

        # Calculate spectrum
        result = spectrometer.calculate_spectrum(self.wavelengths, self.lte_model)

        # Check result properties
        self.assertEqual(len(result), len(self.wavelengths))
        self.assertTrue(all(r >= 0.0 for r in result))  # Non-negative
        self.assertTrue(np.all(np.isfinite(result)))  # Finite values
        self.assertGreater(np.sum(result), 0.0)  # Not all zeros

    def test_spectrometer_different_broadening(self):
        """Test different broadening parameters produce different results."""
        narrow_spec = _libskit.GenericSpectrometer(0.001, 0.001)
        broad_spec = _libskit.GenericSpectrometer(0.1, 0.1)

        narrow_result = narrow_spec.calculate_spectrum(self.wavelengths, self.lte_model)
        broad_result = broad_spec.calculate_spectrum(self.wavelengths, self.lte_model)

        # Results should be different
        self.assertFalse(np.allclose(narrow_result, broad_result, rtol=1e-10))

        # Both should be non-negative
        self.assertTrue(all(r >= 0.0 for r in narrow_result))
        self.assertTrue(all(r >= 0.0 for r in broad_result))

    def test_spectrometer_with_numpy_arrays(self):
        """Test spectrometer with various NumPy array types."""
        spectrometer = _libskit.GenericSpectrometer(self.sigma, self.gamma)

        # Test with different array dtypes
        wavelengths_float32 = self.wavelengths.astype(np.float32)
        result = spectrometer.calculate_spectrum(wavelengths_float32, self.lte_model)

        self.assertEqual(len(result), len(wavelengths_float32))
        self.assertTrue(all(r >= 0.0 for r in result))


class TestUtilityFunctionsPython(unittest.TestCase):
    """Test utility functions Python integration."""

    def test_blackbody_irradiance(self):
        """Test blackbody irradiance calculation."""
        wavelength = 550.0  # nm
        temperature = 5778.0  # K (Sun's surface)

        intensity = _libskit.blackbody_irradiance(wavelength, temperature)

        self.assertGreater(intensity, 0.0)
        self.assertTrue(np.isfinite(intensity))


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components."""

    def setUp(self):
        """Set up components for integration testing."""
        self.wavelengths = np.linspace(400.0, 700.0, 30)

        # Create LTE model
        element_ratios = {"Fe": 0.7, "Si": 0.3}
        self.lte_model = _libskit.LTEModel(element_ratios, 10000.0, 1e23, 1e-3)

        # Create spectrometer
        self.spectrometer = _libskit.GenericSpectrometer(0.01, 0.005)

    def test_spectrometer_lte_integration(self):
        """Test GenericSpectrometer with LTEModel integration."""
        # Calculate spectrum through spectrometer
        spectrum_result = self.spectrometer.calculate_spectrum(
            self.wavelengths, self.lte_model
        )

        # Calculate direct irradiance for comparison
        # Note: We can't use calculate_irradiance due to the known segfault
        # so we use calculate_irradiance_test instead
        direct_result = _libskit.calculate_irradiance_test(
            self.lte_model, self.wavelengths
        )

        # Both should have same length
        self.assertEqual(len(spectrum_result), len(direct_result))
        self.assertEqual(len(spectrum_result), len(self.wavelengths))

        # Both should be non-negative
        self.assertTrue(all(r >= 0.0 for r in spectrum_result))
        self.assertTrue(all(r >= 0.0 for r in direct_result))

        # Spectrometer result should be different from direct (due to broadening)
        self.assertFalse(np.allclose(spectrum_result, direct_result, rtol=1e-6))

    def test_spectrum_data_structure_integration(self):
        """Test Spectrum data structure with calculated results."""
        # Calculate spectrum
        irradiance_data = self.spectrometer.calculate_spectrum(
            self.wavelengths, self.lte_model
        )

        # Create Spectrum object
        spectrum = _libskit.Spectrum(self.wavelengths, irradiance_data)

        # Verify data integrity
        np.testing.assert_array_almost_equal(spectrum.wavelengths, self.wavelengths)
        np.testing.assert_array_almost_equal(spectrum.irradiance, irradiance_data)

        # Test that we can modify the spectrum by replacing the array
        original_sum = np.sum(spectrum.irradiance)
        modified_irradiance = spectrum.irradiance.copy()
        modified_irradiance[0] = modified_irradiance[0] + 100.0
        spectrum.irradiance = modified_irradiance
        modified_sum = np.sum(spectrum.irradiance)

        self.assertNotAlmostEqual(original_sum, modified_sum)
        self.assertAlmostEqual(modified_sum - original_sum, 100.0)

    def test_data_type_consistency(self):
        """Test data type consistency across components."""
        # All wavelength arrays should work consistently
        wavelengths_float64 = self.wavelengths.astype(np.float64)
        wavelengths_float32 = self.wavelengths.astype(np.float32)

        # Test with float64
        result64 = self.spectrometer.calculate_spectrum(
            wavelengths_float64, self.lte_model
        )
        spectrum64 = _libskit.Spectrum(wavelengths_float64, result64)

        # Test with float32
        result32 = self.spectrometer.calculate_spectrum(
            wavelengths_float32, self.lte_model
        )
        spectrum32 = _libskit.Spectrum(wavelengths_float32, result32)

        # Results should be similar (within float32 precision)
        np.testing.assert_allclose(
            result64, result32, rtol=1e-3
        )  # Relaxed tolerance for float32 precision

        # Spectra should be comparable
        self.assertEqual(len(spectrum64.wavelengths), len(spectrum32.wavelengths))
        self.assertEqual(len(spectrum64.irradiance), len(spectrum32.irradiance))


if __name__ == "__main__":
    # Set up test suite
    test_classes = [
        TestSpectrumPython,
        TestGenericSpectrometerPython,
        TestUtilityFunctionsPython,
        TestComponentIntegration,
    ]

    # Run tests
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    sys.exit(exit_code)
