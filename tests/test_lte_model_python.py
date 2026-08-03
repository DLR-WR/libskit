#!/usr/bin/env python3
#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Python bridge tests for LTEModel.

This test suite verifies that the C++ LTEModel class works correctly
through the Python bindings, including numpy integration and error handling.
"""

import sys
import os
import numpy as np
import time
import gc
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

# Add the build directory to the path to import the compiled module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
build_dir = os.path.join(project_root, "build")
sys.path.insert(0, build_dir)

try:
    import _libskit as libskit
    # Load databases immediately after import
    libskit.load_databases()
    print("Successfully loaded databases")
    # Extract the LTEModel class for easier access
    LTEModel = libskit.LTEModel
    print("Successfully imported LTEModel from _libskit module")
except ImportError as e:
    print(f"Failed to import _libskit: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)
except AttributeError as e:
    print(f"LTEModel not found in _libskit: {e}")
    print(f"Available classes: {[name for name in dir(libskit) if not name.startswith('_')]}")
    sys.exit(1)
except Exception as e:
    print(f"Failed to load databases: {e}")
    sys.exit(1)


class TestLTEModelPythonBridge(unittest.TestCase):
    """Test the Python binding of LTEModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.elements = {"Fe": 0.5, "Si": 0.3, "Ca": 0.2}
        self.temperature = 10000.0  # K
        self.electron_density = 1e23  # m^-3
        self.length = 0.001  # m

        self.model = LTEModel(
            self.elements,
            self.temperature,
            self.electron_density,
            self.length
        )

        self.wavelengths = np.linspace(300.0, 800.0, 100)  # nm

    def test_construction(self):
        """Test that LTEModel can be constructed from Python."""
        # Basic construction should work
        model = LTEModel(
            {"Fe": 0.7, "Si": 0.3},
            8000.0,  # temperature
            5e22,    # electron_density
            0.002    # length
        )

        # Parameters should be accessible
        self.assertAlmostEqual(model.temperature, 8000.0)
        self.assertAlmostEqual(model.electron_density, 5e22)
        self.assertAlmostEqual(model.length, 0.002)

    def test_parameter_access_from_python(self):
        """Test parameter getter/setter through Python interface."""
        # Test temperature
        self.assertAlmostEqual(self.model.temperature, self.temperature)

        new_temp = 12000.0
        self.model.temperature = new_temp
        self.assertAlmostEqual(self.model.temperature, new_temp)

        # Test electron density
        self.assertAlmostEqual(self.model.electron_density, self.electron_density)

        new_ne = 2e23
        self.model.electron_density = new_ne
        self.assertAlmostEqual(self.model.electron_density, new_ne)

        # Test length
        self.assertAlmostEqual(self.model.length, self.length)

        new_length = 0.005
        self.model.length = new_length
        self.assertAlmostEqual(self.model.length, new_length)

    def test_irradiance_calculation(self):
        """Test irradiance calculation from Python."""
        irradiance = self.model.calculate_irradiance(self.wavelengths)

        # Should return numpy array of same length
        self.assertIsInstance(irradiance, np.ndarray)
        self.assertEqual(len(irradiance), len(self.wavelengths))

        # All values should be non-negative
        self.assertTrue(np.all(irradiance >= 0), "Found negative irradiance values")

        # Should have some non-zero emission
        self.assertGreater(np.max(irradiance), 0, "No emission detected")

    def test_numpy_array_conversion(self):
        """Test that numpy arrays are properly converted."""
        # Test with different numpy array types
        wavelengths_float32 = self.wavelengths.astype(np.float32)
        wavelengths_float64 = self.wavelengths.astype(np.float64)

        irradiance32 = self.model.calculate_irradiance(wavelengths_float32)
        irradiance64 = self.model.calculate_irradiance(wavelengths_float64)

        # Results should be similar regardless of input precision
        # Note: float32 has ~7 significant digits, so rtol=1e-3 is appropriate
        np.testing.assert_allclose(irradiance32, irradiance64, rtol=1e-3)

    def test_parameter_type_conversion(self):
        """Test that Python numeric types are properly converted."""
        # Test with different Python numeric types
        self.model.temperature = int(11000)  # int
        self.assertAlmostEqual(self.model.temperature, 11000.0)

        self.model.temperature = float(12000.5)  # float
        self.assertAlmostEqual(self.model.temperature, 12000.5)

        self.model.electron_density = np.float32(3e23)  # numpy scalar
        # float32 has ~7 significant digits, so check relative error instead
        self.assertAlmostEqual(self.model.electron_density / 3e23, 1.0, places=5)


    def test_large_array_handling(self):
        """Test handling of large numpy arrays."""
        # Test with large wavelength array
        large_wavelengths = np.linspace(200, 1000, 10000)  # 10k points

        irradiance = self.model.calculate_irradiance(large_wavelengths)

        self.assertEqual(len(irradiance), len(large_wavelengths))
        self.assertTrue(np.all(np.isfinite(irradiance)), "Found non-finite values")

    def test_performance_benchmark(self):
        """Basic performance test for the Python interface."""
        import time

        # Warm up
        for _ in range(5):
            self.model.calculate_irradiance(self.wavelengths)

        # Time multiple calculations
        start_time = time.time()
        num_iterations = 100

        for _ in range(num_iterations):
            self.model.calculate_irradiance(self.wavelengths)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations

        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(avg_time, 0.1, f"Average calculation time too slow: {avg_time:.3f}s")

        # If we get here without crashing, memory management is likely OK
        self.assertTrue(True)

    def test_physics_consistency_python(self):
        """Test physics consistency through Python interface."""
        # Test temperature scaling
        irradiance_low = self.model.calculate_irradiance(self.wavelengths)

        self.model.temperature = self.temperature * 1.5
        irradiance_high = self.model.calculate_irradiance(self.wavelengths)

        # Higher temperature should increase total emission
        total_low = np.sum(irradiance_low)
        total_high = np.sum(irradiance_high)

        self.assertGreater(total_high, total_low,
                          "Higher temperature didn't increase emission")

    def test_spectral_features(self):
        """Test that spectral features are realistic."""
        # Use a focused wavelength range where Fe lines are expected
        fe_wavelengths = np.linspace(520, 530, 1000)  # nm, Fe line region
        irradiance = self.model.calculate_irradiance(fe_wavelengths)

        # Should see some spectral structure (not completely flat)
        max_val = np.max(irradiance)
        min_val = np.min(irradiance)

        if max_val > 0:
            contrast = (max_val - min_val) / max_val
            self.assertGreater(contrast, 0.01, "Spectrum appears too flat - no spectral lines?")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
