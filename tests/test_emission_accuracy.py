#!/usr/bin/env python3
#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Test script for accuracy configuration.

This test suite verifies that the accuracy configuration function works
correctly. The accuracy parameter defines the fraction of the maximum peak
height below which emission lines are discarded and broadening wings are
truncated.
"""

import unittest
import sys
import os

# Add the project to the path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import _libskit as libskit


class TestAccuracy(unittest.TestCase):
    """Test cases for accuracy configuration."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.original_accuracy = libskit.get_accuracy()

    def tearDown(self):
        """Clean up after each test method."""
        libskit.set_accuracy(self.original_accuracy)

    def test_default_accuracy(self):
        """Test that the default accuracy is correct."""
        accuracy = libskit.get_accuracy()
        self.assertEqual(accuracy, 1e-5, "Default accuracy should be 1e-5")

    def test_set_accuracy(self):
        """Test setting and getting accuracy."""
        new_accuracy = 1e-4
        libskit.set_accuracy(new_accuracy)
        retrieved_accuracy = libskit.get_accuracy()
        self.assertEqual(retrieved_accuracy, new_accuracy,
                        f"Expected accuracy {new_accuracy}, got {retrieved_accuracy}")

    def test_multiple_accuracy_changes(self):
        """Test multiple changes to accuracy value."""
        for accuracy in [1e-2, 1e-7, 1e-10, 1e-1]:
            libskit.set_accuracy(accuracy)
            self.assertEqual(libskit.get_accuracy(), accuracy)

    def test_extreme_accuracy_values(self):
        """Test extreme accuracy values."""
        small_accuracy = 1e-15
        libskit.set_accuracy(small_accuracy)
        self.assertEqual(libskit.get_accuracy(), small_accuracy)

        large_accuracy = 1e-1
        libskit.set_accuracy(large_accuracy)
        self.assertEqual(libskit.get_accuracy(), large_accuracy)

    def test_accuracy_persistence(self):
        """Test that accuracy setting persists across multiple calls."""
        val = 5e-5
        libskit.set_accuracy(val)
        for _ in range(10):
            self.assertEqual(libskit.get_accuracy(), val)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Print summary
    print_test_summary()
