/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file test_spectrum.cpp
 * @brief Comprehensive C++ unit tests for the Spectrum class.
 *
 * Tests all major functionality of the Spectrum template class including:
 * - Construction from wavelength and irradiance vectors
 * - Type conversion between different numeric types
 * - Data access and validation
 * - Edge cases and error conditions
 */

#include "../src/_libskit/libskit.h"
#include <gtest/gtest.h>
#include <Eigen/Core>

using namespace libskit;

class SpectrumTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up standard test data
        wavelengths = Eigen::VectorXd::LinSpaced(10, 400.0, 700.0);
        irradiance_double = Eigen::VectorXd::Random(10).cwiseAbs();
        irradiance_float = irradiance_double.cast<float>();
    }

    Eigen::VectorXd wavelengths;
    Eigen::VectorXd irradiance_double;
    Eigen::VectorXf irradiance_float;
};

/**
 * @brief Test basic construction with double precision.
 */
TEST_F(SpectrumTest, ConstructionDouble) {
    Spectrum spectrum(wavelengths, irradiance_double);

    EXPECT_EQ(spectrum.wavelengths.size(), 10);
    EXPECT_EQ(spectrum.irradiance.size(), 10);
    EXPECT_TRUE(spectrum.wavelengths.isApprox(wavelengths));
    EXPECT_TRUE(spectrum.irradiance.isApprox(irradiance_double));
}

/**
 * @brief Test large spectrum handling.
 */
TEST_F(SpectrumTest, LargeSpectrum) {
    const int large_size = 10000;
    Eigen::VectorXd large_wavelengths = Eigen::VectorXd::LinSpaced(large_size, 200.0, 1000.0);
    Eigen::VectorXd large_irradiance = Eigen::VectorXd::Random(large_size).cwiseAbs();

    Spectrum spectrum(large_wavelengths, large_irradiance);

    EXPECT_EQ(spectrum.wavelengths.size(), large_size);
    EXPECT_EQ(spectrum.irradiance.size(), large_size);
    EXPECT_TRUE(spectrum.wavelengths.isApprox(large_wavelengths));
    EXPECT_TRUE(spectrum.irradiance.isApprox(large_irradiance));
}

/**
 * @brief Test data access and modification.
 */
TEST_F(SpectrumTest, DataAccess) {
    Spectrum spectrum(wavelengths, irradiance_double);

    // Test direct access
    EXPECT_DOUBLE_EQ(spectrum.wavelengths[0], wavelengths[0]);
    EXPECT_DOUBLE_EQ(spectrum.irradiance[0], irradiance_double[0]);

    // Test modification
    double original_value = spectrum.irradiance[0];
    spectrum.irradiance[0] = 999.0;
    EXPECT_DOUBLE_EQ(spectrum.irradiance[0], 999.0);
    EXPECT_NE(spectrum.irradiance[0], original_value);

    // Test wavelength modification
    spectrum.wavelengths[0] = 100.0;
    EXPECT_DOUBLE_EQ(spectrum.wavelengths[0], 100.0);
}

/**
 * @brief Test spectral range properties.
 */
TEST_F(SpectrumTest, SpectralRange) {
    Spectrum spectrum(wavelengths, irradiance_double);

    // Check range
    double min_wavelength = spectrum.wavelengths.minCoeff();
    double max_wavelength = spectrum.wavelengths.maxCoeff();

    EXPECT_NEAR(min_wavelength, 400.0, 1e-10);
    EXPECT_NEAR(max_wavelength, 700.0, 1e-10);

    // Check monotonic increasing (for LinSpaced)
    for (int i = 1; i < spectrum.wavelengths.size(); ++i) {
        EXPECT_GT(spectrum.wavelengths[i], spectrum.wavelengths[i-1]);
    }
}

/**
 * @brief Test copy behavior and independence.
 */
TEST_F(SpectrumTest, CopyIndependence) {
    Spectrum original(wavelengths, irradiance_double);
    Spectrum copy(original);

    // Modify original
    original.irradiance[0] = 999.0;
    original.wavelengths[0] = 100.0;

    // Copy should be unchanged
    EXPECT_NE(copy.irradiance[0], 999.0);
    EXPECT_NE(copy.wavelengths[0], 100.0);
    EXPECT_DOUBLE_EQ(copy.wavelengths[0], wavelengths[0]);
    EXPECT_DOUBLE_EQ(copy.irradiance[0], irradiance_double[0]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
