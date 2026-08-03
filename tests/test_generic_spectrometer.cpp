/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file test_generic_spectrometer.cpp
 * @brief Comprehensive C++ unit tests for the GenericSpectrometer class.
 *
 * Tests all major functionality of the GenericSpectrometer template class including:
 * - Construction with broadening parameters
 * - Type conversion between different numeric types
 * - Cloning functionality
 * - Spectrum calculation with instrumental broadening
 * - Parameter validation and edge cases
 */

#include "../src/_libskit/libskit.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <memory>

using namespace libskit;

class GenericSpectrometerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load databases before any testing
        load_databases();

        // Standard broadening parameters for LIBS instruments
        sigma = 0.01;  // 0.01 nm Gaussian broadening
        gamma = 0.005; // 0.005 nm Lorentzian broadening

        // Standard test wavelength grid
        wavelengths = Eigen::VectorXd::LinSpaced(100, 400.0, 700.0);

        // Create a simple LTE plasma model for testing
        std::map<std::string, double> element_ratios = {{"Fe", 1.0}};
        plasma_model = std::make_shared<LTEModel>(
            element_ratios, 10000.0, 1e23, 1e-3
        );
    }

    double sigma, gamma;
    Eigen::VectorXd wavelengths;
    std::shared_ptr<LTEModel> plasma_model;
};

/**
 * @brief Test basic construction with broadening parameters.
 */
TEST_F(GenericSpectrometerTest, BasicConstruction) {
    GenericSpectrometer spectrometer(sigma, gamma);

    EXPECT_DOUBLE_EQ(spectrometer.sigma, sigma);
    EXPECT_DOUBLE_EQ(spectrometer.gamma, gamma);
}

/**
 * @brief Test cloning functionality.
 */
TEST_F(GenericSpectrometerTest, Cloning) {
    GenericSpectrometer original(sigma, gamma);
    auto cloned = original.clone();

    // Should be different objects
    EXPECT_NE(&original, cloned.get());

    // Cast to GenericSpectrometer to access parameters
    auto cloned_generic = std::dynamic_pointer_cast<GenericSpectrometer>(cloned);
    ASSERT_NE(cloned_generic, nullptr);

    EXPECT_DOUBLE_EQ(cloned_generic->sigma, original.sigma);
    EXPECT_DOUBLE_EQ(cloned_generic->gamma, original.gamma);
}

/**
 * @brief Test parameter modification.
 */
TEST_F(GenericSpectrometerTest, ParameterModification) {
    GenericSpectrometer spectrometer(sigma, gamma);

    // Modify parameters
    double new_sigma = 0.02;
    double new_gamma = 0.01;

    spectrometer.sigma = new_sigma;
    spectrometer.gamma = new_gamma;

    EXPECT_DOUBLE_EQ(spectrometer.sigma, new_sigma);
    EXPECT_DOUBLE_EQ(spectrometer.gamma, new_gamma);
}

/**
 * @brief Test spectrum calculation produces reasonable output.
 * Note: This test verifies that the function runs without errors and produces
 * output of correct dimensions, but doesn't validate the full FFT convolution.
 */
TEST_F(GenericSpectrometerTest, SpectrumCalculationBasic) {
    GenericSpectrometer spectrometer(sigma, gamma);

    // Calculate spectrum - this should not throw
    Eigen::VectorXd result;
    EXPECT_NO_THROW({
        result = spectrometer.calculate_spectrum(wavelengths, *plasma_model);
    });

    // Should have correct size
    EXPECT_EQ(result.size(), wavelengths.size());

    // Should have non-negative values (physical constraint)
    EXPECT_TRUE((result.array() >= 0.0).all());

    // Should not be all zeros (assuming non-zero plasma emission)
    EXPECT_GT(result.sum(), 0.0);
}

/**
 * @brief Test spectrum calculation with different broadening parameters.
 */
TEST_F(GenericSpectrometerTest, SpectrumCalculationBroadeningEffect) {
    GenericSpectrometer narrow_spectrometer(0.001, 0.001);
    GenericSpectrometer broad_spectrometer(0.1, 0.1);

    Eigen::VectorXd narrow_result, broad_result;

    EXPECT_NO_THROW({
        narrow_result = narrow_spectrometer.calculate_spectrum(wavelengths, *plasma_model);
        broad_result = broad_spectrometer.calculate_spectrum(wavelengths, *plasma_model);
    });

    EXPECT_EQ(narrow_result.size(), wavelengths.size());
    EXPECT_EQ(broad_result.size(), wavelengths.size());

    // Both should be non-negative
    EXPECT_TRUE((narrow_result.array() >= 0.0).all());
    EXPECT_TRUE((broad_result.array() >= 0.0).all());

    // Results should be different (different broadening)
    EXPECT_FALSE(narrow_result.isApprox(broad_result, 1e-10));
}

/**
 * @brief Test parameter independence between instances.
 */
TEST_F(GenericSpectrometerTest, ParameterIndependence) {
    GenericSpectrometer spec1(0.01, 0.005);
    GenericSpectrometer spec2(0.02, 0.010);

    EXPECT_NE(spec1.sigma, spec2.sigma);
    EXPECT_NE(spec1.gamma, spec2.gamma);

    // Modify one, check other is unchanged
    double original_sigma = spec2.sigma;
    spec1.sigma = 0.99;
    EXPECT_DOUBLE_EQ(spec2.sigma, original_sigma);
}

/**
 * @brief Test inheritance from Instrument base class.
 */
TEST_F(GenericSpectrometerTest, InheritanceInterface) {
    auto spectrometer = std::make_shared<GenericSpectrometer>(sigma, gamma);
    std::shared_ptr<Instrument> instrument_ptr = spectrometer;

    // Should be able to call virtual functions through base pointer
    Eigen::VectorXd result;
    EXPECT_NO_THROW({
        result = instrument_ptr->calculate_spectrum(wavelengths, *plasma_model);
    });

    EXPECT_EQ(result.size(), wavelengths.size());

    // Should be able to clone through base pointer
    auto cloned = instrument_ptr->clone();
    EXPECT_NE(cloned.get(), instrument_ptr.get());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
