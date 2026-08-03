/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file test_utility_functions.cpp
 * @brief Comprehensive C++ unit tests for utility functions in libskit.
 *
 * Tests utility functions including:
 * - Blackbody radiation calculations
 * - Voigt profile approximations
 * - Mathematical consistency and edge cases
 */

#include "../src/_libskit/libskit.h"
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace libskit;

class UtilityFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard test parameters
        typical_wavelength = 550.0;  // nm (green light)
        typical_temperature = 5778.0; // K (Sun's surface temperature)
        room_temperature = 293.15;   // K

        // Voigt profile parameters
        sigma = 0.1;  // Gaussian width
        gamma = 0.05; // Lorentzian width
    }

    double typical_wavelength;
    double typical_temperature;
    double room_temperature;
    double sigma, gamma;
};

/**
 * @brief Test blackbody radiation with typical values.
 */
TEST_F(UtilityFunctionsTest, BlackbodyRadiationTypical) {
    double intensity = blackbody_radiation(typical_wavelength, typical_temperature);

    EXPECT_GT(intensity, 0.0);
    EXPECT_LT(intensity, 1e20);  // Should be finite
    EXPECT_TRUE(std::isfinite(intensity));
}

/**
 * @brief Test blackbody radiation Wien's displacement law.
 * Peak wavelength should be inversely proportional to temperature.
 */
TEST_F(UtilityFunctionsTest, BlackbodyWiensLaw) {
    double temp1 = 3000.0;  // K
    double temp2 = 6000.0;  // K (twice temp1)

    // Calculate intensities at several wavelengths to find approximate peak
    std::vector<double> wavelengths = {400, 500, 600, 700, 800, 900, 1000};

    double max_intensity_temp1 = 0.0;
    double peak_wavelength_temp1 = 0.0;
    double max_intensity_temp2 = 0.0;
    double peak_wavelength_temp2 = 0.0;

    for (double wl : wavelengths) {
        double intensity1 = blackbody_radiation(wl, temp1);
        double intensity2 = blackbody_radiation(wl, temp2);

        if (intensity1 > max_intensity_temp1) {
            max_intensity_temp1 = intensity1;
            peak_wavelength_temp1 = wl;
        }

        if (intensity2 > max_intensity_temp2) {
            max_intensity_temp2 = intensity2;
            peak_wavelength_temp2 = wl;
        }
    }

    // Higher temperature should peak at shorter wavelength
    EXPECT_LT(peak_wavelength_temp2, peak_wavelength_temp1);
}

/**
 * @brief Test blackbody radiation with extreme temperatures.
 */
TEST_F(UtilityFunctionsTest, BlackbodyExtremeTemperatures) {
    // Very low temperature
    double low_temp_intensity = blackbody_radiation(typical_wavelength, 100.0);
    EXPECT_GT(low_temp_intensity, 0.0);
    EXPECT_TRUE(std::isfinite(low_temp_intensity));

    // Very high temperature
    double high_temp_intensity = blackbody_radiation(typical_wavelength, 50000.0);
    EXPECT_GT(high_temp_intensity, 0.0);
    EXPECT_TRUE(std::isfinite(high_temp_intensity));

    // High temperature should give much higher intensity
    EXPECT_GT(high_temp_intensity, low_temp_intensity);
}

/**
 * @brief Test blackbody radiation with extreme wavelengths.
 */
TEST_F(UtilityFunctionsTest, BlackbodyExtremeWavelengths) {
    // Very short wavelength (UV)
    double uv_intensity = blackbody_radiation(100.0, typical_temperature);
    EXPECT_GT(uv_intensity, 0.0);
    EXPECT_TRUE(std::isfinite(uv_intensity));

    // Very long wavelength (IR)
    double ir_intensity = blackbody_radiation(10000.0, typical_temperature);
    EXPECT_GT(ir_intensity, 0.0);
    EXPECT_TRUE(std::isfinite(ir_intensity));
}

/**
 * @brief Test Voigt profile with typical parameters.
 */
TEST_F(UtilityFunctionsTest, VoigtProfileTypical) {
    double x = 0.0;  // Center
    double voigt_value = voigt_profile_approx(x, sigma, gamma);

    EXPECT_GT(voigt_value, 0.0);
    EXPECT_TRUE(std::isfinite(voigt_value));
}

/**
 * @brief Test Voigt profile symmetry around center.
 */
TEST_F(UtilityFunctionsTest, VoigtProfileSymmetry) {
    double offset = 0.1;

    double voigt_plus = voigt_profile_approx(offset, sigma, gamma);
    double voigt_minus = voigt_profile_approx(-offset, sigma, gamma);

    EXPECT_NEAR(voigt_plus, voigt_minus, 1e-10);
}

/**
 * @brief Test Voigt profile maximum at center.
 */
TEST_F(UtilityFunctionsTest, VoigtProfileMaximumAtCenter) {
    double center_value = voigt_profile_approx(0.0, sigma, gamma);

    // Test several points away from center
    std::vector<double> offsets = {0.1, 0.2, 0.5, 1.0};

    for (double offset : offsets) {
        double off_center_value = voigt_profile_approx(offset, sigma, gamma);
        EXPECT_LE(off_center_value, center_value);
    }
}

/**
 * @brief Test Voigt profile approaches zero at large distances.
 */
TEST_F(UtilityFunctionsTest, VoigtProfileTailBehavior) {
    double center_value = voigt_profile_approx(0.0, sigma, gamma);
    double far_value = voigt_profile_approx(10.0, sigma, gamma);

    EXPECT_GT(center_value, far_value);
    EXPECT_GT(far_value, 0.0);  // Should still be positive
}

/**
 * @brief Test Voigt profile with very narrow Gaussian (sigma → 0).
 */
TEST_F(UtilityFunctionsTest, VoigtProfileNarrowGaussian) {
    double narrow_sigma = 1e-6;
    double voigt_value = voigt_profile_approx(0.0, narrow_sigma, gamma);

    EXPECT_GT(voigt_value, 0.0);
    EXPECT_TRUE(std::isfinite(voigt_value));
}

/**
 * @brief Test Voigt profile with very narrow Lorentzian (gamma → 0).
 */
TEST_F(UtilityFunctionsTest, VoigtProfileNarrowLorentzian) {
    double narrow_gamma = 1e-6;
    double voigt_value = voigt_profile_approx(0.0, sigma, narrow_gamma);

    EXPECT_GT(voigt_value, 0.0);
    EXPECT_TRUE(std::isfinite(voigt_value));
}

/**
 * @brief Test Voigt profile width scaling.
 * Broader profiles should have lower peak values (conservation of area).
 */
TEST_F(UtilityFunctionsTest, VoigtProfileWidthScaling) {
    double narrow_sigma = 0.05;
    double broad_sigma = 0.2;

    double narrow_peak = voigt_profile_approx(0.0, narrow_sigma, gamma);
    double broad_peak = voigt_profile_approx(0.0, broad_sigma, gamma);

    // Narrower profile should have higher peak (approximately)
    EXPECT_GT(narrow_peak, broad_peak);
}

/**
 * @brief Test temperature consistency across different wavelengths for blackbody.
 */
TEST_F(UtilityFunctionsTest, BlackbodyTemperatureConsistency) {
    std::vector<double> wavelengths = {400.0, 500.0, 600.0, 700.0};
    std::vector<double> temperatures = {3000.0, 4000.0, 5000.0, 6000.0};

    for (double temp : temperatures) {
        std::vector<double> intensities;
        for (double wl : wavelengths) {
            double intensity = blackbody_radiation(wl, temp);
            EXPECT_GT(intensity, 0.0);
            EXPECT_TRUE(std::isfinite(intensity));
            intensities.push_back(intensity);
        }

        // All intensities should be positive and finite
        EXPECT_TRUE(std::all_of(intensities.begin(), intensities.end(),
                               [](double x) { return x > 0.0 && std::isfinite(x); }));
    }
}

/**
 * @brief Test numerical stability of utility functions.
 */
TEST_F(UtilityFunctionsTest, NumericalStability) {
    // Test with values that might cause numerical issues
    std::vector<double> test_wavelengths = {1e-3, 1e3, 1e6};
    std::vector<double> test_temperatures = {0.1, 1e6};

    for (double wl : test_wavelengths) {
        for (double temp : test_temperatures) {
            double intensity = blackbody_radiation(wl, temp);
            EXPECT_TRUE(std::isfinite(intensity));
            EXPECT_GE(intensity, 0.0);
        }
    }

    // Test Voigt profile with extreme parameters
    std::vector<double> test_widths = {1e-6, 1e3};
    for (double width : test_widths) {
        double voigt_value = voigt_profile_approx(0.0, width, width);
        EXPECT_TRUE(std::isfinite(voigt_value));
        EXPECT_GE(voigt_value, 0.0);
    }
}

/**
 * @brief Test de Broglie wavelength calculation.
 * Example from Wikipedia for thermal electrons at room temperature.
 */
TEST_F(UtilityFunctionsTest, DebroglieWavelength) {
    constexpr double temperature = 298.0;             // K
    constexpr double expected_wavelength = 4.3179e-9; // meters

    double result = calculate_debroglie_wavelength(temperature);
    EXPECT_NEAR(result, expected_wavelength, 1e-5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
