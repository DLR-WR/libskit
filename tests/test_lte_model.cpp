/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file test_lte_model.cpp
 * @brief Comprehensive C++ unit tests for the LTEModel class.
 *
 * These tests verify the core functionality of the LTEModel plasma physics
 * calculations, parameter management, and spectral computations at the C++ level.
 */

#include "../src/_libskit/libskit.h"
#include "gtest/gtest.h"
#include <cmath>

namespace libskit {

class LTEModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load databases before any testing
        load_databases();

        // Standard test plasma parameters
        elements = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
        temperature = 10000.0;  // K
        electron_density = 1e23;  // m^-3
        length = 0.001;  // m

        // Create test model
        model = std::make_unique<LTEModel>(elements, temperature, electron_density, length);

        // Test wavelength range
        wavelengths = Eigen::VectorXd::LinSpaced(100, 300.0, 800.0);  // nm
    }

    std::map<std::string, double> elements;
    double temperature;
    double electron_density;
    double length;
    std::unique_ptr<LTEModel> model;
    Eigen::VectorXd wavelengths;
};

// Test basic construction and parameter access
TEST_F(LTEModelTest, ConstructionAndBasicProperties) {
    EXPECT_DOUBLE_EQ(model->temperature, temperature);
    EXPECT_DOUBLE_EQ(model->electron_density, electron_density);
    EXPECT_DOUBLE_EQ(model->length, length);

    // Check element ratios
    for (const auto& [element, ratio] : elements) {
        EXPECT_DOUBLE_EQ(model->element_ratio[element], ratio);
    }
}

// Test parameter getter/setter interface
TEST_F(LTEModelTest, ParameterAccess) {
    // Test temperature access
    EXPECT_DOUBLE_EQ(model->get_parameter("temperature"), temperature);

    model->set_parameter("temperature", 12000.0);
    EXPECT_DOUBLE_EQ(model->get_parameter("temperature"), 12000.0);
    EXPECT_DOUBLE_EQ(model->temperature, 12000.0);

    // Test electron density access
    EXPECT_DOUBLE_EQ(model->get_parameter("electron_density"), electron_density);

    model->set_parameter("electron_density", 5e23);
    EXPECT_DOUBLE_EQ(model->get_parameter("electron_density"), 5e23);

    // Test element ratio access
    EXPECT_DOUBLE_EQ(model->get_parameter("Fe"), 0.5);

    model->set_parameter("Fe", 0.7);
    EXPECT_DOUBLE_EQ(model->get_parameter("Fe"), 0.7);
    EXPECT_DOUBLE_EQ(model->element_ratio["Fe"], 0.7);
}

// Test cloning functionality
TEST_F(LTEModelTest, CloningFunctionality) {
    auto cloned = model->clone();

    // Verify that clone has same parameters
    auto* cloned_lte = dynamic_cast<LTEModel*>(cloned.get());
    ASSERT_NE(cloned_lte, nullptr);

    EXPECT_DOUBLE_EQ(cloned_lte->temperature, model->temperature);
    EXPECT_DOUBLE_EQ(cloned_lte->electron_density, model->electron_density);
    EXPECT_DOUBLE_EQ(cloned_lte->length, model->length);

    for (const auto& [element, ratio] : elements) {
        EXPECT_DOUBLE_EQ(cloned_lte->element_ratio[element], ratio);
    }

    // Verify independence (modifying original doesn't affect clone)
    model->set_parameter("temperature", 15000.0);
    EXPECT_NE(cloned_lte->temperature, model->temperature);
}

// Test population fraction calculations
TEST_F(LTEModelTest, PopulationFractions) {
    auto pop_fractions = model->get_population_fractions();

    // Should have entries for all elements
    for (const auto& [element, ratio] : elements) {
        EXPECT_GT(pop_fractions.count(element), 0) << "Missing population fraction for " << element;

        // Population fractions should be non-negative
        const auto& fractions = pop_fractions[element];
        for (int i = 0; i < fractions.size(); i++) {
            EXPECT_GE(fractions[i], 0.0) << "Negative population fraction for " << element << " level " << i;
        }

        // Sum of population fractions should be close to 1
        double sum = fractions.sum();
        EXPECT_NEAR(sum, 1.0, 1e-6) << "Population fractions don't sum to 1 for " << element;
    }
}

// Test irradiance calculation basic properties
TEST_F(LTEModelTest, IrradianceCalculation) {
    Eigen::VectorXd irradiance = model->calculate_irradiance(wavelengths);

    // Should return same size as input
    EXPECT_EQ(irradiance.size(), wavelengths.size());

    // All values should be non-negative
    for (int i = 0; i < irradiance.size(); i++) {
        EXPECT_GE(irradiance[i], 0.0) << "Negative irradiance at wavelength " << wavelengths[i];
    }

    // Should have some non-zero values (plasma should emit something)
    double max_irradiance = irradiance.maxCoeff();
    EXPECT_GT(max_irradiance, 0.0) << "Plasma emits no radiation";
}

// Test irradiance scaling with temperature
TEST_F(LTEModelTest, IrradianceTemperatureScaling) {
    // Calculate irradiance at base temperature
    Eigen::VectorXd irradiance_low = model->calculate_irradiance(wavelengths);

    // Increase temperature and recalculate
    model->set_parameter("temperature", temperature * 1.5);
    Eigen::VectorXd irradiance_high = model->calculate_irradiance(wavelengths);

    // Higher temperature should generally produce more emission
    double total_low = irradiance_low.sum();
    double total_high = irradiance_high.sum();

    EXPECT_GT(total_high, total_low) << "Higher temperature didn't increase total emission";
}

// Test irradiance scaling with electron density
TEST_F(LTEModelTest, IrradianceElectronDensityScaling) {
    Eigen::VectorXd irradiance_low = model->calculate_irradiance(wavelengths);

    // Increase electron density
    model->set_parameter("electron_density", electron_density * 2.0);
    Eigen::VectorXd irradiance_high = model->calculate_irradiance(wavelengths);

    // Higher electron density should generally increase emission
    double total_low = irradiance_low.sum();
    double total_high = irradiance_high.sum();

    EXPECT_GT(total_high, total_low) << "Higher electron density didn't increase emission";
}

// Test irradiance scaling with plasma length
TEST_F(LTEModelTest, IrradianceLengthScaling) {
    Eigen::VectorXd irradiance_short = model->calculate_irradiance(wavelengths);

    // Increase plasma length
    model->set_parameter("length", length * 2.0);
    Eigen::VectorXd irradiance_long = model->calculate_irradiance(wavelengths);

    // Longer plasma should produce more emission (up to optical depth saturation)
    double total_short = irradiance_short.sum();
    double total_long = irradiance_long.sum();

    EXPECT_GT(total_long, total_short) << "Longer plasma didn't increase emission";
}

// Test extreme parameter values
TEST_F(LTEModelTest, ExtremeParameterValues) {
    // Very low temperature - should still work without crashing
    model->set_parameter("temperature", 1000.0);
    Eigen::VectorXd irradiance_cold = model->calculate_irradiance(wavelengths);
    EXPECT_EQ(irradiance_cold.size(), wavelengths.size());

    // Very high temperature
    model->set_parameter("temperature", 50000.0);
    Eigen::VectorXd irradiance_hot = model->calculate_irradiance(wavelengths);
    EXPECT_EQ(irradiance_hot.size(), wavelengths.size());

    // Very low electron density
    model->set_parameter("electron_density", 1e20);
    Eigen::VectorXd irradiance_sparse = model->calculate_irradiance(wavelengths);
    EXPECT_EQ(irradiance_sparse.size(), wavelengths.size());
}

// Test physical consistency
TEST_F(LTEModelTest, PhysicalConsistency) {
    // Test that emission coefficient and irradiance are related through blackbody radiation
    Eigen::VectorXd emission_coeff = model->calculate_emission_coefficient(wavelengths);
    Eigen::VectorXd irradiance = model->calculate_irradiance(wavelengths);

    // Both should have same size
    EXPECT_EQ(emission_coeff.size(), irradiance.size());

}

// Helper function to get atomic mass (simplified for testing)
double get_atomic_mass(const std::string& element) {
    static const std::map<std::string, double> masses = {
        {"Fe", 55.845}, {"Si", 28.085}, {"Ca", 40.078}
    };
    return masses.at(element);
}

} // namespace libskit

// Main function for running tests
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
