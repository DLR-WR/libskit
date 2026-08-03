/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "../src/_libskit/libskit.h"
#include "gtest/gtest.h"

namespace libskit {

class LTELayersTest : public ::testing::Test {
protected:
    void SetUp() override {
        load_databases();
    }
};

// Test that LTELayers with single layer produces identical results to LTEModel
TEST_F(LTELayersTest, SingleLayerEquivalence) {

    // Test parameters
    std::map<std::string, double> element_ratio = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
    double temperature = 6000.0;
    double electron_density = 1e17;
    double length = 0.001;

    // Create single LTEModel
    LTEModel single_model(element_ratio, temperature, electron_density, length);

    // Create LTELayers with single layer using the same parameters
    std::vector<std::map<std::string, double>> layer_ratios = {element_ratio};
    std::vector<double> layer_temps = {temperature};
    std::vector<double> layer_densities = {electron_density};
    std::vector<double> layer_lengths = {length};

    LTELayers layered_model(layer_ratios, layer_temps, layer_densities, layer_lengths);

    // Test parameter access equivalence
    EXPECT_NEAR(layered_model.get_parameter("layer_0_temperature"),
                single_model.get_parameter("temperature"), 1e-10);
    EXPECT_NEAR(layered_model.get_parameter("layer_0_electron_density"),
                single_model.get_parameter("electron_density"), 1e-10);
    EXPECT_NEAR(layered_model.get_parameter("layer_0_length"),
                single_model.get_parameter("length"), 1e-10);
    EXPECT_NEAR(layered_model.get_parameter("layer_0_Fe"),
                single_model.get_parameter("Fe"), 1e-10);
    EXPECT_NEAR(layered_model.get_parameter("layer_0_Si"),
                single_model.get_parameter("Si"), 1e-10);
    EXPECT_NEAR(layered_model.get_parameter("layer_0_Ca"),
                single_model.get_parameter("Ca"), 1e-10);
}

TEST_F(LTELayersTest, ParameterModificationEquivalence) {

    // Test parameters
    std::map<std::string, double> element_ratio = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
    double temperature = 6000.0;
    double electron_density = 1e17;
    double length = 0.001;

    LTEModel single_model(element_ratio, temperature, electron_density, length);

    std::vector<std::map<std::string, double>> layer_ratios = {element_ratio};
    std::vector<double> layer_temps = {temperature};
    std::vector<double> layer_densities = {electron_density};
    std::vector<double> layer_lengths = {length};

    LTELayers layered_model(layer_ratios, layer_temps, layer_densities, layer_lengths);

    // Test parameter modifications
    double new_temp = 7000.0;
    single_model.set_parameter("temperature", new_temp);
    layered_model.set_parameter("layer_0_temperature", new_temp);

    EXPECT_NEAR(layered_model.get_parameter("layer_0_temperature"),
                single_model.get_parameter("temperature"), 1e-10);
}

TEST_F(LTELayersTest, IrradianceCalculationEquivalence) {

    // Test parameters
    std::map<std::string, double> element_ratio = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
    double temperature = 6000.0;
    double electron_density = 1e17;
    double length = 0.001;

    LTEModel single_model(element_ratio, temperature, electron_density, length);

    std::vector<std::map<std::string, double>> layer_ratios = {element_ratio};
    std::vector<double> layer_temps = {temperature};
    std::vector<double> layer_densities = {electron_density};
    std::vector<double> layer_lengths = {length};

    LTELayers layered_model(layer_ratios, layer_temps, layer_densities, layer_lengths);

    // Test wavelength range
    Eigen::VectorXd wavelengths = Eigen::VectorXd::LinSpaced(100, 400.0, 500.0);

    // Calculate irradiance from both models
    auto single_irradiance = single_model.calculate_irradiance(wavelengths);
    auto layered_irradiance = layered_model.calculate_irradiance(wavelengths);

    EXPECT_EQ(single_irradiance.size(), layered_irradiance.size());

    // Check that all wavelength points match within numerical precision
    for (int i = 0; i < wavelengths.size(); ++i) {
        EXPECT_NEAR(layered_irradiance[i], single_irradiance[i],
                   std::abs(single_irradiance[i]) * 1e-12 + 1e-15);
    }
}

TEST_F(LTELayersTest, StatisticalMeasuresEquivalence) {

    // Test parameters
    std::map<std::string, double> element_ratio = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
    double temperature = 6000.0;
    double electron_density = 1e17;
    double length = 0.001;

    LTEModel single_model(element_ratio, temperature, electron_density, length);

    std::vector<std::map<std::string, double>> layer_ratios = {element_ratio};
    std::vector<double> layer_temps = {temperature};
    std::vector<double> layer_densities = {electron_density};
    std::vector<double> layer_lengths = {length};

    LTELayers layered_model(layer_ratios, layer_temps, layer_densities, layer_lengths);

    // Test with larger wavelength range for better statistics
    Eigen::VectorXd wavelengths = Eigen::VectorXd::LinSpaced(500, 300.0, 800.0);

    auto single_irradiance = single_model.calculate_irradiance(wavelengths);
    auto layered_irradiance = layered_model.calculate_irradiance(wavelengths);

    // Check total integrated intensity
    double single_total = single_irradiance.sum();
    double layered_total = layered_irradiance.sum();
    EXPECT_NEAR(layered_total, single_total, std::abs(single_total) * 1e-12);

    // Check maximum intensity
    double single_max = single_irradiance.maxCoeff();
    double layered_max = layered_irradiance.maxCoeff();
    EXPECT_NEAR(layered_max, single_max, std::abs(single_max) * 1e-12);

    // Check RMS difference (should be near zero)
    auto diff = layered_irradiance - single_irradiance;
    double rms_diff = std::sqrt(diff.array().square().mean());
    double rms_signal = std::sqrt(single_irradiance.array().square().mean());
    EXPECT_LT(rms_diff / rms_signal, 1e-12);
}

TEST_F(LTELayersTest, CloneEquivalence) {

    // Test parameters
    std::map<std::string, double> element_ratio = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
    double temperature = 6000.0;
    double electron_density = 1e17;
    double length = 0.001;

    LTEModel single_model(element_ratio, temperature, electron_density, length);

    std::vector<std::map<std::string, double>> layer_ratios = {element_ratio};
    std::vector<double> layer_temps = {temperature};
    std::vector<double> layer_densities = {electron_density};
    std::vector<double> layer_lengths = {length};

    LTELayers layered_model(layer_ratios, layer_temps, layer_densities, layer_lengths);

    // Test that cloned models also produce identical results
    auto cloned_single = single_model.clone();
    auto cloned_layered = layered_model.clone();

    Eigen::VectorXd wavelengths = Eigen::VectorXd::LinSpaced(100, 400.0, 500.0);

    auto original_single = single_model.calculate_irradiance(wavelengths);
    auto cloned_single_result = cloned_single->calculate_irradiance(wavelengths);
    auto cloned_layered_result = cloned_layered->calculate_irradiance(wavelengths);

    for (int i = 0; i < wavelengths.size(); ++i) {
        EXPECT_NEAR(cloned_single_result[i], original_single[i],
                   std::abs(original_single[i]) * 1e-12 + 1e-15);
        EXPECT_NEAR(cloned_layered_result[i], original_single[i],
                   std::abs(original_single[i]) * 1e-12 + 1e-15);
    }
}

} // namespace libskit

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
