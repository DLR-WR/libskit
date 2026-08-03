/*
 * Copyright (c) 2026 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "../src/_libskit/libskit.h"
#include "gtest/gtest.h"

namespace libskit {

class LTESharedLayersTest : public ::testing::Test {
protected:
  void SetUp() override {
    set_database_path("databases");
    load_databases();
  }
};

// LTESharedLayers with a single layer must reproduce LTEModel.
TEST_F(LTESharedLayersTest, SingleLayerEquivalence) {
  std::map<std::string, double> er = {{"Fe", 0.5}, {"Si", 0.3}, {"Ca", 0.2}};
  const double T = 6000.0, ne = 1e17, L = 1e-3;

  LTEModel single(er, T, ne, L);
  LTESharedLayers shared(er, {T}, {ne}, {L});

  Eigen::VectorXd wl = Eigen::VectorXd::LinSpaced(200, 280.0, 320.0);
  auto a = single.calculate_irradiance(wl);
  auto b = shared.calculate_irradiance(wl);
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], 1e-6 * (std::abs(a[i]) + 1e-30));
}

// N-layer LTESharedLayers must reproduce LTELayers with the same composition
// duplicated to every layer.
TEST_F(LTESharedLayersTest, MatchesLTELayersWithDuplicatedComposition) {
  std::map<std::string, double> er = {{"Fe", 0.7}, {"Si", 0.3}};
  std::vector<double> temps = {10000.0, 8000.0, 6000.0};
  std::vector<double> dens = {1e23, 5e22, 1e22};
  std::vector<double> lens = {1e-3, 1e-3, 1e-3};

  LTESharedLayers shared(er, temps, dens, lens);
  std::vector<std::map<std::string, double>> per_layer(temps.size(), er);
  LTELayers layered(per_layer, temps, dens, lens);

  Eigen::VectorXd wl = Eigen::VectorXd::LinSpaced(200, 270.0, 290.0);
  auto a = shared.calculate_irradiance(wl);
  auto b = layered.calculate_irradiance(wl);
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], 1e-8 * (std::abs(b[i]) + 1e-30));
}

TEST_F(LTESharedLayersTest, ParameterNames) {
  std::map<std::string, double> er = {{"Fe", 0.7}, {"Si", 0.3}};
  LTESharedLayers m(er, {10000.0, 8000.0}, {1e23, 5e22}, {1e-3, 1e-3});
  auto names = m.get_parameter_names();
  // 2 layers × 3 per-layer + 2 elements = 8
  ASSERT_EQ(names.size(), 8u);
  EXPECT_EQ(names[0], "layer_0_temperature");
  EXPECT_EQ(names[1], "layer_0_electron_density");
  EXPECT_EQ(names[2], "layer_0_length");
  EXPECT_EQ(names[3], "layer_1_temperature");
  EXPECT_EQ(names[4], "layer_1_electron_density");
  EXPECT_EQ(names[5], "layer_1_length");
  // element order follows the map's iteration order (alphabetical for
  // std::map<std::string, ...>): Fe, Si
  EXPECT_EQ(names[6], "Fe");
  EXPECT_EQ(names[7], "Si");
}

TEST_F(LTESharedLayersTest, SetParameterUpdatesAllLayers) {
  std::map<std::string, double> er = {{"Fe", 0.7}, {"Si", 0.3}};
  LTESharedLayers m(er, {10000.0, 8000.0}, {1e23, 5e22}, {1e-3, 1e-3});

  m.set_parameter("Fe", 0.4);
  m.set_parameter("Si", 0.6);
  EXPECT_DOUBLE_EQ(m.element_ratio.at("Fe"), 0.4);
  EXPECT_DOUBLE_EQ(m.element_ratio.at("Si"), 0.6);

  m.set_parameter("layer_1_temperature", 9000.0);
  EXPECT_DOUBLE_EQ(m.temperatures[1], 9000.0);
  EXPECT_DOUBLE_EQ(m.temperatures[0], 10000.0);
}

TEST_F(LTESharedLayersTest, ConstructorValidation) {
  std::map<std::string, double> er = {{"Fe", 1.0}};
  EXPECT_THROW(LTESharedLayers(er, {10000.0}, {1e23, 5e22}, {1e-3}),
               std::invalid_argument);
  EXPECT_THROW(LTESharedLayers(er, {}, {}, {}), std::invalid_argument);
  EXPECT_THROW(
      LTESharedLayers({}, {10000.0}, {1e23}, {1e-3}),
      std::invalid_argument);
}

TEST_F(LTESharedLayersTest, UnknownParameterThrows) {
  std::map<std::string, double> er = {{"Fe", 1.0}};
  LTESharedLayers m(er, {10000.0}, {1e23}, {1e-3});
  EXPECT_THROW(m.set_parameter("nonsense", 1.0), std::invalid_argument);
  EXPECT_THROW(m.set_parameter("layer_5_temperature", 1.0), std::out_of_range);
  EXPECT_THROW(m.set_parameter("layer_0_bogus", 1.0), std::invalid_argument);
}

} // namespace libskit
