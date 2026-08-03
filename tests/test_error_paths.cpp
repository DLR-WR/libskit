/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file test_error_paths.cpp
 * @brief Coverage for exception paths in PlasmaModel implementations.
 *
 * Exercises every `throw std::invalid_argument` and `throw std::out_of_range`
 * branch in LTEModel and LTELayers. These were previously uncovered.
 */

#include "../src/_libskit/libskit.h"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace libskit;

namespace {

LTELayers make_two_layer_model() {
  std::map<std::string, double> er = {{"Fe", 0.5}, {"Si", 0.5}};
  std::vector<std::map<std::string, double>> layer_ratios = {er, er};
  std::vector<double> temps = {10000.0, 8000.0};
  std::vector<double> densities = {1e23, 5e22};
  std::vector<double> lengths = {1e-3, 1e-3};
  return LTELayers(layer_ratios, temps, densities, lengths);
}

}  // namespace

// ── LTEModel::set_parameter ──────────────────────────────────────────

TEST(LTEModelErrors, SetParameterUnknownNameThrows) {
  std::map<std::string, double> er = {{"Fe", 1.0}};
  LTEModel model(er, 10000.0, 1e23, 1e-3);
  EXPECT_THROW(model.set_parameter("not_a_real_parameter", 1.0),
               std::invalid_argument);
}

TEST(LTEModelErrors, SetParameterValidNamesDoNotThrow) {
  std::map<std::string, double> er = {{"Fe", 1.0}};
  LTEModel model(er, 10000.0, 1e23, 1e-3);
  EXPECT_NO_THROW(model.set_parameter("temperature", 11000.0));
  EXPECT_NO_THROW(model.set_parameter("electron_density", 2e23));
  EXPECT_NO_THROW(model.set_parameter("length", 2e-3));
  EXPECT_NO_THROW(model.set_parameter("Fe", 0.5));
}

// ── LTELayers constructor validation ────────────────────────────────

TEST(LTELayersErrors, MismatchedVectorSizesThrows) {
  std::map<std::string, double> er = {{"Fe", 1.0}};
  std::vector<std::map<std::string, double>> ratios = {er, er};
  std::vector<double> temps = {10000.0, 8000.0};
  std::vector<double> densities = {1e23};  // wrong size
  std::vector<double> lengths = {1e-3, 1e-3};
  EXPECT_THROW(LTELayers(ratios, temps, densities, lengths),
               std::invalid_argument);
}

TEST(LTELayersErrors, EmptyVectorsThrow) {
  std::vector<std::map<std::string, double>> ratios;
  std::vector<double> temps;
  std::vector<double> densities;
  std::vector<double> lengths;
  EXPECT_THROW(LTELayers(ratios, temps, densities, lengths),
               std::invalid_argument);
}

// ── LTELayers::set_parameter parsing errors ─────────────────────────

TEST(LTELayersErrors, SetParameterMissingLayerPrefixThrows) {
  auto model = make_two_layer_model();
  EXPECT_THROW(model.set_parameter("temperature", 11000.0),
               std::invalid_argument);
}

TEST(LTELayersErrors, SetParameterMalformedNameThrows) {
  auto model = make_two_layer_model();
  // No second underscore separating index from parameter name.
  EXPECT_THROW(model.set_parameter("layer_0", 11000.0),
               std::invalid_argument);
}

TEST(LTELayersErrors, SetParameterNonNumericIndexThrows) {
  auto model = make_two_layer_model();
  EXPECT_THROW(model.set_parameter("layer_X_temperature", 11000.0),
               std::invalid_argument);
}

TEST(LTELayersErrors, SetParameterIndexOutOfRangeThrows) {
  auto model = make_two_layer_model();  // 2 layers => valid indices 0, 1
  EXPECT_THROW(model.set_parameter("layer_5_temperature", 11000.0),
               std::out_of_range);
}

TEST(LTELayersErrors, SetParameterValidLayerNameDoesNotThrow) {
  auto model = make_two_layer_model();
  EXPECT_NO_THROW(model.set_parameter("layer_0_temperature", 10500.0));
  EXPECT_NO_THROW(model.set_parameter("layer_1_electron_density", 6e22));
}

// ── LTELayers::get_parameter parsing errors (mirror of set) ─────────

TEST(LTELayersErrors, GetParameterMissingLayerPrefixThrows) {
  auto model = make_two_layer_model();
  EXPECT_THROW({ (void)model.get_parameter("temperature"); },
               std::invalid_argument);
}

TEST(LTELayersErrors, GetParameterMalformedNameThrows) {
  auto model = make_two_layer_model();
  EXPECT_THROW({ (void)model.get_parameter("layer_0"); },
               std::invalid_argument);
}

TEST(LTELayersErrors, GetParameterNonNumericIndexThrows) {
  auto model = make_two_layer_model();
  EXPECT_THROW({ (void)model.get_parameter("layer_X_temperature"); },
               std::invalid_argument);
}

TEST(LTELayersErrors, GetParameterIndexOutOfRangeThrows) {
  auto model = make_two_layer_model();
  EXPECT_THROW({ (void)model.get_parameter("layer_5_temperature"); },
               std::out_of_range);
}

TEST(LTELayersErrors, GetParameterValidLayerNameReturnsValue) {
  auto model = make_two_layer_model();
  double t = model.get_parameter("layer_0_temperature");
  EXPECT_DOUBLE_EQ(t, 10000.0);
  double t1 = model.get_parameter("layer_1_temperature");
  EXPECT_DOUBLE_EQ(t1, 8000.0);
}
