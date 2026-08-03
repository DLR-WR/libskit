/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "LTELayers.h"
#include "physics_kernel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace libskit {

LTELayers::LTELayers(std::vector<std::map<std::string, double>> _element_ratios,
                     std::vector<double> _temperatures,
                     std::vector<double> _electron_densities,
                     std::vector<double> _lengths) {

  // All per-layer vectors must describe the same number of layers.
  if (_element_ratios.size() != _temperatures.size() ||
      _temperatures.size() != _electron_densities.size() ||
      _electron_densities.size() != _lengths.size()) {
    throw std::invalid_argument("All input vectors must have the same size");
  }

  if (_element_ratios.empty()) {
    throw std::invalid_argument("Input vectors cannot be empty");
  }

  // Reserve storage and build one LTEModel per layer.
  lte_layers.reserve(_element_ratios.size());

  for (size_t i = 0; i < _element_ratios.size(); ++i) {
    lte_layers.emplace_back(_element_ratios[i], _temperatures[i],
                            _electron_densities[i], _lengths[i]);
  }
}

std::shared_ptr<PlasmaModel> LTELayers::clone() const {
  return std::make_shared<LTELayers>(*this);
}

double &LTELayers::get_parameter(const std::string &parameter_name) {

  // Expected format: layer_X_parameter or layer_X_element.
  if (parameter_name.find("layer_") != 0) {
    throw std::invalid_argument("Parameter name must start with 'layer_'");
  }

  size_t first_underscore = parameter_name.find('_');
  // The second underscore separates the layer index from the actual field name.
  size_t second_underscore = parameter_name.find('_', first_underscore + 1);

  if (second_underscore == std::string::npos) {
    throw std::invalid_argument(
        "Invalid parameter name format. Expected: layer_X_parameter");
  }

  std::string layer_str = parameter_name.substr(
      first_underscore + 1, second_underscore - first_underscore - 1);
  size_t layer_index;
  try {
    layer_index = std::stoul(layer_str);
  } catch (const std::exception &) {
    throw std::invalid_argument("Invalid layer index in parameter name");
  }

  if (layer_index >= lte_layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }

  // Delegate the remaining parameter lookup to the chosen LTE layer.
  std::string actual_parameter = parameter_name.substr(second_underscore + 1);

  return lte_layers[layer_index].get_parameter(actual_parameter);
}

void LTELayers::set_parameter(const std::string &parameter_name, double value) {

  // Expected format: layer_X_parameter or layer_X_element.
  if (parameter_name.find("layer_") != 0) {
    throw std::invalid_argument("Parameter name must start with 'layer_'");
  }

  size_t first_underscore = parameter_name.find('_');
  // The second underscore separates the layer index from the actual field name.
  size_t second_underscore = parameter_name.find('_', first_underscore + 1);

  if (second_underscore == std::string::npos) {
    throw std::invalid_argument(
        "Invalid parameter name format. Expected: layer_X_parameter");
  }

  std::string layer_str = parameter_name.substr(
      first_underscore + 1, second_underscore - first_underscore - 1);
  size_t layer_index;
  try {
    layer_index = std::stoul(layer_str);
  } catch (const std::exception &) {
    throw std::invalid_argument("Invalid layer index in parameter name");
  }

  if (layer_index >= lte_layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }

  // Delegate the remaining parameter update to the chosen LTE layer.
  std::string actual_parameter = parameter_name.substr(second_underscore + 1);

  lte_layers[layer_index].set_parameter(actual_parameter, value);
}

std::vector<std::string> LTELayers::get_parameter_names() const {
  std::vector<std::string> parameter_names;

  // Prefix each LTEModel parameter with its owning layer index.
  for (size_t i = 0; i < lte_layers.size(); ++i) {
    const auto &layer_param_names = lte_layers[i].get_parameter_names();

    for (const auto &param_name : layer_param_names) {
      parameter_names.push_back("layer_" + std::to_string(i) + "_" +
                                param_name);
    }
  }

  return parameter_names;
}

Eigen::VectorXd LTELayers::calculate_irradiance(
    const Eigen::VectorXd &wavelengths) const {

  const long n = wavelengths.size();
  // Start from zero incoming irradiance and apply the multilayer RTE in order.
  std::vector<double> irr_vec(n, 0.0);

  for (const auto &layer : lte_layers) {
    // Each layer contributes emission and attenuates what comes from below.
    const Eigen::VectorXd ec = layer.calculate_emission_coefficient(wavelengths);
    std::vector<double> ec_vec(ec.data(), ec.data() + n);
    physics::apply_rte_multilayer(wavelengths.data(), n, ec_vec,
                                  layer.temperature, layer.length, irr_vec);
  }

  return Eigen::Map<Eigen::VectorXd>(irr_vec.data(), n);
}

double LTELayers::get_minimum_voigt_fwhm(
    const Eigen::VectorXd &wavelengths) const {
  double min_fwhm = std::numeric_limits<double>::max();
  for (const auto &layer : lte_layers) {
    double layer_fwhm = layer.get_minimum_voigt_fwhm(wavelengths);
    if (layer_fwhm < min_fwhm) {
      min_fwhm = layer_fwhm;
    }
  }
  return min_fwhm;
}

} // namespace libskit
