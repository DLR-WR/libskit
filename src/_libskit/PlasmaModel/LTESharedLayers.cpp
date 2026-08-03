/*
 * Copyright (c) 2026 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "LTESharedLayers.h"
#include "physics_kernel.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace libskit {

namespace {

// Parse "layer_<i>_<rest>" -> (i, rest). Returns false if the prefix does
// not match the expected scheme.
bool parse_layer_prefix(const std::string &name, size_t &index,
                        std::string &rest) {
  static const std::string kPrefix = "layer_";
  if (name.compare(0, kPrefix.size(), kPrefix) != 0) {
    return false;
  }
  size_t sep = name.find('_', kPrefix.size());
  if (sep == std::string::npos) {
    return false;
  }
  try {
    index = std::stoul(name.substr(kPrefix.size(), sep - kPrefix.size()));
  } catch (const std::exception &) {
    return false;
  }
  rest = name.substr(sep + 1);
  return true;
}

} // namespace

LTESharedLayers::LTESharedLayers(
    std::map<std::string, double> _element_ratio,
    std::vector<double> _temperatures,
    std::vector<double> _electron_densities, std::vector<double> _lengths)
    : element_ratio(std::move(_element_ratio)),
      temperatures(std::move(_temperatures)),
      electron_densities(std::move(_electron_densities)),
      lengths(std::move(_lengths)) {

  if (temperatures.size() != electron_densities.size() ||
      electron_densities.size() != lengths.size()) {
    throw std::invalid_argument(
        "LTESharedLayers: temperatures, electron_densities and lengths "
        "must have the same length");
  }
  if (temperatures.empty()) {
    throw std::invalid_argument("LTESharedLayers: at least one layer required");
  }
  if (element_ratio.empty()) {
    throw std::invalid_argument(
        "LTESharedLayers: element_ratio must contain at least one element");
  }
}

std::shared_ptr<PlasmaModel> LTESharedLayers::clone() const {
  return std::make_shared<LTESharedLayers>(*this);
}

double &LTESharedLayers::get_parameter(const std::string &parameter_name) {
  size_t idx;
  std::string rest;
  if (parse_layer_prefix(parameter_name, idx, rest)) {
    if (idx >= temperatures.size()) {
      throw std::out_of_range("LTESharedLayers::get_parameter: layer index "
                              "out of range");
    }
    if (rest == "temperature")
      return temperatures[idx];
    if (rest == "electron_density")
      return electron_densities[idx];
    if (rest == "length")
      return lengths[idx];
    throw std::invalid_argument(
        "LTESharedLayers::get_parameter: unknown per-layer parameter '" +
        rest + "'");
  }
  auto it = element_ratio.find(parameter_name);
  if (it == element_ratio.end()) {
    throw std::invalid_argument(
        "LTESharedLayers::get_parameter: unknown parameter '" +
        parameter_name + "'");
  }
  return it->second;
}

void LTESharedLayers::set_parameter(const std::string &parameter_name,
                                    double value) {
  size_t idx;
  std::string rest;
  if (parse_layer_prefix(parameter_name, idx, rest)) {
    if (idx >= temperatures.size()) {
      throw std::out_of_range("LTESharedLayers::set_parameter: layer index "
                              "out of range");
    }
    if (rest == "temperature") {
      temperatures[idx] = value;
      return;
    }
    if (rest == "electron_density") {
      electron_densities[idx] = value;
      return;
    }
    if (rest == "length") {
      lengths[idx] = value;
      return;
    }
    throw std::invalid_argument(
        "LTESharedLayers::set_parameter: unknown per-layer parameter '" +
        rest + "'");
  }
  auto it = element_ratio.find(parameter_name);
  if (it == element_ratio.end()) {
    throw std::invalid_argument(
        "LTESharedLayers::set_parameter: unknown parameter '" +
        parameter_name + "'");
  }
  it->second = value;
}

std::vector<std::string> LTESharedLayers::get_parameter_names() const {
  std::vector<std::string> names;
  names.reserve(3 * temperatures.size() + element_ratio.size());
  for (size_t i = 0; i < temperatures.size(); ++i) {
    const std::string prefix = "layer_" + std::to_string(i) + "_";
    names.push_back(prefix + "temperature");
    names.push_back(prefix + "electron_density");
    names.push_back(prefix + "length");
  }
  for (const auto &[el, _] : element_ratio) {
    names.push_back(el);
  }
  return names;
}

Eigen::VectorXd LTESharedLayers::calculate_irradiance(
    const Eigen::VectorXd &wavelengths) const {

  const long n = wavelengths.size();
  std::vector<double> irr_vec(n, 0.0);

  for (size_t li = 0; li < temperatures.size(); ++li) {
    // Build a temporary single-layer LTEModel that reuses the shared
    // composition with this layer's plasma state. The element map copy is
    // negligible for the typical small number of species.
    LTEModel layer(element_ratio, temperatures[li], electron_densities[li],
                   lengths[li]);
    const Eigen::VectorXd ec = layer.calculate_emission_coefficient(wavelengths);
    std::vector<double> ec_vec(ec.data(), ec.data() + n);
    physics::apply_rte_multilayer(wavelengths.data(), n, ec_vec,
                                  layer.temperature, layer.length, irr_vec);
  }

  return Eigen::Map<Eigen::VectorXd>(irr_vec.data(), n);
}

double LTESharedLayers::get_minimum_voigt_fwhm(
    const Eigen::VectorXd &wavelengths) const {
  double min_fwhm = std::numeric_limits<double>::max();
  for (size_t li = 0; li < temperatures.size(); ++li) {
    LTEModel layer(element_ratio, temperatures[li], electron_densities[li],
                   lengths[li]);
    double layer_fwhm = layer.get_minimum_voigt_fwhm(wavelengths);
    if (layer_fwhm < min_fwhm) {
      min_fwhm = layer_fwhm;
    }
  }
  return min_fwhm;
}

} // namespace libskit
