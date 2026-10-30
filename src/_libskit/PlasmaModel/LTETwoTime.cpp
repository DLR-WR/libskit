/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "PlasmaModel/LTETwoTime.h"

namespace libskit {

LTETwoTime::LTETwoTime(std::map<std::string, double> _element_ratio,
                       double _early_temperature,
                       double _early_electron_density, double _early_length,
                       double _delta_temperature,
                       double _delta_electron_density, double _delta_length,
                       double _weight_early)
    : LTEModel(_element_ratio, _early_temperature, _early_electron_density,
               _early_length),
      early_temperature(_early_temperature),
      early_electron_density(_early_electron_density),
      early_length(_early_length), delta_temperature(_delta_temperature),
      delta_electron_density(_delta_electron_density),
      delta_length(_delta_length), weight_early(_weight_early) {}

LTETwoTime LTETwoTime::from_late_state(
    std::map<std::string, double> element_ratio, double early_temperature,
    double early_electron_density, double early_length,
    double late_temperature, double late_electron_density,
    double late_length, double weight_early) {
  return LTETwoTime(std::move(element_ratio), early_temperature,
                     early_electron_density, early_length,
                     early_temperature - late_temperature,
                     early_electron_density - late_electron_density,
                     late_length - early_length, weight_early);
}

std::shared_ptr<PlasmaModel> LTETwoTime::clone() const {
  return std::make_shared<LTETwoTime>(*this);
}

double &LTETwoTime::get_parameter(const std::string &parameter_name) {
  if (parameter_name == "early_temperature") {
    return early_temperature;
  } else if (parameter_name == "early_electron_density") {
    return early_electron_density;
  } else if (parameter_name == "early_length") {
    return early_length;
  } else if (parameter_name == "delta_temperature") {
    return delta_temperature;
  } else if (parameter_name == "delta_electron_density") {
    return delta_electron_density;
  } else if (parameter_name == "delta_length") {
    return delta_length;
  } else if (parameter_name == "weight_early") {
    return weight_early;
  } else {
    return LTEModel::get_parameter(parameter_name);
  }
}

void LTETwoTime::set_parameter(const std::string &parameter_name,
                              double value) {
  if (parameter_name == "early_temperature") {
    early_temperature = value;
    this->temperature = value; // keep base in sync with early state
  } else if (parameter_name == "early_electron_density") {
    early_electron_density = value;
    this->electron_density = value;
  } else if (parameter_name == "early_length") {
    early_length = value;
    this->length = value;
  } else if (parameter_name == "delta_temperature") {
    delta_temperature = value;
  } else if (parameter_name == "delta_electron_density") {
    delta_electron_density = value;
  } else if (parameter_name == "delta_length") {
    delta_length = value;
  } else if (parameter_name == "weight_early") {
    weight_early = value;
  } else {
    LTEModel::set_parameter(parameter_name, value);
  }
}

std::vector<std::string> LTETwoTime::get_parameter_names() const {
  // LTEModel::get_parameter_names() returns {temperature, electron_density,
  // length, <elements...>}; the first three are not fittable here (the
  // fittable plasma parameters are early_*/delta_*), so only the trailing
  // element names are kept.
  auto base_names = LTEModel::get_parameter_names();
  std::vector<std::string> element_names(base_names.begin() + 3,
                                          base_names.end());
  std::vector<std::string> parameter_names = {
      "early_temperature", "early_electron_density", "early_length",
      "delta_temperature", "delta_electron_density", "delta_length",
      "weight_early"};
  parameter_names.insert(parameter_names.end(), element_names.begin(),
                          element_names.end());
  return parameter_names;
}

Eigen::VectorXd
LTETwoTime::calculate_irradiance(const Eigen::VectorXd &wavelengths) const {
  // Evaluate the early and late LTE states separately and combine them with a
  // convex weight. The late state is derived from early_* and delta_*.
  LTEModel early_model(this->element_ratio, early_temperature,
                       early_electron_density, early_length);
  LTEModel late_model(this->element_ratio, late_temperature(),
                      late_electron_density(), late_length());

  Eigen::VectorXd irradiance_early =
      early_model.calculate_irradiance(wavelengths);
  Eigen::VectorXd irradiance_late =
      late_model.calculate_irradiance(wavelengths);

  return weight_early * irradiance_early +
         (1.0 - weight_early) * irradiance_late;
}

} // namespace libskit
