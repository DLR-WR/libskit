/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "PlasmaModel/LTETwoTime.h"

namespace libskit {

LTETwoTime::LTETwoTime(std::map<std::string, double> _element_ratio,
                       double _early_temperature,
                       double _early_electron_density, double _early_length,
                       double _late_temperature,
                       double _late_electron_density, double _late_length,
                       double _weight_early)
    : LTEModel(_element_ratio, _early_temperature, _early_electron_density,
               _early_length),
      early_temperature(_early_temperature),
      early_electron_density(_early_electron_density),
      early_length(_early_length), late_temperature(_late_temperature),
      late_electron_density(_late_electron_density),
      late_length(_late_length), weight_early(_weight_early) {}

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
  } else if (parameter_name == "late_temperature") {
    return late_temperature;
  } else if (parameter_name == "late_electron_density") {
    return late_electron_density;
  } else if (parameter_name == "late_length") {
    return late_length;
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
  } else if (parameter_name == "late_temperature") {
    late_temperature = value;
  } else if (parameter_name == "late_electron_density") {
    late_electron_density = value;
  } else if (parameter_name == "late_length") {
    late_length = value;
  } else if (parameter_name == "weight_early") {
    weight_early = value;
  } else {
    LTEModel::set_parameter(parameter_name, value);
  }
}

std::vector<std::string> LTETwoTime::get_parameter_names() const {
  auto base_names = LTEModel::get_parameter_names();
  base_names.insert(base_names.end(),
                    {"early_temperature", "early_electron_density",
                     "early_length", "late_temperature",
                     "late_electron_density", "late_length", "weight_early"});
  return base_names;
}

Eigen::VectorXd
LTETwoTime::calculate_irradiance(const Eigen::VectorXd &wavelengths) const {
  // Evaluate the early and late LTE states separately and combine them with a
  // convex weight.
  LTEModel early_model(this->element_ratio, early_temperature,
                       early_electron_density, early_length);
  LTEModel late_model(this->element_ratio, late_temperature,
                      late_electron_density, late_length);

  Eigen::VectorXd irradiance_early =
      early_model.calculate_irradiance(wavelengths);
  Eigen::VectorXd irradiance_late =
      late_model.calculate_irradiance(wavelengths);

  return weight_early * irradiance_early +
         (1.0 - weight_early) * irradiance_late;
}

} // namespace libskit
