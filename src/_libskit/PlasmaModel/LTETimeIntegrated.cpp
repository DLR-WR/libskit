/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "PlasmaModel/LTETimeIntegrated.h"
#include <cmath>

namespace libskit {

LTETimeIntegrated::LTETimeIntegrated(
    std::map<std::string, double> _element_ratio, double _start_temperature,
    double _end_temperature, double _tau_temperature,
    double _start_electron_density, double _end_electron_density,
    double _tau_electron_density, double _start_length, double _end_length,
    double _tau_length, double _integration_time, int _integration_points)
    : LTEModel(_element_ratio, _start_temperature, _start_electron_density,
               _start_length),
      start_temperature(_start_temperature), end_temperature(_end_temperature),
      tau_temperature(_tau_temperature),
      start_electron_density(_start_electron_density),
      end_electron_density(_end_electron_density),
      tau_electron_density(_tau_electron_density), start_length(_start_length),
      end_length(_end_length), tau_length(_tau_length),
      integration_time(_integration_time),
      integration_points(_integration_points) {}

std::shared_ptr<PlasmaModel> LTETimeIntegrated::clone() const {
  return std::make_shared<LTETimeIntegrated>(*this);
}

double &LTETimeIntegrated::get_parameter(const std::string &parameter_name) {
  if (parameter_name == "start_temperature") {
    return start_temperature;
  } else if (parameter_name == "end_temperature") {
    return end_temperature;
  } else if (parameter_name == "tau_temperature") {
    return tau_temperature;
  } else if (parameter_name == "start_electron_density") {
    return start_electron_density;
  } else if (parameter_name == "end_electron_density") {
    return end_electron_density;
  } else if (parameter_name == "tau_electron_density") {
    return tau_electron_density;
  } else if (parameter_name == "start_length") {
    return start_length;
  } else if (parameter_name == "end_length") {
    return end_length;
  } else if (parameter_name == "tau_length") {
    return tau_length;
  } else if (parameter_name == "integration_time") {
    return integration_time;
  } else {
    return LTEModel::get_parameter(parameter_name);
  }
}

void LTETimeIntegrated::set_parameter(const std::string &parameter_name,
                                      double value) {
  if (parameter_name == "start_temperature") {
    start_temperature = value;
    this->temperature = value;
  } else if (parameter_name == "end_temperature") {
    end_temperature = value;
  } else if (parameter_name == "tau_temperature") {
    tau_temperature = value;
  } else if (parameter_name == "start_electron_density") {
    start_electron_density = value;
    this->electron_density = value;
  } else if (parameter_name == "end_electron_density") {
    end_electron_density = value;
  } else if (parameter_name == "tau_electron_density") {
    tau_electron_density = value;
  } else if (parameter_name == "start_length") {
    start_length = value;
    this->length = value;
  } else if (parameter_name == "end_length") {
    end_length = value;
  } else if (parameter_name == "tau_length") {
    tau_length = value;
  } else if (parameter_name == "integration_time") {
    integration_time = value;
  } else {
    LTEModel::set_parameter(parameter_name, value);
  }
}

std::vector<std::string> LTETimeIntegrated::get_parameter_names() const {
  auto base_names = LTEModel::get_parameter_names();
  base_names.insert(base_names.end(),
                    {"start_temperature", "end_temperature", "tau_temperature",
                     "start_electron_density", "end_electron_density",
                     "tau_electron_density", "start_length", "end_length",
                     "tau_length", "integration_time"});
  return base_names;
}

Eigen::VectorXd LTETimeIntegrated::calculate_irradiance(
    const Eigen::VectorXd &wavelengths) const {

  Eigen::VectorXd integrated_irradiance =
      Eigen::VectorXd::Zero(wavelengths.size());

  // Use uniform time steps for the time-weighted sum.
  double dt = integration_time / static_cast<double>(integration_points);

  for (int i = 0; i < integration_points; ++i) {
    // Evaluate at the midpoint of each time interval.
    double t = (i + 0.5) * dt;

    // T(t)  = T_f  + (T_0  - T_f)  * exp(-t / tau_T)
    double temp_t = end_temperature + (start_temperature - end_temperature) *
                                          std::exp(-t / tau_temperature);
    // ne(t) = ne_f + (ne_0 - ne_f) * exp(-t / tau_ne)
    double density_t =
        end_electron_density + (start_electron_density - end_electron_density) *
                                   std::exp(-t / tau_electron_density);
    // L(t)  = L_f  + (L_0  - L_f)  * exp(-t / tau_L)
    double length_t =
        end_length + (start_length - end_length) * std::exp(-t / tau_length);

    // Build a temporary static LTE model for this time slice.
    LTEModel temp_model(this->element_ratio, temp_t, density_t, length_t);
    Eigen::VectorXd irradiance_t = temp_model.calculate_irradiance(wavelengths);

    // I_total = sum(I_i * dt / t_gate)
    integrated_irradiance += irradiance_t * (dt / integration_time);
  }

  return integrated_irradiance;
}

} // namespace libskit
