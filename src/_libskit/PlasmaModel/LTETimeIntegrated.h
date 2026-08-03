/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef LTETIMEINTEGRATED_H
#define LTETIMEINTEGRATED_H

#include <map>

#include "PlasmaModel.h"
#include "PlasmaModel/LTEModel.h"

namespace libskit {

/**
 * @brief LTE plasma model with exponentially evolving state integrated over
 * time.
 *
 * The plasma state decays from a start value to an end value for temperature,
 * electron density, and length. The observed spectrum is the time-weighted sum
 * over this evolution during the integration window.
 */
struct LTETimeIntegrated : public LTEModel {

  /// @brief Initial temperature T0 in Kelvin.
  double start_temperature;
  /// @brief Asymptotic temperature Tf in Kelvin.
  double end_temperature;
  /// @brief Temperature decay time constant.
  double tau_temperature;
  /// @brief Initial electron density ne0 in m^-3.
  double start_electron_density;
  /// @brief Asymptotic electron density nef in m^-3.
  double end_electron_density;
  /// @brief Electron-density decay time constant.
  double tau_electron_density;
  /// @brief Initial plasma length L0 in meters.
  double start_length;
  /// @brief Asymptotic plasma length Lf in meters.
  double end_length;
  /// @brief Length decay time constant.
  double tau_length;

  /// @brief Total integration window.
  double integration_time;
  /// @brief Number of discrete time points used in the weighted sum.
  int integration_points;

  /**
   * @brief Construct an LTETimeIntegrated object.
   *
   * Temporal evolution follows:
   *   T(t)  = T_f  + (T_0  - T_f)  * exp(-t / tau_T)
   *   ne(t) = ne_f + (ne_0 - ne_f) * exp(-t / tau_ne)
   *   L(t)  = L_f  + (L_0  - L_f)  * exp(-t / tau_L)
    *
    * @param _element_ratio Mass ratio of elements in the model.
    * @param _start_temperature Initial temperature T0 in Kelvin.
    * @param _end_temperature Asymptotic temperature Tf in Kelvin.
    * @param _tau_temperature Temperature decay time constant.
    * @param _start_electron_density Initial electron density ne0 in m^-3.
    * @param _end_electron_density Asymptotic electron density nef in m^-3.
    * @param _tau_electron_density Electron-density decay time constant.
    * @param _start_length Initial length L0 in meters.
    * @param _end_length Asymptotic final length Lf in meters.
    * @param _tau_length Length decay time constant.
    * @param _integration_time Total integration time window.
    * @param _integration_points Number of midpoint samples for the weighted sum.
   */
  LTETimeIntegrated(std::map<std::string, double> _element_ratio,
                    double _start_temperature, double _end_temperature,
                    double _tau_temperature, double _start_electron_density,
                    double _end_electron_density, double _tau_electron_density,
                    double _start_length, double _end_length,
                    double _tau_length, double _integration_time,
                    int _integration_points = 5);

  /** @brief Create an exact copy of the time-integrated model. */
  std::shared_ptr<PlasmaModel> clone() const override;

  /** @brief Access a model parameter by name. */
  double &get_parameter(const std::string &parameter_name) override;

  /** @brief Set a model parameter by name. */
  void set_parameter(const std::string &parameter_name, double value) override;

  /**
   * @brief List all exposed model parameters.
   *
   * Includes the base LTE parameters plus the start/end/tau values and the
   * integration window.
   */
  std::vector<std::string> get_parameter_names() const override;

  /**
   * @brief Calculate the time-averaged irradiance over the integration window.
   *
   * Computes a midpoint-rule time average of the instantaneous LTE irradiance.
   */
  Eigen::VectorXd
  calculate_irradiance(const Eigen::VectorXd &wavelengths) const override;
};

} // namespace libskit

#endif // LTETIMEINTEGRATED_H
