/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef LTETWOTIME_H
#define LTETWOTIME_H

#include <map>

#include "PlasmaModel.h"
#include "PlasmaModel/LTEModel.h"

namespace libskit {

/**
 * @brief Two-time surrogate plasma model.
 *
 * Approximates a time-integrated LIBS plasma as a weighted sum of two
 * static LTE plasmas (an "early" and a "late" state) sharing the same
 * elemental composition:
 *
 *   I(λ) = w · I_early(λ) + (1 − w) · I_late(λ)
 *
 * where w ≡ weight_early ∈ [0, 1]. This is the minimal generalisation of
 * `LTEModel` that captures the dominant effect of plasma evolution.
 */
struct LTETwoTime : public LTEModel {

  /// @brief Early-time plasma temperature [K].
  double early_temperature;
  /// @brief Early-time electron density [m^-3].
  double early_electron_density;
  /// @brief Early-time plasma length [m].
  double early_length;
  /// @brief Late-time plasma temperature [K].
  double late_temperature;
  /// @brief Late-time electron density [m^-3].
  double late_electron_density;
  /// @brief Late-time plasma length [m].
  double late_length;
  /// @brief Weight of the early-time contribution.
  double weight_early;

public:
  /**
   * @brief Construct an LTETwoTime model.
   *
   * @param _element_ratio Mass ratios of elements (shared between both times).
   * @param _early_temperature Early plasma temperature [K].
   * @param _early_electron_density Early electron density [m^-3].
   * @param _early_length Early plasma length [m].
   * @param _late_temperature Late plasma temperature [K].
   * @param _late_electron_density Late electron density [m^-3].
   * @param _late_length Late plasma length [m].
   * @param _weight_early Weight of the early contribution in [0, 1].
   *                      The late weight is implicitly 1 - weight_early.
   */
  LTETwoTime(std::map<std::string, double> _element_ratio,
             double _early_temperature, double _early_electron_density,
             double _early_length, double _late_temperature,
             double _late_electron_density, double _late_length,
             double _weight_early);

  /**
   * @brief Create an exact copy of the two-time model.
   */
  std::shared_ptr<PlasmaModel> clone() const override;

  /** @brief Access a model parameter by name. */
  double &get_parameter(const std::string &parameter_name) override;
  /** @brief Set a model parameter by name. */
  void set_parameter(const std::string &parameter_name, double value) override;
  /** @brief List all available parameter names. */
  std::vector<std::string> get_parameter_names() const override;

  /**
   * @brief Calculate irradiance as the weighted sum of early and late LTE
   * states.
   */
  Eigen::VectorXd calculate_irradiance(
      const Eigen::VectorXd &wavelengths) const override;
};

} // namespace libskit

#endif // LTETWOTIME_H
