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
 *
 * The late state is *not* stored directly. Instead the model stores the
 * early state plus three non-negative "delta" parameters that encode the
 * physically expected direction of plasma evolution during the gate
 * (cooling, recombination, expansion):
 *
 *   late_temperature       = early_temperature       - delta_temperature
 *   late_electron_density  = early_electron_density  - delta_electron_density
 *   late_length             = early_length             + delta_length
 *
 * This is the parametrization exposed to the optimizer via
 * `get_parameter_names()` / `get_parameter()` / `set_parameter()`: only the
 * early_* and delta_* parameters (plus `weight_early`) are fittable. The
 * late_* values remain readable as computed, read-only quantities.
 */
struct LTETwoTime : public LTEModel {

  /// @brief Early-time plasma temperature [K].
  double early_temperature;
  /// @brief Early-time electron density [m^-3].
  double early_electron_density;
  /// @brief Early-time plasma length [m].
  double early_length;
  /// @brief early_temperature - late_temperature [K]. Expected > 0 (cooling).
  double delta_temperature;
  /// @brief early_electron_density - late_electron_density [m^-3]. Expected > 0 (recombination).
  double delta_electron_density;
  /// @brief late_length - early_length [m]. Expected > 0 (expansion).
  double delta_length;
  /// @brief Weight of the early-time contribution.
  double weight_early;

public:
  /**
   * @brief Construct an LTETwoTime model from the early state and the
   * early-to-late deltas (the parametrization used by the optimizer).
   *
   * @param _element_ratio Mass ratios of elements (shared between both times).
   * @param _early_temperature Early plasma temperature [K].
   * @param _early_electron_density Early electron density [m^-3].
   * @param _early_length Early plasma length [m].
   * @param _delta_temperature early_temperature - late_temperature [K].
   * @param _delta_electron_density early_electron_density - late_electron_density [m^-3].
   * @param _delta_length late_length - early_length [m].
   * @param _weight_early Weight of the early contribution in [0, 1].
   *                      The late weight is implicitly 1 - weight_early.
   */
  LTETwoTime(std::map<std::string, double> _element_ratio,
             double _early_temperature, double _early_electron_density,
             double _early_length, double _delta_temperature,
             double _delta_electron_density, double _delta_length,
             double _weight_early);

  /**
   * @brief Construct an LTETwoTime model directly from early and late
   * plasma states, converting late_* into the internal delta_*
   * representation. Convenience factory for callers who think in terms of
   * absolute early/late states rather than deltas.
   */
  static LTETwoTime
  from_late_state(std::map<std::string, double> element_ratio,
                  double early_temperature, double early_electron_density,
                  double early_length, double late_temperature,
                  double late_electron_density, double late_length,
                  double weight_early = 0.5);

  /** @brief Computed late-time plasma temperature [K]. */
  double late_temperature() const { return early_temperature - delta_temperature; }
  /** @brief Computed late-time electron density [m^-3]. */
  double late_electron_density() const { return early_electron_density - delta_electron_density; }
  /** @brief Computed late-time plasma length [m]. */
  double late_length() const { return early_length + delta_length; }

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
