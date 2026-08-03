/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef LTEMODEL_H
#define LTEMODEL_H

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "AtomicData/elemental_data.h"
#include "PlasmaModel.h"
#include "functions.h"

namespace libskit {

/**
 * @brief LTEModel represents a Local Thermodynamic Equilibrium plasma model.
 *
 * This class models a homogeneous plasma in local thermodynamic equilibrium,
 * parameterized by elemental mass ratios, temperature, electron density, and
 * plasma length. It provides the line and continuum emission needed by the
 * instrument models.
 *
 * The model exposes helpers for population fractions, line-wise emission
 * coefficients, continuum emission, and radiative transfer. Those helpers are
 * used directly by some callers and also underpin the higher-level irradiance
 * calculation used during fitting.
 */
struct LTEModel : public PlasmaModel {
  /// @brief Mass ratios of the plasma elements.
  std::map<std::string, double> element_ratio;
  /// @brief Plasma temperature in Kelvin.
  double temperature;
  /// @brief Electron number density in m^-3.
  double electron_density;
  /// @brief Physical plasma length in meters.
  double length;
  /// @brief Whether to include continuum emission contributions.
  bool include_continuum = false;

  /**
   * @brief Construct an LTEModel.
   *
   * @param _element_ratio Element mass ratios.
   * @param _temperature Plasma temperature in Kelvin.
   * @param _electron_density Electron density in m^-3.
   * @param _length Plasma length in meters.
   */
  LTEModel(std::map<std::string, double> _element_ratio, double _temperature,
           double _electron_density, double _length);

  /**
   * @brief Create an exact copy of the LTE model.
   */
  std::shared_ptr<PlasmaModel> clone() const override;

    /**
     * @brief Access a parameter by name.
     *
     * Valid names are temperature, electron_density, length, or any element
     * symbol present in element_ratio.
     */
  double &get_parameter(const std::string &parameter_name) override;
    /**
     * @brief Set a parameter by name.
     *
     * Valid names are temperature, electron_density, length, or any element
     * symbol present in element_ratio.
     */
  void set_parameter(const std::string &parameter_name, double value) override;
    /**
     * @brief List all available parameter names.
     *
     * Returns the three main plasma parameters followed by all element symbols.
     */
  std::vector<std::string> get_parameter_names() const override;

  /**
   * @brief Calculate the weighted sum of ionization fractions used in the LTE
     * density relations.
     *
     * Computes
     *   f_sum = sum_k (c_k / M_k) * sum_i (i * n_k,i)
     * which couples the ionization populations to the requested electron
     * density.
   */
  double calculate_f_sum(
      const std::map<std::string, Eigen::VectorXd> &population_fractions) const;

  /**
   * @brief Compute ionization population fractions for all plasma elements.
    *
    * The population fractions are obtained from Saha equilibrium for each
    * element at the current temperature and electron density.
   */
  std::map<std::string, Eigen::VectorXd> get_population_fractions() const;

  /**
   * @brief Calculate element number densities from the LTE state.
    *
    * Returns particle densities in m^-3 for each element species.
   */
  std::map<std::string, double> _calculate_element_fractions() const;

  /**
   * @brief Calculate the emission coefficient for one spectral line.
    *
    * Uses the partition function, ionization population, and Einstein-A data to
    * compute the spontaneous-emission coefficient for a single transition.
   */
  double calculate_emission_for_one_line(
      const unsigned int ionization_level, const double n_k,
      const Eigen::VectorXd &population_fraction,
      const double partition_function, const EmissionLine &line_data) const;

  /**
   * @brief Gather emission-line data within the requested wavelength range.
    *
    * Returns both the per-ion emission data and the maximum line-center
    * irradiance peak, which is later used for accuracy-based filtering.
   */
  std::pair<EmissionData<double>, double> get_emission_coefficient_data(
      const Eigen::VectorXd &wavelengths) const;

  /**
   * @brief Calculate the full line emission coefficient spectrum.
    *
    * Applies Doppler and Stark broadening, line shifts, and optional continuum
    * emission to produce a continuous emission-coefficient spectrum.
   */
  Eigen::VectorXd calculate_emission_coefficient(
      const Eigen::VectorXd &wavelengths) const;

  /**
   * @brief Compute free-free and free-bound continuum emission separately.
    *
    * Uses the hydrogen-approximation formulas from Noll (2012), Eqs.
    * 10.14-10.18.
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd>
  calculate_continuum_emission(const Eigen::VectorXd &wavelengths) const;

  /**
   * @brief Add continuum emission to an existing line-emission spectrum.
    *
    * The continuum contribution is added in-place.
   */
  void add_continuum_emission(const Eigen::VectorXd &wavelengths,
                              Eigen::VectorXd &emission_coefficient) const;

  /**
   * @brief Calculate spectral irradiance by applying LTE radiative transfer.
    *
    * Solves
    *   I(lambda) = B(lambda, T) * (1 - exp(-epsilon(lambda) / B(lambda, T) * L)).
   */
  Eigen::VectorXd
  calculate_irradiance(const Eigen::VectorXd &wavelengths) const override;

  /**
   * @brief Return the lines that contribute to the requested wavelength range.
    *
    * For each line this reports ion identity, wavelength, peak emission, Voigt
    * FWHM, and the line-accuracy fraction from the database.
   */
  std::vector<std::tuple<std::string, double, double, double, double>>
  get_lines(const Eigen::VectorXd &wavelengths) const;

  /**
   * @brief Get the minimum Voigt FWHM among contributing lines.
   */
  double get_minimum_voigt_fwhm(
      const Eigen::VectorXd &wavelengths) const override;
};

} // namespace libskit

#endif // LTEMODEL_H
