/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef PLASMAMODEL_H
#define PLASMAMODEL_H

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "AtomicData/elemental_data.h"

namespace libskit {

/**
 * @brief Abstract base class for plasma models.
 *
 * Plasma models provide the intrinsic emission physics used by instrument
 * models. Implementations expose their parameters by name, calculate
 * irradiance on arbitrary wavelength grids, and estimate the narrowest
 * relevant line width so instruments can choose an adequate fine grid for
 * spectral convolution.
 */
struct PlasmaModel {

  /**
   * @brief Calculate the spectral irradiance of the plasma.
   *
   * @param wavelengths Wavelength grid in nanometers.
   * @return Irradiance values corresponding to each wavelength.
   */
  virtual Eigen::VectorXd
  calculate_irradiance(const Eigen::VectorXd &wavelengths) const = 0;

  /**
   * @brief Get a reference to a model parameter by name.
    *
    * The exact parameter set depends on the concrete plasma model.
   */
  virtual double &get_parameter(const std::string &parameter_name) = 0;

  /**
   * @brief Set a model parameter by name.
    *
    * The exact parameter set depends on the concrete plasma model.
   */
  virtual void set_parameter(const std::string &parameter_name,
                             double value) = 0;

  /**
   * @brief Get names of all available model parameters.
    *
    * These names are the valid keys for get_parameter() and set_parameter().
   */
  virtual std::vector<std::string> get_parameter_names() const = 0;

  /**
   * @brief Get the minimum Voigt FWHM of all emission lines in the given range.
   *
   * Used by instruments to determine the fine grid resolution needed
   * for accurate FFT-based convolution.
    *
    * @param wavelengths Wavelength range of interest.
    * @return Minimum Voigt FWHM in nm across all contributing lines.
   */
  virtual double
  get_minimum_voigt_fwhm(const Eigen::VectorXd &wavelengths) const = 0;

  /**
   * @brief Get the fine grid spacing determined by the narrowest emission line.
    *
    * Returns min_voigt_fwhm / fine_samples_per_fwhm().
    *
    * @param wavelengths Wavelength range of interest.
    * @return Fine-grid spacing in nm.
   */
  double get_fine_grid_spacing(const Eigen::VectorXd &wavelengths) const {
    return get_minimum_voigt_fwhm(wavelengths) / get_fine_samples_per_fwhm();
  }

  /**
   * @brief Build a uniform fine wavelength grid for accurate spectral computation.
    *
    * Constructs a grid spanning the input wavelength range with spacing chosen
    * from the narrowest contributing line, but never coarser than the user grid.
    *
    * @param wavelengths Requested wavelength range.
    * @return Fine uniform wavelength grid.
   */
  Eigen::VectorXd
  build_fine_wavelength_grid(const Eigen::VectorXd &wavelengths) const {
    const double dl_user = wavelengths[1] - wavelengths[0];
    const double dl_fine = get_fine_grid_spacing(wavelengths);
    const double dl = std::min(dl_fine, dl_user);

    const double wl_start = wavelengths[0];
    const double wl_end = wavelengths[wavelengths.size() - 1];
    const long n = std::max(
        2L, static_cast<long>(std::ceil((wl_end - wl_start) / dl)) + 1);

    Eigen::VectorXd wl(n);
    for (long i = 0; i < n; ++i)
      wl[i] = wl_start + i * dl;
    return wl;
  }

  /**
   * @brief Create an exact copy of the plasma model.
   */
  virtual std::shared_ptr<PlasmaModel> clone() const = 0;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~PlasmaModel() = default;
};
} // namespace libskit

#endif
