/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file Spectrum.h
 * @brief Data structure for representing spectroscopic data.
 */
#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <Eigen/Core>

namespace libskit {

/**
 * @brief Container for spectroscopic data with wavelength-intensity pairs.
 *
 * This is the basic spectrum container used for both experimental and
 * simulated data throughout the library. Wavelengths are always stored in
 * double precision for consistency across the codebase.
 */
struct Spectrum {
  /// @brief Wavelength values in nanometers.
  Eigen::VectorXd wavelengths;
  /// @brief Corresponding irradiance or intensity values.
  Eigen::VectorXd irradiance;

  /**
   * @brief Construct a spectrum from wavelength and irradiance vectors.
   *
    * The two vectors must have the same length.
   */
  Spectrum(Eigen::VectorXd _wavelengths, Eigen::VectorXd _irradiance)
      : wavelengths(std::move(_wavelengths)),
        irradiance(std::move(_irradiance)) {}
};
} // namespace libskit

#endif // !SPECTRUM_H
