/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef GENERICSPECTROMETER_H
#define GENERICSPECTROMETER_H

#include "Instrument.h"

namespace libskit {

/**
 * @brief Generic spectrometer model with configurable line broadening.
 *
 * This class applies instrumental broadening to plasma spectra by convolving
 * the intrinsic irradiance with a Voigt profile, combining Gaussian and
 * Lorentzian line shapes. The implementation uses FFT-based convolution for
 * efficiency on dense wavelength grids.
 */
class GenericSpectrometer : public Instrument {

public:
  /// @brief Gaussian broadening parameter (standard deviation) in nm
  double sigma;

  /// @brief Lorentzian broadening parameter (half-width at half-maximum) in nm
  double gamma;

  /**
   * @brief Construct a GenericSpectrometer with specified broadening
   * parameters.
   *
   * @param _sigma Gaussian broadening parameter (standard deviation) in nm.
   * @param _gamma Lorentzian broadening parameter (HWHM) in nm.
   */
  GenericSpectrometer(double _sigma, double _gamma);

  /**
   * @brief Create an exact copy of the spectrometer.
   *
   * @return Shared pointer to a newly created copy of the spectrometer.
   */
  std::shared_ptr<Instrument> clone() const override;

  /**
   * @brief Calculate the measured spectrum with instrumental broadening
   * applied.
   *
   * This simulates how the plasma emission would be recorded by the
   * spectrometer by evaluating the intrinsic irradiance on a fine wavelength
   * grid, convolving it with a Voigt kernel, and interpolating the result back
   * to the requested output grid.
   *
   * @param wavelengths Wavelength grid in nanometers.
   * @param plasma_model Plasma model providing the intrinsic emission.
   * @return Vector of broadened intensities representing the measured spectrum.
   */
  Eigen::VectorXd
  calculate_spectrum(const Eigen::VectorXd &wavelengths,
                     const PlasmaModel &plasma_model) const override;
};

} // namespace libskit
#endif
