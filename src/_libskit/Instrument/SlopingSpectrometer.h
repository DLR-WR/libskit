/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef SLOPINGSPECTROMETER_H
#define SLOPINGSPECTROMETER_H

#include "Instrument.h"

namespace libskit {

/**
 * @brief Spectrometer with wavelength-dependent Gaussian broadening via
 *        change-of-variable FFT convolution.
 *
 * The instrumental broadening is modelled as a Gaussian whose FWHM grows
 * linearly with wavelength:
 *
 *   FWHM(lambda) = m * lambda + b  (nm)
 *
 * A direct FFT convolution cannot handle a position-dependent kernel. This
 * model instead transforms the wavelength grid into a coordinate where the
 * broadening becomes constant, performs a fixed-kernel FFT convolution there,
 * and interpolates back to wavelength space.
 */
class SlopingSpectrometer : public Instrument {

public:
  /// @brief Slope of FWHM(λ) in nm/nm (dimensionless)
  double fwhm_slope;

  /// @brief Intercept of FWHM(λ) in nm
  double fwhm_intercept;

  /**
   * @brief Construct a SlopingSpectrometer.
   *
   * @param _fwhm_slope Slope m of FWHM(lambda) = m * lambda + b.
   * @param _fwhm_intercept Intercept b of FWHM(lambda) = m * lambda + b.
   */
  SlopingSpectrometer(double _fwhm_slope, double _fwhm_intercept);

  /**
   * @brief Create an exact copy of the spectrometer.
   */
  std::shared_ptr<Instrument> clone() const override;

  /**
   * @brief Calculate the measured spectrum with wavelength-dependent Gaussian
   * broadening.
   */
  Eigen::VectorXd
  calculate_spectrum(const Eigen::VectorXd &wavelengths,
                     const PlasmaModel &plasma_model) const override;
};

} // namespace libskit
#endif // SLOPINGSPECTROMETER_H
