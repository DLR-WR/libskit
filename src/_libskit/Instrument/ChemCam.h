/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef CHEMCAM_H
#define CHEMCAM_H

#include "Instrument.h"

namespace libskit {

/**
 * @brief ChemCam spectrometer model (UV / VIO / VISNIR).
 *
 * Models the ChemCam LIBS instrument, which consists of three separate CCD
 * spectrometers covering different wavelength ranges with different spectral
 * resolutions:
 *
 *   - UV:     240.1 - 342.2 nm,  FWHM = 0.15 nm
 *   - VIO:    382.1 - 469.3 nm,  FWHM = 0.20 nm
 *   - VISNIR: 474.0 - 906.5 nm,  FWHM = 0.61 nm
 *
 * Each CCD is broadened with a pure Gaussian via FFT by delegating to
 * GenericSpectrometer. Wavelengths that fall into the detector gaps receive
 * zero intensity, matching the physical instrument response.
 */
class ChemCam : public Instrument {

public:
  /// UV range limits (nm)
  static constexpr double UV_MIN = 240.1;
  static constexpr double UV_MAX = 342.2;
  static constexpr double UV_FWHM = 0.15;

  /// VIO range limits (nm)
  static constexpr double VIO_MIN = 382.1;
  static constexpr double VIO_MAX = 469.3;
  static constexpr double VIO_FWHM = 0.20;

  /// VISNIR range limits (nm)
  static constexpr double VISNIR_MIN = 474.0;
  static constexpr double VISNIR_MAX = 906.5;
  static constexpr double VISNIR_FWHM = 0.61;

  /// Per-CCD relative scale factors. UV is the reference (implicitly 1).
  double scale_vio = 1.0;
  double scale_visnir = 1.0;

  /**
   * @brief Default constructor for the ChemCam spectrometer.
   */
  ChemCam() = default;

  /**
   * @brief Create an exact copy of the ChemCam spectrometer.
   */
  std::shared_ptr<Instrument> clone() const override;

  /**
   * @brief Calculate the measured spectrum with ChemCam instrumental
   * broadening.
   *
   * Splits the wavelength grid into up to three sub-ranges matching the UV,
   * VIO, and VISNIR CCDs. Each sub-range is convolved with the appropriate
   * Gaussian kernel, and wavelengths in the inter-detector gaps are set to
   * zero.
   *
   * @param wavelengths Wavelength grid in nanometers (vacuum).
   * @param plasma_model Plasma model providing intrinsic emission.
   * @return Broadened spectrum as measured by ChemCam.
   */
  Eigen::VectorXd
  calculate_spectrum(const Eigen::VectorXd &wavelengths,
                     const PlasmaModel &plasma_model) const override;
};

} // namespace libskit
#endif // CHEMCAM_H
