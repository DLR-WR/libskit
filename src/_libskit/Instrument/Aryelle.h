/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef ARYELLE_H
#define ARYELLE_H

#include "Instrument.h"

namespace libskit {

/**
 * @brief Aryelle echelle spectrometer model.
 *
 * Convolves an emission spectrum with Aryelle's wavelength-dependent Voigt
 * instrumental profile. The Gaussian contribution follows the calibrated
 * wavelength-dependent FWHM, while the Lorentzian contribution comes from the
 * line-specific Stark broadening in the active database.
 *
 * The convolution is evaluated by direct trapezoid integration over a dynamic
 * window around each output wavelength. This is more expensive than the FFT
 * models but closely mirrors the original Aryelle broadening workflow.
 *
 * The Gaussian calibration is derived from Hg / Kr / Ne lamp lines, so this
 * class is the highest-fidelity option when that original broadening path is
 * more important than runtime.
 */
class Aryelle : public Instrument {

public:
  /**
   * @brief Default constructor for the Aryelle spectrometer.
   */
  Aryelle() = default;

  /**
   * @brief Create an exact copy of the Aryelle spectrometer.
   */
  std::shared_ptr<Instrument> clone() const override;

  /**
   * @brief Calculate the measured spectrum as recorded by the Aryelle
   * spectrometer.
    *
    * This function simulates how the plasma emission would be recorded by the
    * Aryelle echelle spectrometer using direct numerical integration of the
    * wavelength-dependent Voigt kernel.
   *
   * @param wavelengths Wavelength grid in nanometers.
   * @param plasma_model Plasma model providing intrinsic emission.
   * @return Spectrum after Aryelle instrumental broadening.
   */
  Eigen::VectorXd
  calculate_spectrum(const Eigen::VectorXd &wavelengths,
                     const PlasmaModel &plasma_model) const override;
};

} // namespace libskit

#endif // !ARYELLE_H
