/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include "PlasmaModel/PlasmaModel.h"
#include <Eigen/Core>
#include <memory>

namespace libskit {

/**
 * @brief Abstract base class for spectroscopic instruments.
 *
 * This class defines the common interface for all instrument models in
 * libskit. Instruments map intrinsic plasma emission to the spectrum that a
 * specific spectrometer would record, including effects such as instrumental
 * broadening, detector segmentation, and interpolation to the measurement grid.
 */
class Instrument {
public:
  /**
   * @brief Calculate the measured spectrum as it would appear on the instrument.
   *
   * Derived classes implement the instrument-specific observation model.
   *
   * @param wavelengths Wavelength grid in nanometers.
   * @param plasma_model Plasma model providing the intrinsic emission.
   * @return Measured spectrum on the requested wavelength grid.
   */
  virtual Eigen::VectorXd
  calculate_spectrum(const Eigen::VectorXd &wavelengths,
                     const PlasmaModel &plasma_model) const = 0;

  /**
    * @brief Create an exact copy of the instrument model.
    *
    * @return Shared pointer to a newly created copy of the instrument.
   */
  virtual std::shared_ptr<Instrument> clone() const = 0;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~Instrument() = default;
};
} // namespace libskit

#endif // !INSTRUMENT_H
