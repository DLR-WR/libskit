/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "ChemCam.h"
#include "GenericSpectrometer.h"
#include <cmath>

namespace libskit {

// FWHM → Gaussian sigma conversion factor: sigma = FWHM / (2 * sqrt(2 * ln(2)))
static constexpr double FWHM_TO_SIGMA = 1.0 / (2.0 * 1.17741002251547469);

std::shared_ptr<Instrument> ChemCam::clone() const {
  return std::make_shared<ChemCam>(*this);
}

Eigen::VectorXd ChemCam::calculate_spectrum(
    const Eigen::VectorXd &wavelengths,
    const PlasmaModel &plasma_model) const {

  const long n = wavelengths.size();
  Eigen::VectorXd result = Eigen::VectorXd::Zero(n);

  struct Range {
    double wl_min, wl_max, fwhm;
  };
  // Spectral ranges: {min, max, fwhm}
  const Range ranges[] = {
      {UV_MIN, UV_MAX, UV_FWHM},
      {VIO_MIN, VIO_MAX, VIO_FWHM},
      {VISNIR_MIN, VISNIR_MAX, VISNIR_FWHM},
  };

  // Per-range relative scale factors: UV is reference (1), VIO and VISNIR are free.
  const double range_scales[] = {1.0, this->scale_vio, this->scale_visnir};

  int range_idx = 0;
  for (const auto &range : ranges) {
    long i_start = -1, i_end = -1;
    // Find indices that fall inside the current detector window.
    for (long i = 0; i < n; ++i) {
      if (wavelengths[i] >= range.wl_min && wavelengths[i] <= range.wl_max) {
        if (i_start < 0)
          i_start = i;
        i_end = i;
      }
    }

    if (i_start < 0) {
      ++range_idx;
      continue;
    }

    const long n_sub = i_end - i_start + 1;

  // Extract the CCD-local wavelength grid.
    Eigen::VectorXd wl_sub = wavelengths.segment(i_start, n_sub);

  // Create a Gaussian-only GenericSpectrometer for this detector.
    const double sigma = range.fwhm * FWHM_TO_SIGMA;
    GenericSpectrometer sub_spectrometer(sigma, 0.0);

  // Reuse the GenericSpectrometer FFT convolution implementation.
    Eigen::VectorXd sub_result =
        sub_spectrometer.calculate_spectrum(wl_sub, plasma_model);

  // Copy the detector result back into the full output and apply its scale.
    result.segment(i_start, n_sub) = sub_result * range_scales[range_idx];
    ++range_idx;
  }

  return result;
}

} // namespace libskit
