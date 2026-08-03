/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "Aryelle.h"
#include "functions.h"

namespace libskit {

std::shared_ptr<Instrument> Aryelle::clone() const {
  return std::make_shared<Aryelle>(*this);
}

Eigen::VectorXd
Aryelle::calculate_spectrum(const Eigen::VectorXd &wavelengths,
                            const PlasmaModel &plasma_model) const {

  // Evaluate the intrinsic plasma irradiance on the requested wavelength grid.
  Eigen::VectorXd irradiance = plasma_model.calculate_irradiance(wavelengths);

  // Each output wavelength accumulates a direct trapezoid integral over the
  // broadened source spectrum.
  Eigen::VectorXd spectrum = Eigen::VectorXd::Zero(wavelengths.size());

  for (unsigned int i = 0; i < wavelengths.size() - 1; i++) {

    // σ and γ of the Voigt instrumental broadening, linear in λ
    // (Oriel calibration lamps, see libskit_experiments/experiments/
    // wavelength_calibration). The Lorentzian contribution is negligible.
    double sigma = 6.3984e-05 * wavelengths[i] + 3.3232e-03;
    double fwhm_lorentz = 0.0;
    double gamma = fwhm_lorentz / 2.0;

    auto indices =
        get_indices_via_heuristic(wavelengths, sigma, gamma, wavelengths[i]);
    auto [lower_index, upper_index] = indices;

    for (long j = lower_index + 1; j < upper_index; j++) {

      // Trapezoid rule: integral_a^b f(x) dx ≈ 0.5 * (b - a) * (f(a) + f(b)).
      spectrum[i] +=
          0.5 * (wavelengths[j] - wavelengths[j - 1]) *
          (irradiance[j - 1] *
               voigt_profile_approx(wavelengths[i] - wavelengths[j - 1], sigma,
                                    gamma) +
           irradiance[j] * voigt_profile_approx(wavelengths[i] - wavelengths[j],
                                                sigma, gamma));
    }
  }

  return spectrum;
}

} // namespace libskit
