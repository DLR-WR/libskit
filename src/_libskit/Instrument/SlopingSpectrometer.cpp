/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "SlopingSpectrometer.h"
#include "ConvolutionEngine.h"
#include "FFT.h"
#include "functions.h"
#include <cmath>
#include <numbers>

namespace libskit {

SlopingSpectrometer::SlopingSpectrometer(double _fwhm_slope,
                                         double _fwhm_intercept)
    : fwhm_slope(_fwhm_slope), fwhm_intercept(_fwhm_intercept) {}

std::shared_ptr<Instrument> SlopingSpectrometer::clone() const {
  return std::make_shared<SlopingSpectrometer>(*this);
}

Eigen::VectorXd SlopingSpectrometer::calculate_spectrum(
    const Eigen::VectorXd &wavelengths,
    const PlasmaModel &plasma_model) const {

  // α = m / 2.355,  β = b / 2.355
  const double fwhm_to_sigma = 2.0 * std::sqrt(2.0 * std::log(2.0));
  const double alpha = fwhm_slope / fwhm_to_sigma;
  const double beta = fwhm_intercept / fwhm_to_sigma;

  const long N_orig = wavelengths.size();
  const double lam_min = wavelengths[0];
  const double lam_max = wavelengths[N_orig - 1];

  // ── 1. Build uniform x-grid ────────────────────────────────────────
  // x(lambda) = (1 / alpha) * ln(alpha * lambda + beta)
  const double x_min = std::log(alpha * lam_min + beta) / alpha;
  const double x_max = std::log(alpha * lam_max + beta) / alpha;

  const long N_signal = N_orig;
  const double dx = (x_max - x_min) / (N_signal - 1);
  const double sigma_pix = 1.0 / dx;

  // Zero-pad to the next power of two for the FFT and to avoid circular
  // wrap-around in the linear convolution.
  const long N = next_power_of_two(static_cast<long>(N_signal));

  Eigen::VectorXd x_grid(N_signal);
  Eigen::VectorXd lam_grid(N_signal);
  for (long i = 0; i < N_signal; i++) {
    x_grid[i] = x_min + i * dx;
    // lambda(x) = (exp(alpha * x) - beta) / alpha
    lam_grid[i] = (std::exp(alpha * x_grid[i]) - beta) / alpha;
  }

  // ── 2. Evaluate irradiance on the non-uniform λ-grid ──────────────
  Eigen::VectorXd irradiance = plasma_model.calculate_irradiance(lam_grid);

  // ── 3. Weight by Jacobian and zero-pad ────────────────────────────
  Eigen::VectorXcd signal = Eigen::VectorXcd::Zero(N);
  for (long i = 0; i < N_signal; i++) {
    // f(x) = I(lambda(x)) * d lambda / d x, with d lambda / d x = alpha * lambda + beta
    double jacobian = alpha * lam_grid[i] + beta;
    signal[i] = irradiance[i] * jacobian;
  }

  // ── 4. FFT convolution with Gaussian (σ_x = 1) ────────────────────
  Eigen::VectorXcd kernel = Eigen::VectorXcd::Zero(N);
  const Eigen::VectorXd x_kernel =
      Eigen::VectorXd::LinSpaced(N, -N / 2, N / 2);
  for (int i = 0; i < N; i++) {
    double val =
        std::exp(-0.5 * (x_kernel[i] / sigma_pix) * (x_kernel[i] / sigma_pix));
    kernel[i] = val;
  }
  std::complex<double> ksum(0);
  for (int i = 0; i < N; i++)
    ksum += kernel[i];
  for (int i = 0; i < N; i++)
    kernel[i] /= ksum;

  // Convolution happens in x-space, where the Gaussian width is constant.
  fft_convolve_inplace(signal, kernel);

  // fftshift compensates for the kernel being centered at N/2.
  Eigen::VectorXcd shifted(N);
  shifted << signal.tail(N / 2), signal.head(N / 2);

  // ── 5. Remove Jacobian weighting ──────────────────────────────────
  Eigen::VectorXd broadened(N_signal);
  for (long i = 0; i < N_signal; i++) {
    double jacobian = alpha * lam_grid[i] + beta;
    broadened[i] = shifted[i].real() / jacobian;
  }

  // ── 6. Interpolate back to original wavelength grid ────────────────
  Eigen::VectorXd result(N_orig);
  int j = 0;
  for (long i = 0; i < N_orig; i++) {
    double lam = wavelengths[i];
    while (j < N_signal - 2 && lam_grid[j + 1] < lam) {
      j++;
    }
    if (j >= N_signal - 1) {
      result[i] = broadened[N_signal - 1];
    } else {
      double t = (lam - lam_grid[j]) / (lam_grid[j + 1] - lam_grid[j]);
      result[i] = broadened[j] * (1.0 - t) + broadened[j + 1] * t;
    }
  }

  return result;
}

} // namespace libskit
