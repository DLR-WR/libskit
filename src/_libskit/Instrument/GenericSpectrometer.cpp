/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "GenericSpectrometer.h"
#include "ConvolutionEngine.h"
#include "FFT.h"
#include "functions.h"
#include <algorithm>
#include <cmath>

namespace libskit {

GenericSpectrometer::GenericSpectrometer(double _sigma, double _gamma)
    : sigma(_sigma), gamma(_gamma) {}

std::shared_ptr<Instrument> GenericSpectrometer::clone() const {
  return std::make_shared<GenericSpectrometer>(*this);
}

Eigen::VectorXd GenericSpectrometer::calculate_spectrum(
    const Eigen::VectorXd &wavelengths,
    const PlasmaModel &plasma_model) const {

  const long n_out = wavelengths.size();

  // 1. Build fine grid (resolution determined by the model's narrowest line)
  Eigen::VectorXd wl_fine =
      plasma_model.build_fine_wavelength_grid(wavelengths);
  const double dl = wl_fine[1] - wl_fine[0];
  const long n_fine = wl_fine.size();

  // 2. Evaluate irradiance on the fine grid
  Eigen::VectorXd irradiance = plasma_model.calculate_irradiance(wl_fine);

  // 4. Build Voigt kernel on the fine grid spacing
  const double kernel_extent = compute_kernel_extent(sigma, gamma, get_accuracy());
  const long half = static_cast<long>(std::ceil(kernel_extent / dl));
  const long n_kernel = 2 * half + 1;

  Eigen::VectorXd kernel(n_kernel);
  for (long i = 0; i < n_kernel; ++i) {
    kernel[i] = voigt_profile_approx<double>((i - half) * dl, sigma, gamma);
  }
  // Scale by dl to get quadrature weights (Σ kernel * dl ≈ 1),
  // then normalize to exactly sum to 1 to compensate for truncation
  double kernel_sum = kernel.sum() * dl;
  kernel *= dl;
  kernel /= kernel_sum;

  // 5. Pad to next power-of-2 for linear convolution
  const long conv_len = n_fine + n_kernel - 1;
  const long N = next_power_of_two(conv_len);

  // Padded signal: emission at the front, zeros after
  Eigen::VectorXcd signal_padded = Eigen::VectorXcd::Zero(N);
  for (long i = 0; i < n_fine; ++i) {
    signal_padded[i] = irradiance[i];
  }

  // Padded kernel with center at index 0 (DFT-ready wrap-around):
  //   kernel[half]   -> index 0        (center)
  //   kernel[half+k] -> index k        (right wing)
  //   kernel[half-k] -> index N - k    (left wing wrapped)
  Eigen::VectorXcd kernel_padded = Eigen::VectorXcd::Zero(N);
  for (long i = 0; i <= half; ++i) {
    kernel_padded[i] = kernel[half + i];
  }
  for (long i = 1; i <= half; ++i) {
    kernel_padded[N - i] = kernel[half - i];
  }

  // 6. FFT convolution (signal_padded becomes IFFT(FFT(s)*FFT(k)) in-place)
  fft_convolve_inplace(signal_padded, kernel_padded);

  // 7. Extract the valid convolution result (first n_fine elements);
  //    clamp tiny FFT round-off negatives (~1e-12 of peak) to zero.
  Eigen::VectorXd convolved;
  extract_real_clamp(signal_padded, n_fine, convolved);

  // 8. Interpolate from fine grid to user's output grid
  Eigen::VectorXd result(n_out);
  const double wl_start = wl_fine[0];
  for (long i = 0; i < n_out; ++i) {
    double pos = (wavelengths[i] - wl_start) / dl;
    long idx = static_cast<long>(pos);
    if (idx < 0)
      idx = 0;
    if (idx >= n_fine - 1)
      idx = n_fine - 2;
    double frac = pos - idx;
    result[i] = convolved[idx] * (1.0 - frac) + convolved[idx + 1] * frac;
  }

  return result;
}

} // namespace libskit
