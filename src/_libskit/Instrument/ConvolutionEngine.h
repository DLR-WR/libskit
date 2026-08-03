/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file ConvolutionEngine.h
 * @brief Shared FFT-based linear-convolution primitives for spectrometers.
 *
 * libskit's frequency-domain spectrometer broadening (`GenericSpectrometer`,
 * `SlopingSpectrometer`, and the per-CCD path of `ChemCam`) all do the same
 * three things:
 *
 *   1. Pad the signal and kernel to the next power of two,
 *   2. Run forward FFT on both, multiply pointwise, run the inverse FFT,
 *   3. Take the real part of the result (and optionally clamp the small
 *      negative excursions that the recursive FFT introduces at ~1e-12).
 *
 * Before this header existed, every instrument hand-rolled all three steps,
 * which made it easy for sign / normalization / clamping policies to drift
 * apart between files. The helpers here factor the common pieces out so that
 * each instrument only owns the *unique* bits of its forward model
 * (kernel construction, Jacobian weighting, fftshift policy, interpolation
 * back to the requested grid).
 *
 * Sign convention: the underlying `fft()` / `ifft()` in `FFT.h` use the
 * `+2πi/N` exponent for the forward transform and `(1/N)·exp(-2πi/N)` for
 * the inverse. The convolution helpers here are sign-agnostic since they
 * always do `IFFT(FFT(s) · FFT(k))`.
 */
#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include "FFT.h"
#include <Eigen/Core>
#include <complex>

namespace libskit {

/**
 * @brief Smallest power of two that is >= @p n.
 *
 * Used by every FFT-convolving spectrometer to pick a transform length that
 * is large enough to hold the linear convolution (`n_signal + n_kernel - 1`)
 * without wrap-around. For @p n <= 0 this returns 1.
 */
inline long next_power_of_two(long n) {
  long N = 1;
  while (N < n)
    N <<= 1;
  return N;
}

/**
 * @brief In-place linear-convolution kernel: signal ← IFFT(FFT(signal) · FFT(kernel)).
 *
 * Both vectors must already be the same power-of-two length and contain the
 * caller-prepared signal / kernel layout (zero-padding, kernel wrapping,
 * Jacobian weighting, ...). After the call, @p signal holds the (still
 * complex) convolution result and @p kernel holds its forward FFT.
 *
 * Thin wrapper, but having a single named function makes intent obvious at
 * call sites and keeps the FFT call ordering consistent across instruments.
 */
template <typename T>
void fft_convolve_inplace(Eigen::VectorX<std::complex<T>> &signal,
                          Eigen::VectorX<std::complex<T>> &kernel) {
  fft(signal);
  fft(kernel);
  signal = signal.cwiseProduct(kernel);
  ifft(signal);
}

/**
 * @brief Copy the first @p n_valid real parts of @p complex_in into
 *        @p real_out, clamping any negative samples to zero.
 *
 * The recursive FFT in `FFT.cpp` introduces tiny negative excursions
 * (roughly 1e-12 of the peak) for what should be non-negative spectra; this
 * helper centralises the clamping policy used by `GenericSpectrometer` and
 * the `ChemCam` per-CCD path. Spectrometers that compensate via Jacobian
 * weighting (e.g. `SlopingSpectrometer`) do not call this.
 */
template <typename T>
void extract_real_clamp(const Eigen::VectorX<std::complex<T>> &complex_in,
                        long n_valid, Eigen::VectorX<T> &real_out) {
  real_out.resize(n_valid);
  for (long i = 0; i < n_valid; ++i) {
    T val = complex_in[i].real();
    real_out[i] = val > T(0.0) ? val : T(0.0);
  }
}

} // namespace libskit

#endif // CONVOLUTION_ENGINE_H
