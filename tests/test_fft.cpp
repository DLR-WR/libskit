/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file test_fft.cpp
 * @brief Unit tests for the templated FFT/IFFT primitives in
 *        src/_libskit/Instrument/FFT.{h,cpp}.
 *
 * The library implements a power-of-two recursive Cooley-Tukey FFT used
 * internally for instrument convolution. These tests cover:
 *   - Round-trip identity (ifft(fft(x)) == x)
 *   - Known DFT outputs for trivial inputs
 *   - Linearity property
 *   - Edge cases (size 1, size 2)
 */

#include "../src/_libskit/Instrument/FFT.h"

#include <Eigen/Core>
#include <complex>
#include <gtest/gtest.h>

using cd = std::complex<double>;

namespace {

// Direct O(N^2) DFT for ground-truth comparison.
// NOTE: libskit's `fft` uses the +2πi/N sign convention (see
// src/_libskit/Instrument/FFT.cpp), so this reference uses the same sign.
Eigen::VectorX<cd> reference_dft(const Eigen::VectorX<cd> &x) {
  const long N = x.size();
  Eigen::VectorX<cd> X(N);
  for (long k = 0; k < N; ++k) {
    cd s(0.0, 0.0);
    for (long n = 0; n < N; ++n) {
      double ang = 2.0 * M_PI * static_cast<double>(k * n) / N;
      s += x[n] * cd(std::cos(ang), std::sin(ang));
    }
    X[k] = s;
  }
  return X;
}

void expect_close(const Eigen::VectorX<cd> &a, const Eigen::VectorX<cd> &b,
                  double tol = 1e-10) {
  ASSERT_EQ(a.size(), b.size());
  for (long i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i].real(), b[i].real(), tol)
        << "real mismatch at index " << i;
    EXPECT_NEAR(a[i].imag(), b[i].imag(), tol)
        << "imag mismatch at index " << i;
  }
}

}  // namespace

// ── Round-trip ───────────────────────────────────────────────────────

TEST(FFT, RoundTripPowerOfTwo) {
  for (long N : {2, 4, 8, 16, 32, 64, 128}) {
    Eigen::VectorX<cd> x(N);
    for (long i = 0; i < N; ++i) {
      x[i] = cd(std::sin(0.1 * i), std::cos(0.2 * i));
    }
    Eigen::VectorX<cd> y = x;
    fft(y);
    ifft(y);
    expect_close(y, x, 1e-10);
  }
}

TEST(FFT, RoundTripRealValuedSignal) {
  const long N = 64;
  Eigen::VectorX<cd> x(N);
  for (long i = 0; i < N; ++i) {
    x[i] = cd(static_cast<double>(i % 7) - 3.0, 0.0);
  }
  Eigen::VectorX<cd> y = x;
  fft(y);
  ifft(y);
  expect_close(y, x, 1e-10);
}

// ── Comparison against reference DFT ────────────────────────────────

TEST(FFT, MatchesReferenceDFT) {
  const long N = 16;
  Eigen::VectorX<cd> x(N);
  for (long i = 0; i < N; ++i) {
    x[i] = cd(0.5 * i - 3.0, 2.0 * std::sin(i));
  }
  Eigen::VectorX<cd> X_fft = x;
  fft(X_fft);
  Eigen::VectorX<cd> X_ref = reference_dft(x);
  expect_close(X_fft, X_ref, 1e-10);
}

// ── Known values ────────────────────────────────────────────────────

TEST(FFT, ImpulseInputProducesFlatSpectrum) {
  // δ[n] (impulse at n=0) → X[k] = 1 for all k.
  const long N = 8;
  Eigen::VectorX<cd> x = Eigen::VectorX<cd>::Zero(N);
  x[0] = cd(1.0, 0.0);
  fft(x);
  for (long k = 0; k < N; ++k) {
    EXPECT_NEAR(x[k].real(), 1.0, 1e-12) << "k=" << k;
    EXPECT_NEAR(x[k].imag(), 0.0, 1e-12) << "k=" << k;
  }
}

TEST(FFT, ConstantInputProducesDCOnly) {
  // Constant x[n] = c → X[0] = N*c, all other bins = 0.
  const long N = 16;
  const double c = 3.5;
  Eigen::VectorX<cd> x = Eigen::VectorX<cd>::Constant(N, cd(c, 0.0));
  fft(x);
  EXPECT_NEAR(x[0].real(), N * c, 1e-10);
  EXPECT_NEAR(x[0].imag(), 0.0, 1e-10);
  for (long k = 1; k < N; ++k) {
    EXPECT_NEAR(x[k].real(), 0.0, 1e-10) << "k=" << k;
    EXPECT_NEAR(x[k].imag(), 0.0, 1e-10) << "k=" << k;
  }
}

// ── Linearity ───────────────────────────────────────────────────────

TEST(FFT, LinearityHolds) {
  const long N = 32;
  Eigen::VectorX<cd> a(N), b(N);
  for (long i = 0; i < N; ++i) {
    a[i] = cd(std::cos(0.3 * i), 0.0);
    b[i] = cd(0.0, std::sin(0.7 * i));
  }
  const double alpha = 1.7, beta = -0.4;

  Eigen::VectorX<cd> combo(N);
  for (long i = 0; i < N; ++i) {
    combo[i] = alpha * a[i] + beta * b[i];
  }

  Eigen::VectorX<cd> A_fft = a, B_fft = b, C_fft = combo;
  fft(A_fft);
  fft(B_fft);
  fft(C_fft);

  for (long k = 0; k < N; ++k) {
    cd expected = alpha * A_fft[k] + beta * B_fft[k];
    EXPECT_NEAR(C_fft[k].real(), expected.real(), 1e-10) << "k=" << k;
    EXPECT_NEAR(C_fft[k].imag(), expected.imag(), 1e-10) << "k=" << k;
  }
}

// ── Edge cases ──────────────────────────────────────────────────────

TEST(FFT, SizeOneIsNoOp) {
  Eigen::VectorX<cd> x(1);
  x[0] = cd(7.0, -3.0);
  fft(x);
  EXPECT_DOUBLE_EQ(x[0].real(), 7.0);
  EXPECT_DOUBLE_EQ(x[0].imag(), -3.0);
  ifft(x);
  EXPECT_DOUBLE_EQ(x[0].real(), 7.0);
  EXPECT_DOUBLE_EQ(x[0].imag(), -3.0);
}

TEST(FFT, SizeTwoMatchesAnalyticForm) {
  // For N=2: X[0] = x[0] + x[1],  X[1] = x[0] - x[1].
  Eigen::VectorX<cd> x(2);
  x[0] = cd(1.0, 0.5);
  x[1] = cd(2.0, -1.5);
  Eigen::VectorX<cd> X = x;
  fft(X);
  EXPECT_NEAR(X[0].real(), 3.0, 1e-12);
  EXPECT_NEAR(X[0].imag(), -1.0, 1e-12);
  EXPECT_NEAR(X[1].real(), -1.0, 1e-12);
  EXPECT_NEAR(X[1].imag(), 2.0, 1e-12);
}

TEST(FFT, IFFTAppliesOneOverNNormalisation) {
  // For impulse δ[0] forward: all bins == 1.
  // Then ifft must yield (1, 0, 0, ..., 0) — the original impulse.
  const long N = 8;
  Eigen::VectorX<cd> X = Eigen::VectorX<cd>::Constant(N, cd(1.0, 0.0));
  ifft(X);
  EXPECT_NEAR(X[0].real(), 1.0, 1e-12);
  EXPECT_NEAR(X[0].imag(), 0.0, 1e-12);
  for (long n = 1; n < N; ++n) {
    EXPECT_NEAR(X[n].real(), 0.0, 1e-12) << "n=" << n;
    EXPECT_NEAR(X[n].imag(), 0.0, 1e-12) << "n=" << n;
  }
}
