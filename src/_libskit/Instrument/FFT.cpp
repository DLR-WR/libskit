/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "FFT.h"
#include <numbers>

// Recursive Cooley-Tukey FFT using the same sign convention as the
// Enzyme-side iterative implementation.
void fft(Eigen::VectorXcd &a) {
  long int n = a.size();
  if (n <= 1)
    return;

  Eigen::VectorXcd even(n / 2);
  Eigen::VectorXcd odd(n / 2);

  for (int i = 0; i < n / 2; i++) {
    even[i] = a[2 * i];
    odd[i] = a[2 * i + 1];
  }
  fft(even);
  fft(odd);

  double ang = 2 * std::numbers::pi / n;
  std::complex<double> w(1), wn(cos(ang), sin(ang));
  for (int i = 0; i < n / 2; i++) {
    a[i] = even[i] + w * odd[i];
    a[i + n / 2] = even[i] - w * odd[i];
    w *= wn;
  }
}

// Matching inverse transform: negative angle convention and divide by two at
// each merge stage.
void ifft(Eigen::VectorXcd &a) {
  long int n = a.size();
  if (n <= 1)
    return;

  Eigen::VectorXcd even(n / 2);
  Eigen::VectorXcd odd(n / 2);

  for (int i = 0; i < n / 2; i++) {
    even[i] = a[2 * i];
    odd[i] = a[2 * i + 1];
  }
  ifft(even);
  ifft(odd);

  double ang = -2 * std::numbers::pi / n;
  std::complex<double> w(1), wn(cos(ang), sin(ang));
  for (int i = 0; i < n / 2; i++) {
    a[i] = (even[i] + w * odd[i]) / 2.0;
    a[i + n / 2] = (even[i] - w * odd[i]) / 2.0;
    w *= wn;
  }
}

