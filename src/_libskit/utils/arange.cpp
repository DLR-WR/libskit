/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "utils/arange.h"

#include <vector>

namespace libskit {

template <typename T> std::vector<T> arange(T start, T stop, T step) {
  std::vector<T> result;

  T value = start;
  result.push_back(value);

  while (value <= stop) {
    value += step;
    result.push_back(value);
  }

  return result;
}
template std::vector<double> arange(double start, double stop, double step);

Eigen::VectorXd arange_d(double start, double stop, double step) {

  return Eigen::VectorXd::LinSpaced((int)((stop - start) / step), start, stop);
}

} // namespace libskit
