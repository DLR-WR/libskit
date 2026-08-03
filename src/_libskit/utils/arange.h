/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef ARANGE_H
#define ARANGE_H

#include <vector>

#include <Eigen/Core>

namespace libskit {

/**
 * @brief Generate a range of double values
 *
 * @param start starting value
 * @param stop stop value
 * @param step step value
 * @return std::vector<double> range of values
 */
template <typename T> std::vector<T> arange(T start, T stop, T step);

/**
 * @brief Generate an Eigen-compatible range of values.
 *
 * Similar to arange() but returns an Eigen::VectorXd for code that operates on
 * the library's standard vector representation.
 *
 * @param start Starting value.
 * @param stop Stop value (exclusive).
 * @param step Step size between consecutive values.
 * @return Eigen::VectorXd Eigen vector of linearly spaced values.
 */
Eigen::VectorXd arange_d(double start, double stop, double step);

} // namespace libskit

#endif /* ARANGE_H */
