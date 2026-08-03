/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "gradients.h"
#include <cmath>

namespace libskit {

// Sum a binary functor `op(true_i, guess_i)` over corresponding elements.
// This is the common backbone of all pointwise losses.
template <typename Op>
inline double pointwise_loss_sum(const Eigen::VectorXd &a,
                                 const Eigen::VectorXd &b, Op op) {
  double result = 0.0;
  const long n = a.size();
  for (long i = 0; i < n; ++i) {
    result += op(a[i], b[i]);
  }
  return result;
}

double l2loss(const Eigen::VectorXd &true_irradiance,
              const Eigen::VectorXd &guess_irradiance) {
  return pointwise_loss_sum(
      true_irradiance, guess_irradiance,
      [](double t, double g) { return std::pow(t - g, 2); });
}

double intensity_weighted_loss(const Eigen::VectorXd &true_irradiance,
                  const Eigen::VectorXd &guess_irradiance) {
  return pointwise_loss_sum(
      true_irradiance, guess_irradiance, [](double t, double g) {
        return std::pow(t - g, 2) / (0.05 + std::pow(t, 2));
      });
}

double l1loss(const Eigen::VectorXd &true_irradiance,
              const Eigen::VectorXd &guess_irradiance) {
  return pointwise_loss_sum(
      true_irradiance, guess_irradiance,
      [](double t, double g) { return std::abs(t - g); });
}

double log_l2loss(const Eigen::VectorXd &true_irradiance,
                  const Eigen::VectorXd &guess_irradiance) {
  return pointwise_loss_sum(
      true_irradiance, guess_irradiance, [](double t, double g) {
        const double diff = std::log(1.0 + t) - std::log(1.0 + g);
        return diff * diff;
      });
}

double weighted_l2loss(const Eigen::VectorXd &true_irradiance,
                       const Eigen::VectorXd &guess_irradiance,
                       const Eigen::VectorXd &weights) {
  // Empty weights fall back to ordinary L2.
  if (weights.size() == 0)
    return l2loss(true_irradiance, guess_irradiance);

  double result = 0.0;
  double wsum = 0.0;
  const long n = true_irradiance.size();
  for (long i = 0; i < n; i++) {
    const double w = weights[i];
    result += w * std::pow(true_irradiance[i] - guess_irradiance[i], 2);
    wsum += w;
  }
  return result / (wsum + 1e-30);
}

double optimal_scale_factor(const Eigen::VectorXd &y_sim,
                            const Eigen::VectorXd &y_ref) {
  // Analytic minimiser of the L2 loss ‖y_sim/c − y_ref‖² wrt c, i.e. the
  // scale factor c such that y_sim / c ≈ y_ref:
  //   c* = ⟨y_sim, y_sim⟩ / ⟨y_sim, y_ref⟩.
  // (This is *not* the same as the ⟨y_sim, y_ref⟩/⟨y_ref, y_ref⟩ formula,
  //  which minimises ‖y_sim − c·y_ref‖² instead.)
  double num = y_sim.dot(y_sim);
  double den = y_sim.dot(y_ref) + 1e-30;
  return num / den;
}

double loss_func(const Experiment &experiment) {

  // Full forward simulation: plasma model -> instrument model -> simulated spectrum.
  Eigen::VectorXd y_sim = experiment.instrument_ptr->calculate_spectrum(
      experiment.true_spectrum.wavelengths, *experiment.plasma_model_ptr);

  const Eigen::VectorXd &y_exp = experiment.true_spectrum.irradiance;

  Eigen::VectorXd vec1, vec2;

  switch (experiment.normalization_type) {
  case AnalyticNorm: {
    // Find c such that y_sim / c ≈ y_exp (analytic minimiser of the L2
    // loss ‖y_sim/c − y_exp‖²) and normalize against the scaled
    // experimental maximum.
    const double analytic_scale_factor = optimal_scale_factor(y_sim, y_exp);
    const double scale = analytic_scale_factor * y_exp.maxCoeff();
    vec1 = (y_sim / scale).eval();
    vec2 = (y_exp / y_exp.maxCoeff()).eval();
    break;
  }
  case OptimizedScaleNorm: {
    // Same structure as AnalyticNorm, but c is optimized explicitly.
    const double scale = experiment.scale_factor * y_exp.maxCoeff();
    vec1 = (y_sim / scale).eval();
    vec2 = (y_exp / y_exp.maxCoeff()).eval();
    break;
  }
  case MaxNorm:
  default: {
    // Legacy normalization: each spectrum gets normalized by its own maximum.
    vec1 = (y_sim / y_sim.maxCoeff()).eval();
    vec2 = (y_exp / y_exp.maxCoeff()).eval();
    break;
  }
  }

  switch (experiment.loss_type) {
  case L2Loss:
    return l2loss(vec1, vec2);
  case IntensityWeightedLoss:
    return intensity_weighted_loss(vec1, vec2);
  case L1Loss:
    return l1loss(vec1, vec2);
  case LogL2Loss:
    return log_l2loss(vec1, vec2);
  case WeightedL2Loss:
    return weighted_l2loss(vec1, vec2, experiment.loss_weights);
  default:
    return l2loss(vec1, vec2);
  }
}

Eigen::VectorXd calculate_irradiance_test(PlasmaModel &plasma_model,
                                          Eigen::VectorXd &wavelengths) {
  return plasma_model.calculate_irradiance(wavelengths);
}

} // namespace libskit
