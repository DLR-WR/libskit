/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file gradients.h
 * @brief Loss functions and Experiment struct for gradient-based optimization.
 */
#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "Instrument/Instrument.h"
#include "PlasmaModel/PlasmaModel.h"
#include "Spectrum/Spectrum.h"
#include <Eigen/Core>
#include <memory>

namespace libskit {

/**
 * @brief Enumeration of available loss function types.
 */
enum LossType {
  L2Loss,         ///< Standard L2 (least squares) loss
  IntensityWeightedLoss,      ///< Custom LIBS loss
  L1Loss,         ///< L1 loss — robust to outliers
  LogL2Loss,      ///< L2 on log(1 + spectrum)
  WeightedL2Loss, ///< Per-pixel weighted L2
  HuberLoss,      ///< Robust loss: quadratic near zero, linear beyond
                  ///< `Experiment.robust_threshold` (delta)
  TukeyBiweightLoss ///< Redescending robust loss: saturates to a constant
                    ///< beyond `Experiment.robust_threshold` (c), giving
                    ///< exactly-zero gradient contribution for residuals
                    ///< that large (full outlier rejection)
};

/**
 * @brief Enumeration of available normalization strategies.
 *
 * Controls how simulated and experimental spectra are brought to a common
 * scale before computing the loss.
 */
enum NormalizationType {
  MaxNorm,
  AnalyticNorm,
  OptimizedScaleNorm
};

/**
 * @brief Experimental configuration for forward simulation and loss evaluation.
 *
 * Bundles the plasma model, instrument model, target spectrum, and loss setup
 * used for optimization and gradient calculations.
 */
struct Experiment {
  /// @brief Type of loss function to use.
  enum LossType loss_type;
  /// @brief Normalization strategy applied before loss evaluation.
  enum NormalizationType normalization_type;

  /// @brief Plasma model being optimized.
  std::shared_ptr<PlasmaModel> plasma_model_ptr;
  /// @brief Instrument model used for forward simulation.
  std::shared_ptr<Instrument> instrument_ptr;

  /// @brief Experimental spectrum used as the fitting target.
  Spectrum true_spectrum;

  /// @brief Free scale factor for OptimizedScaleNorm.
  double scale_factor = 1.0;

  /// @brief Per-pixel loss weights (used only when loss_type == WeightedL2Loss).
  /// Empty ⇒ uniform weights.
  Eigen::VectorXd loss_weights;

  /// @brief Robust-loss threshold: the Huber "delta" or Tukey biweight "c"
  /// constant, in the same (normalized) units as the residuals computed
  /// inside the configured NormalizationType. Only used when loss_type is
  /// HuberLoss or TukeyBiweightLoss.
  double robust_threshold = 1.0;

  /**
   * @brief Construct an Experiment object.
   */
  Experiment(enum LossType _loss_type, const PlasmaModel &_plasma_model,
             const Instrument &_instrument, Spectrum _true_spectrum,
             enum NormalizationType _normalization_type = MaxNorm)
      : loss_type(_loss_type), normalization_type(_normalization_type),
        plasma_model_ptr(_plasma_model.clone()),
        instrument_ptr(_instrument.clone()),
        true_spectrum(std::move(_true_spectrum)),
        scale_factor(1.0),
        loss_weights(),
        robust_threshold(1.0) {}

  /** @brief Get a mutable reference to the plasma model. */
  PlasmaModel &get_plasma_model() { return *plasma_model_ptr; }
  /** @brief Get a mutable reference to the instrument model. */
  Instrument &get_instrument() { return *instrument_ptr; }
};

/** @brief Calculate standard L2 loss: ||true - guess||^2. */
double l2loss(const Eigen::VectorXd &true_irradiance,
              const Eigen::VectorXd &guess_irradiance);

/** @brief Calculate the custom IntensityWeighted loss used in some LIBS workflows. */
double intensity_weighted_loss(const Eigen::VectorXd &true_irradiance,
                  const Eigen::VectorXd &guess_irradiance);

/** @brief Calculate per-pixel weighted L2 loss. */
double weighted_l2loss(const Eigen::VectorXd &true_irradiance,
                       const Eigen::VectorXd &guess_irradiance,
                       const Eigen::VectorXd &weights);

/** @brief Calculate Huber loss: quadratic for |r| <= delta, linear beyond. */
double huber_loss(const Eigen::VectorXd &true_irradiance,
                  const Eigen::VectorXd &guess_irradiance, double delta);

/** @brief Calculate Tukey's biweight (redescending) loss: saturates to a
 * constant for |r| > c, giving exactly-zero gradient contribution there. */
double tukey_biweight_loss(const Eigen::VectorXd &true_irradiance,
                          const Eigen::VectorXd &guess_irradiance, double c);

/**
 * @brief Compute the analytically optimal scale factor between two spectra.
 */
double optimal_scale_factor(const Eigen::VectorXd &y_sim,
                            const Eigen::VectorXd &y_ref);

/**
 * @brief Run the full forward model and evaluate the configured loss.
 */
double loss_func(const Experiment &experiment);

/**
 * @brief Test helper for direct irradiance calculation.
 */
Eigen::VectorXd calculate_irradiance_test(PlasmaModel &plasma_model,
                                          Eigen::VectorXd &wavelengths);

} // namespace libskit

#endif // !GRADIENTS_H
