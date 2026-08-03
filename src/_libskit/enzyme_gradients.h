/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file enzyme_gradients.h
 * @brief Enzyme reverse-mode AD gradient calculation.
 *
 * Provides get_gradients() which computes all gradients of the loss
 * function in a single reverse-mode pass using Enzyme.  This is O(1) in the
 * number of parameters (compared to the forward-mode get_gradients() which
 * is O(N)).
 *
 * Requires compilation with ClangEnzyme plugin and flags:
 *   -fplugin=ClangEnzyme-18.so -O2
 *   -mllvm -enzyme-globals-default-inactive=1
 */
#ifndef ENZYME_GRADIENTS_H
#define ENZYME_GRADIENTS_H

#include "gradients.h"
#include <Eigen/Core>
#include <map>
#include <string>

namespace libskit {

/**
 * @brief Calculate gradients using Enzyme reverse-mode AD.
 *
 * Computes all gradients of the loss function w.r.t. plasma model
 * parameters in a single reverse-mode pass.  The result is identical
 * to get_gradients() but computed more efficiently for many parameters.
 *
 * @param experiment_double Experimental configuration (double precision).
 * @return Map of parameter names to their gradient values.
 */
std::map<std::string, double>
get_gradients(const Experiment &experiment_double);

/**
 * @brief Compute loss using the same numerical pipeline as the Enzyme path.
 *
 * Useful for verifying that the reimplemented pipeline matches loss_func<double>.
 *
 * @param experiment_double Experimental configuration (double precision).
 * @return Loss value.
 */
double compute_loss(const Experiment &experiment_double);

/**
 * @brief Jacobian of the simulated spectrum w.r.t. plasma parameters.
 *
 * Computes the matrix J where J(i,j) = dI(λ_i) / d(param_j) using
 * n_params forward-mode AD passes (one column per parameter).
 * Cost is O(n_params × cost_of_one_forward_pass), which is much faster
 * than a reverse-mode row-loop whenever n_wavelengths >> n_params
 * (the typical LIBS case).
 *
 * @param experiment_double Experimental configuration.
 * @return Matrix of shape (n_wavelengths × n_params).
 */
Eigen::MatrixXd
get_jacobian(const Experiment &experiment_double);

/**
 * @brief Reverse-mode gradient of the loss w.r.t. the Einstein A coefficients.
 *
 * Returns a dense vector indexed in build_const_data's iteration order:
 * walk `ion_line_groups` in order, then `.lines` in order, appending one
 * entry per emission line (including those outside the active grid).
 * This is the same ordering accepted by `compute_emission_coeff`'s
 * `aki_overrides` parameter.
 *
 * @param experiment_double Experimental configuration (double precision).
 * @return Vector ∂L/∂A_ji of length sum_g |g.lines|.
 */
Eigen::VectorXd
get_aki_gradient(const Experiment &experiment_double);

/**
 * @brief Vector-Jacobian product with the Aki Jacobian: (J_A)^T @ adjoint.
 *
 * J_A(i, l) = ∂y_sim_i / ∂A_l where y_sim has length n_out and A has
 * length n_aki. One Enzyme reverse pass computes the contraction
 * Σ_i adjoint[i] · ∂y_sim_i/∂A_l for every l.
 *
 * @param experiment_double Experimental configuration.
 * @param adjoint Adjoint covector of length n_out.
 * @return (J_A)^T @ adjoint, length n_aki.
 */
Eigen::VectorXd
spectrum_vjp_aki(const Experiment &experiment_double,
                 const Eigen::VectorXd &adjoint);

/**
 * @brief Reverse-mode gradient of the loss w.r.t. the per-line Stark FWHM
 * multipliers.
 *
 * Same ordering as `get_aki_gradient`.  Values default to 1.0 when
 * no `set_stark_multipliers` override is active.
 *
 * @param experiment_double Experimental configuration.
 * @return Vector ∂L/∂m_l of length n_aki.
 */
Eigen::VectorXd
get_stark_gradient(const Experiment &experiment_double);

/**
 * @brief Vector-Jacobian product with the Stark-multiplier Jacobian:
 *        (J_m)^T @ adjoint  where  J_m(i, l) = ∂y_sim_i / ∂m_l.
 *
 * @param experiment_double Experimental configuration.
 * @param adjoint Adjoint covector of length n_out.
 * @return (J_m)^T @ adjoint, length n_aki.
 */
Eigen::VectorXd
spectrum_vjp_stark(const Experiment &experiment_double,
                   const Eigen::VectorXd &adjoint);

} // namespace libskit

#endif // ENZYME_GRADIENTS_H
