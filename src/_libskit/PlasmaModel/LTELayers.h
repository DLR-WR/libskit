/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef LTELAYERS_H
#define LTELAYERS_H

#include "PlasmaModel/LTEModel.h"
#include <vector>
namespace libskit {

/**
 * @brief Multi-layer LTE plasma model with independent composition per layer.
 *
 * Each layer is a full LTEModel with its own elemental composition,
 * temperature, electron density, and length. The total irradiance is built by
 * propagating through the layers in sequence.
 */
struct LTELayers : public PlasmaModel {

  std::vector<LTEModel> lte_layers;

  /**
   * @brief Construct a stack of LTE layers.
   */
  LTELayers(std::vector<std::map<std::string, double>> _element_ratios,
            std::vector<double> _temperatures,
            std::vector<double> _electron_densities,
            std::vector<double> _lengths);

  /** @brief Create an exact copy of the layered model. */
  std::shared_ptr<PlasmaModel> clone() const override;

  /** @brief Access a layer parameter by name. */
  double &get_parameter(const std::string &parameter_name) override;

  /** @brief Set a layer parameter by name. */
  void set_parameter(const std::string &parameter_name, double value) override;

  /** @brief List all exposed layer parameter names. */
  std::vector<std::string> get_parameter_names() const override;

  /** @brief Calculate irradiance for the full layered plasma. */
  Eigen::VectorXd
  calculate_irradiance(const Eigen::VectorXd &wavelengths) const override;

  /** @brief Return the narrowest contributing line width across all layers. */
  double
  get_minimum_voigt_fwhm(const Eigen::VectorXd &wavelengths) const override;
};

} // namespace libskit

#endif // LTELAYERS_H
