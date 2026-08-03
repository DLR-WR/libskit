/*
 * Copyright (c) 2026 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef LTESHAREDLAYERS_H
#define LTESHAREDLAYERS_H

#include "PlasmaModel/LTEModel.h"
#include <map>
#include <string>
#include <vector>

namespace libskit {

/**
 * @brief Multi-layer LTE plasma model with one shared elemental composition.
 *
 * Like ``LTELayers``, each layer has its own temperature, electron density,
 * and length, but the elemental mass-ratio composition is shared across all
 * layers. This is the natural object for inverse problems where the only
 * spatially varying quantities are the plasma state.
 *
 * Parameter names exposed via ``get_parameter_names()`` are:
 *   ``layer_0_temperature``, ``layer_0_electron_density``, ``layer_0_length``,
 *   ``layer_1_temperature``, ..., followed by the unprefixed element symbols
 *   (e.g. ``"Fe"``, ``"Si"``).
 */
struct LTESharedLayers : public PlasmaModel {

  std::map<std::string, double> element_ratio;
  std::vector<double> temperatures;
  std::vector<double> electron_densities;
  std::vector<double> lengths;

  LTESharedLayers(std::map<std::string, double> _element_ratio,
                  std::vector<double> _temperatures,
                  std::vector<double> _electron_densities,
                  std::vector<double> _lengths);

  /** @brief Create an exact copy of the shared-layer model. */
  std::shared_ptr<PlasmaModel> clone() const override;

  /** @brief Access a parameter by name. */
  double &get_parameter(const std::string &parameter_name) override;

  /** @brief Set a parameter by name. */
  void set_parameter(const std::string &parameter_name, double value) override;

  /** @brief List all exposed layer and composition parameters. */
  std::vector<std::string> get_parameter_names() const override;

  /** @brief Calculate irradiance for the full shared-composition layer stack. */
  Eigen::VectorXd
  calculate_irradiance(const Eigen::VectorXd &wavelengths) const override;

  /** @brief Return the narrowest contributing line width across all layers. */
  double
  get_minimum_voigt_fwhm(const Eigen::VectorXd &wavelengths) const override;

  /** @brief Number of layers in the stack. */
  size_t n_layers() const { return temperatures.size(); }
};

} // namespace libskit

#endif // LTESHAREDLAYERS_H
