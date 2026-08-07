/*
 * Copyright (c) 2026 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 *
 * physics_kernel.h
 *
 * Shared, double-only, allocation-light implementations of the LTE plasma
 * physics primitives + the line-broadening + RTE kernels.
 *
 * These functions are the single source of truth used by both the high-level
 * PlasmaModel classes (LTEModel, LTELayers, LTESharedLayers, ...) and the
 * Enzyme reverse-mode gradient path in enzyme_gradients.cpp.
 *
 * Everything is `inline` so Enzyme can see the bodies through any TU that
 * includes this header.
 */
#ifndef LIBSKIT_PHYSICS_KERNEL_H
#define LIBSKIT_PHYSICS_KERNEL_H

#include "AtomicData/atom_info.h"
#include "AtomicData/constants.h"
#include "AtomicData/elemental_data.h"
#include "utils/to_roman.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <string>
#include <vector>

namespace libskit::physics {

// =====================================================================
// Flat, plain-data representation of all the per-element / per-line
// information the kernel needs.  Built once from string-keyed maps
// (ATOM_INFO / LEVELS / LINES / IONIZATION_ENERGIES) and then handed
// to the kernel by pointer.
// =====================================================================

struct IonLevels {
  int n_levels = 0;
  double ionization_energy = 1e30;
  std::vector<double> g_values;
  std::vector<double> energies;
};

struct IonLineData {
  int element_idx;
  unsigned int ion_level;
  double atomic_weight;
  std::vector<EmissionLine> lines;
};

struct PlasmaData {
  int n_elements = 0;

  // Per-element
  std::vector<double> atomic_weights;
  std::vector<unsigned int> max_ions;
  int max_max_ions = 0;

  // Ion-level data: ion_levels_flat[element_offset[e] + ion_level]
  std::vector<IonLevels> ion_levels_flat;
  std::vector<int> element_offset;
  std::vector<double> saha_ion_energies;

  // Emission lines, pre-gathered by (element, ion_level)
  std::vector<IonLineData> ion_line_groups;

  // Line-filtering threshold
  double accuracy = 1e-3;
};

// =====================================================================
// Build PlasmaData from element symbols using the global LEVELS / LINES /
// IONIZATION_ENERGIES / ATOM_INFO maps.
// =====================================================================
inline PlasmaData build_data_from_elements(
    const std::vector<std::string> &element_names, double accuracy) {
  PlasmaData plasma_data;
  plasma_data.n_elements = static_cast<int>(element_names.size());
  plasma_data.accuracy = accuracy;

  plasma_data.atomic_weights.resize(plasma_data.n_elements);
  plasma_data.max_ions.resize(plasma_data.n_elements);
  plasma_data.element_offset.resize(plasma_data.n_elements);

  int total_ionization_slots = 0;
  unsigned int maximum_ion_count = 0;
  for (int element_index = 0; element_index < plasma_data.n_elements;
       element_index++) {
    plasma_data.atomic_weights[element_index] =
        ATOM_INFO.at(element_names[element_index]).atomic_weight;
    plasma_data.max_ions[element_index] =
        get_maximum_ionization_level(element_names[element_index]);
    if (plasma_data.max_ions[element_index] > maximum_ion_count)
      maximum_ion_count = plasma_data.max_ions[element_index];
    plasma_data.element_offset[element_index] = total_ionization_slots;
    total_ionization_slots += plasma_data.max_ions[element_index] + 1;
  }
  plasma_data.max_max_ions = static_cast<int>(maximum_ion_count);

  plasma_data.ion_levels_flat.resize(total_ionization_slots);
  plasma_data.saha_ion_energies.assign(total_ionization_slots, 1e30);

  for (int element_index = 0; element_index < plasma_data.n_elements;
       element_index++) {
    int element_offset = plasma_data.element_offset[element_index];
    for (unsigned int ionization_level = 0;
         ionization_level <= plasma_data.max_ions[element_index];
         ionization_level++) {
      std::string ion_name =
          element_names[element_index] + " " + to_roman(ionization_level + 1);
      IonLevels &ion_levels =
          plasma_data.ion_levels_flat[element_offset + ionization_level];

      if (LEVELS.count(ion_name) && IONIZATION_ENERGIES.count(ion_name)) {
        const auto &level_data = LEVELS.at(ion_name);
        ion_levels.ionization_energy = IONIZATION_ENERGIES.at(ion_name);
        ion_levels.n_levels = static_cast<int>(level_data.size());
        ion_levels.g_values.resize(ion_levels.n_levels);
        ion_levels.energies.resize(ion_levels.n_levels);
        for (int level_index = 0; level_index < ion_levels.n_levels;
             level_index++) {
          ion_levels.g_values[level_index] = level_data[level_index].g;
          ion_levels.energies[level_index] = level_data[level_index].energy;
        }
      }

      if (ionization_level < plasma_data.max_ions[element_index]) {
        plasma_data.saha_ion_energies[element_offset + ionization_level] =
            IONIZATION_ENERGIES.count(ion_name)
                ? IONIZATION_ENERGIES.at(ion_name)
                : 1e30;
      }
    }
  }

  for (int element_index = 0; element_index < plasma_data.n_elements;
       element_index++) {
    for (unsigned int ionization_level = 0;
         ionization_level < plasma_data.max_ions[element_index];
         ionization_level++) {
      std::string ion_name =
          element_names[element_index] + " " + to_roman(ionization_level + 1);
      if (!LINES.count(ion_name))
        continue;
      IonLineData ion_line_group;
      ion_line_group.element_idx = element_index;
      ion_line_group.ion_level = ionization_level;
      ion_line_group.atomic_weight = plasma_data.atomic_weights[element_index];
      ion_line_group.lines = LINES.at(ion_name);
      plasma_data.ion_line_groups.push_back(std::move(ion_line_group));
    }
  }
  return plasma_data;
}

// =====================================================================
// Inline scalar physics primitives.
// =====================================================================

inline double voigt(double x, double sigma, double gamma) {
  constexpr double A[4] = {-1.2150, -1.3509, -1.2150, -1.3509};
  constexpr double B[4] = {1.2359, 0.3786, -1.2359, -0.3786};
  constexpr double C[4] = {-0.3085, 0.5906, -0.3085, 0.5906};
  constexpr double D[4] = {0.0210, -1.1858, -0.0210, 1.1858};

  const double inv_sqrt2_sigma = 1.0 / (std::numbers::sqrt2 * sigma);
  const double X = x * inv_sqrt2_sigma;
  const double Y = gamma * inv_sqrt2_sigma;

  double result = 0.0;
  for (int i = 0; i < 4; i++) {
    const double dy = Y - A[i];
    const double dx = X - B[i];
    result += (C[i] * dy + D[i] * dx) / (dy * dy + dx * dx);
  }
  return (std::numbers::inv_sqrtpi / std::numbers::sqrt2 / sigma) * result;
}

inline double partition_function(const IonLevels &il, double T) {
  double sum = 0.0;
  for (int i = 0; i < il.n_levels; i++) {
    if (il.energies[i] >= il.ionization_energy)
      continue;
    sum += il.g_values[i] * std::exp(-il.energies[i] / (BOLTZMANN_IN_EV_K * T));
  }
  return sum;
}

inline double debroglie(double T) {
  return std::sqrt((2.0 * std::numbers::pi * HBAR * HBAR) /
                   (ELECTRON_MASS * BOLTZMANN_IN_J_K * T));
}

inline double saha(const IonLevels &lower, const IonLevels &upper,
                   double ion_energy, double T) {
  return 2.0 / std::pow(debroglie(T), 3) * partition_function(upper, T) /
         partition_function(lower, T) *
         std::exp(-ion_energy / (BOLTZMANN_IN_EV_K * T));
}

// population_out must have space for max_ionization_level elements.
inline void population_levels(const PlasmaData *data, int element_index,
                              double T, double electron_density,
                              double *population_out,
                              int max_ionization_level) {
  int element_offset = data->element_offset[element_index];
  std::vector<double> omega_products(max_ionization_level + 1);
  omega_products[0] = 1.0;
  double denominator = 1.0;

  for (int ionization_level = 0; ionization_level < max_ionization_level;
       ionization_level++) {
    double saha_ratio =
        saha(data->ion_levels_flat[element_offset + ionization_level],
             data->ion_levels_flat[element_offset + ionization_level + 1],
             data->saha_ion_energies[element_offset + ionization_level], T) /
        electron_density;
    omega_products[ionization_level + 1] =
        saha_ratio * omega_products[ionization_level];
    denominator += omega_products[ionization_level + 1];
  }
  for (int ionization_level = 0; ionization_level < max_ionization_level;
       ionization_level++) {
    population_out[ionization_level] =
        omega_products[ionization_level] / denominator;
  }
}

inline double doppler_sigma(double atomic_weight, double wavelength, double T) {
  return std::sqrt(BOLTZMANN_IN_J_K * T /
                   (atomic_weight * ATOMIC_MASS * SPEED_OF_LIGHT *
                    SPEED_OF_LIGHT)) *
         wavelength;
}

inline double stark_fwhm(const EmissionLine &line, double ne, double T) {
  // exp(x*ln10) instead of pow(10,x): avoids llvm.exp10 intrinsic, which
  // Enzyme v0.0.256 cannot differentiate on newer LLVMs (e.g. LLVM 22).
  const double logT = std::log10(T);
  double w = std::exp((line.a0 + line.a1 * logT + line.a2 * logT * logT) *
                      std::numbers::ln10) /
             10.0;
  return w * ne / line.n_e_ref * 1e-6;
}

inline double stark_shift(const EmissionLine &line, double T) {
  const double logT = std::log10(T);
  double w = std::exp((line.a0 + line.a1 * logT + line.a2 * logT * logT) *
                      std::numbers::ln10);
  return w *
         (line.b0 + line.b1 * logT +
          line.b2 * logT * logT) /
         10.0;
}

inline double blackbody(double wavelength_nm, double T) {
  double wl = wavelength_nm * 1e-9;
  return 2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT * SPEED_OF_LIGHT /
         (wl * wl * wl * wl * wl) * 1.0 /
         (std::exp(PLANCK_CONSTANT * SPEED_OF_LIGHT /
                   (wl * BOLTZMANN_IN_J_K * T)) -
          1.0);
}

// =====================================================================
// Line-broadening kernel: compute LTE emission coefficient ε(λ) on the
// supplied wavelength grid.  Two-pass algorithm: first pass finds the
// max irradiance peak (used as a threshold scale), second pass adds
// filtered, broadened lines onto the grid.
//
// Continuum (free-free + free-bound) is NOT included — callers that
// want it (LTEModel with include_continuum) must add it separately.
// =====================================================================
// `aki_overrides` (optional, may be nullptr): flat array containing one A_ji
// value per line in the global line ordering — same ordering as iterating
// over `data->ion_line_groups` in order and then `.lines` in order. When
// non-null it shadows `line.A_ji`; this is the hook the Enzyme AD path uses
// to make Einstein coefficients first-class differentiable inputs.
// `stark_overrides` (optional, may be nullptr): flat array of per-line
// multipliers applied to the Stark FWHM (Lorentzian width in the Voigt).
// Same ordering as `aki_overrides`. When non-null, replaces `line.stark_mult`
// for the corresponding line. Used by the Enzyme AD path to make the Stark
// widths first-class differentiable inputs.
inline void compute_emission_coeff(double temperature, double electron_density,
                                   double length, const double *element_ratios,
                                   int n_elements, const PlasmaData *data,
                                   const double *wl, long n_wl,
                                   std::vector<double> &emission_coeff,
                                   const double *aki_overrides = nullptr,
                                   const double *stark_overrides = nullptr) {
  emission_coeff.assign(n_wl, 0.0);

  const int max_ionization_slots = data->max_max_ions;
  std::vector<double> population_by_element(
      n_elements * (max_ionization_slots + 1), 0.0);
  for (int element_index = 0; element_index < n_elements; element_index++) {
    population_levels(data, element_index, temperature, electron_density,
                      &population_by_element[element_index *
                                             (max_ionization_slots + 1)],
                      static_cast<int>(data->max_ions[element_index]));
  }

  double electron_balance_sum = 0.0;
  for (int element_index = 0; element_index < n_elements; element_index++) {
    double ionized_population_sum = 0.0;
    for (unsigned int ionization_level = 0;
         ionization_level < data->max_ions[element_index] - 1;
         ionization_level++) {
      ionized_population_sum +=
          (ionization_level + 1) *
          population_by_element[element_index * (max_ionization_slots + 1) +
                                ionization_level + 1];
    }
    electron_balance_sum +=
        element_ratios[element_index] / data->atomic_weights[element_index] *
        ionized_population_sum;
  }

  const double accuracy = data->accuracy;
  double max_irradiance_peak = 0.0;

  // First pass: find max_irradiance_peak.
  // group_line_offset tracks the global index of the first line in the
  // current ion-line group so we can index `aki_overrides` correctly even
  // when the inner loop `break`s out of an unfinished group.
  size_t group_line_offset = 0;
    for (size_t line_group_index = 0;
       line_group_index < data->ion_line_groups.size(); line_group_index++) {
    const IonLineData &ion_line_group = data->ion_line_groups[line_group_index];
    const int element_index = ion_line_group.element_idx;
    const double element_ratio = element_ratios[element_index];
    const double element_number_density =
      electron_density * element_ratio /
      data->atomic_weights[element_index] / electron_balance_sum;

    int element_offset = data->element_offset[element_index];
    const double partition_value = partition_function(
      data->ion_levels_flat[element_offset + ion_line_group.ion_level],
      temperature);

    for (size_t line_index = 0; line_index < ion_line_group.lines.size();
       line_index++) {
      const EmissionLine &line = ion_line_group.lines[line_index];
      const double A_ji_eff = aki_overrides
                                  ? aki_overrides[group_line_offset + line_index]
                                  : line.A_ji;
      const double stark_mult_eff =
          stark_overrides ? stark_overrides[group_line_offset + line_index]
                          : line.stark_mult;
      if (line.wavelength < wl[0])
        continue;
      if (line.wavelength > wl[n_wl - 1])
        break;

      const double excitation_fraction =
        line.g_j / partition_value *
          std::exp(-line.E_j / (BOLTZMANN_IN_EV_K * temperature));
      const double excited_state_density =
        element_number_density *
        population_by_element[element_index * (max_ionization_slots + 1) +
                  ion_line_group.ion_level] *
        excitation_fraction;
      const double emission =
          ELEMENTARY_CHARGE / (4.0 * std::numbers::pi) *
        (line.E_j - line.E_i) * A_ji_eff * excited_state_density;

      const double doppler_sigma_value =
        doppler_sigma(ion_line_group.atomic_weight, line.wavelength,
              temperature);
      const double stark_gamma_value =
          0.5 * stark_fwhm(line, electron_density, temperature) * stark_mult_eff;
      const double peak_emission_coeff =
        emission * 1.0e9 * voigt(0.0, doppler_sigma_value, stark_gamma_value);

      const double blackbody_intensity = blackbody(line.wavelength, temperature);
      const double peak_irradiance =
        blackbody_intensity *
        (1.0 - std::exp(-peak_emission_coeff / blackbody_intensity * length));

      if (peak_irradiance > max_irradiance_peak)
      max_irradiance_peak = peak_irradiance;
    }
    group_line_offset += ion_line_group.lines.size();
  }

  // Second pass: broaden filtered lines onto the grid.
  const double wing_threshold = (max_irradiance_peak > 0.0)
                                    ? accuracy * max_irradiance_peak / length
                                    : 0.0;
  group_line_offset = 0;

    for (size_t line_group_index = 0;
       line_group_index < data->ion_line_groups.size(); line_group_index++) {
    const IonLineData &ion_line_group = data->ion_line_groups[line_group_index];
    const int element_index = ion_line_group.element_idx;
    const double element_ratio = element_ratios[element_index];
    const double element_number_density =
      electron_density * element_ratio /
      data->atomic_weights[element_index] / electron_balance_sum;

    int element_offset = data->element_offset[element_index];
    const double partition_value = partition_function(
      data->ion_levels_flat[element_offset + ion_line_group.ion_level],
      temperature);

    for (size_t line_index = 0; line_index < ion_line_group.lines.size();
       line_index++) {
      const EmissionLine &line = ion_line_group.lines[line_index];
      const double A_ji_eff = aki_overrides
                                  ? aki_overrides[group_line_offset + line_index]
                                  : line.A_ji;
      const double stark_mult_eff =
          stark_overrides ? stark_overrides[group_line_offset + line_index]
                          : line.stark_mult;
      if (line.wavelength < wl[0])
        continue;
      if (line.wavelength > wl[n_wl - 1])
        break;

      const double excitation_fraction =
        line.g_j / partition_value *
          std::exp(-line.E_j / (BOLTZMANN_IN_EV_K * temperature));
      const double excited_state_density =
        element_number_density *
        population_by_element[element_index * (max_ionization_slots + 1) +
                  ion_line_group.ion_level] *
        excitation_fraction;
      const double emission =
          ELEMENTARY_CHARGE / (4.0 * std::numbers::pi) *
        (line.E_j - line.E_i) * A_ji_eff * excited_state_density;

      const double doppler_sigma_value =
        doppler_sigma(ion_line_group.atomic_weight, line.wavelength,
              temperature);
      const double stark_gamma_value =
          0.5 * stark_fwhm(line, electron_density, temperature) * stark_mult_eff;

      const double peak_emission_coeff =
        emission * 1.0e9 * voigt(0.0, doppler_sigma_value, stark_gamma_value);
      const double blackbody_intensity = blackbody(line.wavelength, temperature);
      const double peak_irradiance =
        blackbody_intensity *
        (1.0 - std::exp(-peak_emission_coeff / blackbody_intensity * length));
      if (peak_irradiance < accuracy * max_irradiance_peak)
        continue;

      const double stark_shift_value = stark_shift(line, temperature);
      const double line_center = line.wavelength + stark_shift_value;

      auto it = std::upper_bound(wl, wl + n_wl, line.wavelength);
      long index = it - wl;

      for (long i = index; i > 0; i--) {
        double e_line =
            emission * 1.0e9 *
            voigt(wl[i] - line_center, doppler_sigma_value, stark_gamma_value);
        if (e_line > wing_threshold)
          emission_coeff[i] += e_line;
        else
          break;
      }
      for (long i = index + 1; i < n_wl; i++) {
        double e_line =
            emission * 1.0e9 *
            voigt(wl[i] - line_center, doppler_sigma_value, stark_gamma_value);
        if (e_line > wing_threshold)
          emission_coeff[i] += e_line;
        else
          break;
      }
    }
    group_line_offset += ion_line_group.lines.size();
  }
}

// =====================================================================
// Single-layer RTE: I(λ) = B(λ,T) * (1 - exp(-ε(λ)/B * L))
// =====================================================================
inline void apply_rte_single(const double *wl, long n_wl,
                             const std::vector<double> &emission_coeff,
                             double temperature, double length,
                             std::vector<double> &irradiance) {
  irradiance.resize(n_wl);
  for (long i = 0; i < n_wl; i++) {
    const double blackbody_intensity = blackbody(wl[i], temperature);
    irradiance[i] =
        blackbody_intensity *
        (1.0 - std::exp(-emission_coeff[i] / blackbody_intensity * length));
  }
}

// =====================================================================
// Multilayer RTE update (in-place):
//   I[i] = I[i] * G + B * (1 - G),  G = exp(-ε[i]/B * L)
// Pass the existing accumulated irradiance in `total_irradiance`; this
// applies one layer's contribution on top of it.
// =====================================================================
inline void apply_rte_multilayer(const double *wl, long n_wl,
                                 const std::vector<double> &emission_coeff,
                                 double temperature, double length,
                                 std::vector<double> &total_irradiance) {
  for (long i = 0; i < n_wl; i++) {
                                  const double blackbody_intensity = blackbody(wl[i], temperature);
                                  const double transmission_factor =
                                      std::exp(-emission_coeff[i] / blackbody_intensity * length);
                                  total_irradiance[i] = total_irradiance[i] * transmission_factor +
                                                        blackbody_intensity * (1.0 - transmission_factor);
  }
}

// =====================================================================
// Convenience: single-layer LTE irradiance in one call.
// =====================================================================
inline void compute_irradiance_lte(double temperature, double electron_density,
                                   double length, const double *element_ratios,
                                   int n_elements, const PlasmaData *data,
                                   const double *wl, long n_wl,
                                   std::vector<double> &irradiance,
                                   const double *aki_overrides = nullptr,
                                   const double *stark_overrides = nullptr) {
  std::vector<double> emission_coeff;
  compute_emission_coeff(temperature, electron_density, length, element_ratios,
                         n_elements, data, wl, n_wl, emission_coeff,
                         aki_overrides, stark_overrides);
  apply_rte_single(wl, n_wl, emission_coeff, temperature, length, irradiance);
}

} // namespace libskit::physics

#endif // LIBSKIT_PHYSICS_KERNEL_H
