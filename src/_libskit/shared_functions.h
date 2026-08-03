/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file shared_functions.h
 * @brief Pure-numeric template functions shared between forward-mode AD
 *        (autodiff::real) and reverse-mode AD (Enzyme).
 *
 * These functions contain NO string operations, NO map lookups by string key,
 * and NO references to global data tables (LEVELS, LINES, IONIZATION_ENERGIES,
 * ATOM_INFO).  All such lookups must be performed by the caller, which passes
 * in pre-resolved numeric data.
 *
 * Because Enzyme operates on LLVM IR and needs to see function bodies in the
 * translation unit, all functions are defined inline in this header.
 */

#ifndef SHARED_FUNCTIONS_H
#define SHARED_FUNCTIONS_H

#include "AtomicData/constants.h"
#include <cmath>
#include <numbers>

namespace libskit {

// =====================================================================
// De Broglie wavelength  (numeric-only, no string ops)
// =====================================================================

/**
 * @brief Thermal de Broglie wavelength of the electron.
 * Λ = sqrt(2π ħ² / (m_e k_B T))
 */
template <typename T>
inline T debroglie_wavelength(T temperature) {
  return sqrt((2.0 * std::numbers::pi * HBAR * HBAR) /
              (ELECTRON_MASS * BOLTZMANN_IN_J_K * temperature));
}

// =====================================================================
// Partition function sum  (numeric-only: caller provides arrays)
// =====================================================================

/**
 * @brief Compute the partition function Z(T) = Σ g_i exp(-E_i / kT)
 *        for energy levels below the ionization energy.
 *
 * @param g_values    Statistical weights (degeneracies).
 * @param energies    Energy levels in eV.
 * @param n_levels    Number of levels.
 * @param ionization_energy  Ionization energy in eV (levels above are skipped).
 * @param temperature Temperature in Kelvin.
 */
template <typename T>
inline T partition_function_sum(const double *g_values,
                                const double *energies,
                                int n_levels,
                                double ionization_energy,
                                T temperature) {
  T sum = 0.0;
  for (int i = 0; i < n_levels; i++) {
    if (energies[i] >= ionization_energy)
      break;
    sum += g_values[i] * exp(-energies[i] / (BOLTZMANN_IN_EV_K * temperature));
  }
  return sum;
}

// =====================================================================
// Saha ratio  (numeric-only: caller provides partition functions)
// =====================================================================

/**
 * @brief Compute the Saha ratio  ω = (2 / Λ³) · (Z_upper / Z_lower) · exp(-E_ion / kT)
 *
 * @param Z_lower     Partition function of the lower ionization stage.
 * @param Z_upper     Partition function of the upper ionization stage.
 * @param ionization_energy  Ionization energy of the lower stage (eV).
 * @param temperature Temperature in Kelvin.
 */
template <typename T>
inline T saha_ratio(T Z_lower, T Z_upper,
                    double ionization_energy,
                    T temperature) {
  T lambda = debroglie_wavelength(temperature);
  return 2.0 / (lambda * lambda * lambda) *
         Z_upper / Z_lower *
         exp(-ionization_energy / (BOLTZMANN_IN_EV_K * temperature));
}

// =====================================================================
// Doppler standard deviation  (numeric-only: caller provides atomic weight)
// =====================================================================

/**
 * @brief Doppler broadening standard deviation σ_D = λ · sqrt(k_B T / (M c²))
 *
 * @param atomic_weight  Atomic weight in atomic mass units (u).
 * @param wavelength     Line center wavelength in nm.
 * @param temperature    Temperature in Kelvin.
 */
template <typename T>
inline T doppler_sigma(T temperature, double atomic_weight, double wavelength) {
  return sqrt(BOLTZMANN_IN_J_K * temperature /
              (atomic_weight * ATOMIC_MASS * SPEED_OF_LIGHT *
               SPEED_OF_LIGHT)) *
         wavelength;
}

} // namespace libskit

#endif // SHARED_FUNCTIONS_H
