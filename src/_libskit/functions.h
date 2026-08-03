/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <array>
#include <Eigen/Core>
#include <complex>
#include <map>
#include <numbers>
#include <vector>

#include "AtomicData/atom_info.h"
#include "AtomicData/constants.h"
#include "AtomicData/elemental_data.h"
#include "utils/to_roman.h"

namespace libskit {

template <typename T>
using EmissionData =
    std::map<std::string, std::vector<std::pair<EmissionLine, T>>>;

/**
 * @brief Calculate the De Broglie wavelength for a given temperature.
 *
 * @param temperature The temperature in Kelvin.
 * @return The De Broglie wavelength in meters.
 */
template <typename T>
constexpr T calculate_debroglie_wavelength(const T temperature) {
  return sqrt((2.0 * std::numbers::pi * pow(HBAR, 2)) /
              (ELECTRON_MASS * BOLTZMANN_IN_J_K * temperature));
}

/**
 * @brief Calculate the partition function for a given ion and temperature.
 *
 * @param ion The ion to calculate the partition function for, e.g. "O II".
 * @param temperature The temperature in Kelvin.
 * @return The partition function.
 */
template <typename T>
constexpr T calculate_partition_function(const std::string &ion,
                                         const T temperature) {
  // get the ionization energy of the ion
  const double ionization_energy = IONIZATION_ENERGIES.at(ion);

  // for every energy level of the ion...
  T sum = 0.0;
  for (const auto &[g, level] : LEVELS.at(ion)) {

    // break if energy level is above ionization energy
    if (level >= ionization_energy) {
      break;
    }

    // add level to sum
    sum += g * exp(-level / (BOLTZMANN_IN_EV_K * temperature));
  }

  return sum;
}

/**
 * @brief Calculate the Saha function for a given element and ionization level.
 *
 * @param element The element to calculate the Saha function for, e.g. "H".
 * @param ionization_level The ionization level of the element.
 * @param temperature The temperature in Kelvin.
 * @return The Saha function.
 */
template <typename T>
const T calculate_saha_function(const std::string &element,
                                const unsigned int ionization_level,
                                const T temperature) {
  // destructure data and create string for ions
  const std::string ion_lower = element + " " + to_roman(ionization_level + 1);
  const std::string ion_higher = element + " " + to_roman(ionization_level + 2);

  // get ionization energy of lower ion
  const double ionization_energy = IONIZATION_ENERGIES.at(ion_lower);

  // calculate right hand side of saha equation
  const T saha = 2.0 / pow(calculate_debroglie_wavelength(temperature), 3) *
                 calculate_partition_function(ion_higher, temperature) /
                 calculate_partition_function(ion_lower, temperature) *
                 exp(-ionization_energy / (BOLTZMANN_IN_EV_K * temperature));

  return saha;
}

/**
 * @brief Calculate the population levels of an element at a given temperature
 * and electron density.
 * @param element The element to calculate the population levels for, e.g. "Fe".
 * @param temperature The temperature in Kelvin.
 * @param electron_density The electron density in m^-3.
 * @return A vector containing the population levels of each ionization level.
 */
template <typename T>
constexpr Eigen::VectorX<T>
calculate_population_levels(const std::string &element, const T temperature,
                            const T electron_density) {

  // get the maximum ionization level of the element
  unsigned int max_ionization_level =
      std::min((unsigned int)5, ATOM_INFO.at(element).element_number - 1);

  // {1; omega[0]; omega[0]omega[1]; ... }
  Eigen::VectorX<T> omega_products(max_ionization_level + 1);
  omega_products[0] = 1.0;

  // will be 1 + omega[0] + omega[0]omega[1] + ...
  T denominator = 1.0;

  // calculate all omegas
  for (unsigned int i = 0; i < max_ionization_level; i++) {
    T omega_i =
        calculate_saha_function(element, i, temperature) / electron_density;

    omega_products[i + 1] = omega_i * omega_products[i];

    denominator += omega_products[i + 1];
  }

  Eigen::VectorX<T> population_levels(max_ionization_level);
  for (unsigned int i = 0; i < max_ionization_level; i++) {
    population_levels[i] = omega_products[i] / denominator;
  }

  return population_levels;
}

/**
 * @brief Calculate the blackbody radiation at a given wavelength and
 * temperature.
 *
 * @param wavelength The wavelength in nanometers.
 * @param temperature The temperature in Kelvin.
 * @return The blackbody radiation intensity.
 */
template <typename T> T blackbody_radiation(double wavelength, T temperature) {

  wavelength *= 1e-9; // wavelength is given in nanometer

  return 2.0 * PLANCK_CONSTANT * pow(SPEED_OF_LIGHT, 2) / pow(wavelength, 5) *
         1.0 /
         (exp(PLANCK_CONSTANT * SPEED_OF_LIGHT /
              (wavelength * BOLTZMANN_IN_J_K * temperature)) -
          1.0);
}

/**
 * @brief Voigt profile calculation.
 * The Voigt profile is a convolution of a Gaussian and a Lorentzian function.
 * This implementation uses the numerical approximation given by McLean
 * @param x The independent variable, in our case the wavelength.
 * @param sigma The standard deviation of the Gaussian component.
 * @param gamma The half-width at half-maximum of the Lorentzian component.
 * @return The value of the Voigt profile at x.
 */
template <typename T> T voigt_profile_approx(T x, T sigma, T gamma) {
  static constexpr std::array<double, 4> A = {-1.2150, -1.3509, -1.2150, -1.3509};
  static constexpr std::array<double, 4> B = {1.2359, 0.3786, -1.2359, -0.3786};
  static constexpr std::array<double, 4> C = {-0.3085, 0.5906, -0.3085, 0.5906};
  static constexpr std::array<double, 4> D = {0.0210, -1.1858, -0.0210, 1.1858};

  static constexpr double SQRT2 = std::numbers::sqrt2;
  // sqrt(pi) / (pi * sqrt(2)) = 1 / (sqrt(pi) * sqrt(2)) = inv_sqrtpi / sqrt2
  static constexpr double SCALE = std::numbers::inv_sqrtpi / std::numbers::sqrt2;

  const T inv_sqrt2_sigma = 1.0 / (SQRT2 * sigma);
  const T X = x * inv_sqrt2_sigma;
  const T Y = gamma * inv_sqrt2_sigma;

  T result = 0;
  for (int i = 0; i < 4; i++) {
    const T dy = Y - A[i];
    const T dx = X - B[i];
    result += (C[i] * dy + D[i] * dx) / (dy * dy + dx * dx);
  }

  return (SCALE / sigma) * result;
}

/**
 * @brief Calculate the standard deviation of the Doppler broadening.
 * @param ion The ion species, e.g., "Fe I".
 * @param line The emission line information.
 * @param temperature The temperature in Kelvin.
 * @return The standard deviation of the Doppler broadening.
 */
template <typename T>
T get_doppler_standard_deviation(const std::string &ion,
                                 const EmissionLine &line, T temperature) {

  const auto element =
      ion.substr(0, ion.find(' ')); // extract element from ion string
  const T std = sqrt(BOLTZMANN_IN_J_K * temperature /
                     (ATOM_INFO.at(element).atomic_weight * ATOMIC_MASS *
                      pow(SPEED_OF_LIGHT, 2))) *
                line.wavelength;
  return std;
}

/**
 * @brief Calculate the full width at half maximum (FWHM) of the Stark
 * broadening.
 * @param line The emission line information.
 * @param electron_density The electron density in m^-3.
 * @param temperature The temperature in Kelvin.
 * @return The FWHM of the Stark broadening in nanometers.
 * @note This uses the formulas given in the Stark-b database.
 */
template <typename T>
T get_stark_broadening_fwhm(const EmissionLine &line, T electron_density,
                            T temperature) {

  const auto w = pow(10, line.a0 + line.a1 * log10(temperature) +
                             line.a2 * pow(log10(temperature), 2)) /
                 10.0; // w is given in angstroem, divide by 10 to get nanometer
  const auto fwhm = w * electron_density / line.n_e_ref *
                    1e-6; // the line n_e_ref is in cm^-3

  return fwhm;
}

/**
 * @brief Calculate the Stark shift of the emission line.
 * @param line The emission line information.
 * @param stark_fwhm The full width at half maximum of the Stark broadening in
 * nanometers.
 * @param temperature The temperature in Kelvin.
 * @return The Stark shift in nanometers.
 * @note This uses the formulas given in the Stark-b database.
 */
template <typename T>
T get_stark_shift(const EmissionLine &line, T temperature) {

  const auto w = pow(10, line.a0 + line.a1 * log10(temperature) +
                             line.a2 * pow(log10(temperature), 2));
  const auto stark_shift_nm = w *
                 (line.b0 + line.b1 * log10(temperature) +
                  line.b2 * pow(log10(temperature), 2)) /
                 10.0; // stark_shift_nm is given in angstroem, divide by 10 to get nanometer
  return stark_shift_nm;
}

inline unsigned int get_maximum_ionization_level(const std::string &element) {
  const unsigned int maximum_ionization_level =
      std::min(ATOM_INFO.at(element).element_number - 1, (unsigned int)5);

  return maximum_ionization_level;
}

template <typename T>
std::pair<int, int>
get_indices_via_heuristic(const Eigen::VectorX<T> &wavelengths,
                          T standard_deviation, T gamma, T center) {

  // use a heuristic for range where we apply broadening
  // here I have used the quartile functions of the normal and lorentz
  // distributions to calculate the 0.999 quantile as an upper bound for the
  // quantile of the voigt we simply use the sum of quantiles of normal an
  // lorentz (haven't checked whether that's true)
  const auto search_range = 4 * standard_deviation + 300 * gamma;

  // search for lower index, that is the index corresponding to the lower
  // end of the broadening range
  auto lower_index =
      std::distance(wavelengths.begin(),
                    std::upper_bound(wavelengths.begin(), wavelengths.end(),
                                     center - search_range));

  // search for upper index, that is the index corresponding to the upper
  // end of the broadening range
  auto upper_index =
      std::distance(wavelengths.begin(),
                    std::lower_bound(wavelengths.begin(), wavelengths.end(),
                                     center + search_range));

  return std::pair<int, int>(lower_index, upper_index);
}

/**
 * @brief Compute the half-extent of a Voigt kernel at a given accuracy.
 *
 * Finds the distance from center at which the Voigt profile drops below
 * accuracy * peak_value. Uses bisection search starting from a heuristic
 * upper bound based on the Gaussian and Lorentzian quantiles.
 *
 * @param sigma Gaussian standard deviation.
 * @param gamma Lorentzian HWHM.
 * @param accuracy Fraction of peak value at which to truncate.
 * @return The half-extent in the same units as sigma/gamma.
 */
inline double compute_kernel_extent(double sigma, double gamma,
                                    double accuracy) {
  const double peak = voigt_profile_approx<double>(0.0, sigma, gamma);
  const double threshold = accuracy * peak;

  // Heuristic upper bound (same as get_indices_via_heuristic)
  double hi = 4 * sigma + 300 * gamma;
  double lo = 0.0;

  // Bisection to refine
  for (int iter = 0; iter < 60; ++iter) {
    double mid = 0.5 * (lo + hi);
    double val = voigt_profile_approx<double>(mid, sigma, gamma);
    if (val > threshold) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return hi;
}

} // namespace libskit

#endif /* FUNCTIONS_H */
