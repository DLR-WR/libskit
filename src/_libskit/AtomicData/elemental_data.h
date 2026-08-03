/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#ifndef ELEMENTAL_DATA_H
#define ELEMENTAL_DATA_H

#include "constants.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace libskit {

struct Ion {
  Element element;
  unsigned int ionization_level;
};

/* define data structures */
struct EnergyLevel {
  double g;
  double energy;

  EnergyLevel(double g_, double energy_) : g(g_), energy(energy_) {};
};
typedef std::unordered_map<std::string, std::vector<EnergyLevel>> Levels;

typedef std::unordered_map<std::string, double> IonizationEnergies;

struct EmissionLine {
  double wavelength;
  double A_ji;
  double E_i;
  double E_j;
  double g_i;
  double g_j;
  double n_e_ref;
  double a0;
  double a1;
  double a2;
  double b0;
  double b1;
  double b2;
  double acc;  // Accuracy fraction for A_ji (0.003–0.50)
  // Multiplier applied to the Stark FWHM (Lorentzian width in the Voigt).
  // Default 1.0; set via `set_stark_multipliers(...)` for Monte-Carlo
  // perturbation and picked up by the Enzyme AD path as the differentiable
  // Stark input.
  double stark_mult = 1.0;
};
typedef std::unordered_map<std::string, std::vector<EmissionLine>>
    EmissionLines;

// Database path management - must be called before using data
void set_database_path(const std::string &path);
std::string get_database_path();
void load_databases(int verbose = 1);
void reload_databases();

// Aki modification API for Monte Carlo perturbation studies
void set_aki_multipliers(
    const std::unordered_map<std::string, std::vector<double>> &multipliers);
void reset_aki();

// Get accuracy fractions for all lines in the database, keyed by ion
std::unordered_map<std::string, std::vector<double>> get_aki_accuracies();

// Get the current in-memory A_ji values for all lines, keyed by ion.
// Reflects any active set_aki_multipliers() override.
std::unordered_map<std::string, std::vector<double>> get_aki_values();

// Stark-FWHM modification API for Monte-Carlo perturbation studies.
// Multipliers are absolute (line.stark_mult := factor); default 1.0.
// The multiplier is applied to the Stark FWHM (Lorentzian width in the
// Voigt) inside `compute_emission_coeff`. `stark_shift` is untouched.
void set_stark_multipliers(
    const std::unordered_map<std::string, std::vector<double>> &multipliers);
void reset_stark();
std::unordered_map<std::string, std::vector<double>> get_stark_multipliers();

// Accuracy management: fraction of maximum peak height below which
// emission lines are discarded and broadening wings are truncated
void set_accuracy(double accuracy);
double get_accuracy();

// Fine grid resolution control
void set_fine_samples_per_fwhm(double samples);
double get_fine_samples_per_fwhm();

// Database loading functions
IonizationEnergies get_ionization_energies(const std::string &file_path);
EmissionLines get_lines_data(const std::string &file_path);
Levels get_levels_data(const std::string &file_path);

// Global constants (lazy-loaded with configurable path)
extern EmissionLines LINES;
extern IonizationEnergies IONIZATION_ENERGIES;
extern Levels LEVELS;

} // namespace libskit

#endif /* ELEMENTAL_DATA_H */
