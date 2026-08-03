/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "elemental_data.h"
#include "include/csv.h"
#include "utils/to_roman.h"
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace libskit {

// Static variables for database path
static std::string database_path = "databases";  // Default path
static bool databases_loaded = false;

// Fraction of maximum peak height below which lines are discarded
// and broadening wings are truncated
static double accuracy = 1e-5;
static double fine_samples_per_fwhm = 5.0;   // Fine grid points per narrowest FWHM

// Global database storage
EmissionLines LINES;
EmissionLines LINES_BACKUP;
IonizationEnergies IONIZATION_ENERGIES;
Levels LEVELS;

// Forward declaration
void load_databases();

IonizationEnergies get_ionization_energies(const std::string &file_path) {

  io::CSVReader<2> in(file_path);
  in.read_header(io::ignore_extra_column, "Sp. Name",
                 "Ionization Energy (b) (eV)");
  std::string species_name;
  double energy;

  IonizationEnergies ionization_energies;

  while (in.read_row(species_name, energy)) {
    ionization_energies[species_name] = energy;
  }

  return ionization_energies;
}

EmissionLines get_lines_data(const std::string &file_path) {

  io::CSVReader<16> in(file_path);
  in.read_header(io::ignore_extra_column, "element", "sp_num",
                 "ritz_wl_vac(nm)", "Aki(s^-1)", "Ei(eV)", "Ek(eV)", "g_i",
                 "g_k", "N (cm-3)", "electron a0", "electron a1", "electron a2",
                 "electron b0", "electron b1", "electron b2", "Acc");

  std::string element;
  int sp_num;
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
  double acc;

  EmissionLines lines_data;

  while (in.read_row(element, sp_num, wavelength, A_ji, E_i, E_j, g_i, g_j,
                     n_e_ref, a0, a1, a2, b0, b1, b2, acc)) {
    const std::string ion = element + " " + to_roman(sp_num);

    lines_data[ion].push_back(EmissionLine(wavelength, A_ji, E_i, E_j, g_i, g_j,
                                           n_e_ref, a0, a1, a2, b0, b1, b2, acc));
  }

  return lines_data;
}

Levels get_levels_data(const std::string &file_path) {

  io::CSVReader<3> in(file_path);
  in.read_header(io::ignore_extra_column, "g", "Level (eV)", "Sp. Name");

  double g;
  double level;
  std::string species_name;

  Levels levels;

  while (in.read_row(g, level, species_name)) {

    levels[species_name].push_back(EnergyLevel(g, level));
  }

  return levels;
}

// Database path and loading management
void set_database_path(const std::string &path) {
    database_path = path;
    databases_loaded = false;  // Mark for reload
}

std::string get_database_path() {
    return database_path;
}

void load_databases(int verbose) {
    if (databases_loaded) return;

    if (verbose > 0) {
        std::cout << "Loading databases from: " << database_path << std::endl;
    }

    try {
        LINES = get_lines_data(database_path + "/lines_processed_all.csv");
        LINES_BACKUP = LINES;  // Deep copy for reset_aki()
        IONIZATION_ENERGIES = get_ionization_energies(database_path + "/ionization_energies_processed.csv");
        LEVELS = get_levels_data(database_path + "/levels_processed.csv");

        databases_loaded = true;

        if (verbose > 0) {
            std::cout << "✓ Databases loaded successfully!" << std::endl;

            int total_lines = 0;
            if (verbose > 1) {
                // Sort ions alphabetically for verbose output
                std::vector<std::pair<std::string, size_t>> sorted_ions;
                for (const auto& [ion, lines] : LINES) {
                    total_lines += lines.size();
                    sorted_ions.emplace_back(ion, lines.size());
                }
                std::sort(sorted_ions.begin(), sorted_ions.end());

                for (const auto& [ion, line_count] : sorted_ions) {
                    std::cout << "  " << ion << ": " << line_count << " lines" << std::endl;
                }
            } else {
                // Just count total lines for verbose == 1
                for (const auto& [ion, lines] : LINES) {
                    total_lines += lines.size();
                }
            }
            std::cout << "  Lines: " << total_lines << " lines across " << LINES.size() << " ions" << std::endl;

            std::cout << "  Ionization energies: " << IONIZATION_ENERGIES.size() << " entries" << std::endl;
            std::cout << "  Levels: " << LEVELS.size() << " ions" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading databases: " << e.what() << std::endl;
        throw;
    }
}

void reload_databases() {
    databases_loaded = false;
    load_databases(1);  // Default verbosity
}

void set_aki_multipliers(
    const std::unordered_map<std::string, std::vector<double>> &multipliers) {
  for (const auto &[ion, factors] : multipliers) {
    auto it = LINES.find(ion);
    if (it == LINES.end()) {
      throw std::runtime_error("set_aki_multipliers: unknown ion '" + ion +
                               "'");
    }
    auto &lines = it->second;
    if (factors.size() != lines.size()) {
      throw std::runtime_error(
          "set_aki_multipliers: expected " + std::to_string(lines.size()) +
          " factors for '" + ion + "', got " +
          std::to_string(factors.size()));
    }
    for (size_t i = 0; i < lines.size(); ++i) {
      lines[i].A_ji *= factors[i];
    }
  }
}

void reset_aki() {
  // Restore only A_ji from the backup; leave stark_mult (and any other
  // future mutable per-line state) untouched so that stark and aki
  // perturbations can be reset independently.
  for (auto &[ion, lines] : LINES) {
    auto it = LINES_BACKUP.find(ion);
    if (it == LINES_BACKUP.end())
      continue;
    const auto &backup_lines = it->second;
    const size_t n = std::min(lines.size(), backup_lines.size());
    for (size_t i = 0; i < n; ++i) {
      lines[i].A_ji = backup_lines[i].A_ji;
    }
  }
}

void set_stark_multipliers(
    const std::unordered_map<std::string, std::vector<double>> &multipliers) {
  for (const auto &[ion, factors] : multipliers) {
    auto it = LINES.find(ion);
    if (it == LINES.end()) {
      throw std::runtime_error("set_stark_multipliers: unknown ion '" + ion +
                               "'");
    }
    auto &lines = it->second;
    if (factors.size() != lines.size()) {
      throw std::runtime_error(
          "set_stark_multipliers: expected " + std::to_string(lines.size()) +
          " factors for '" + ion + "', got " +
          std::to_string(factors.size()));
    }
    for (size_t i = 0; i < lines.size(); ++i) {
      lines[i].stark_mult = factors[i];
    }
  }
}

void reset_stark() {
  for (auto &[ion, lines] : LINES) {
    for (auto &line : lines)
      line.stark_mult = 1.0;
  }
}

std::unordered_map<std::string, std::vector<double>> get_stark_multipliers() {
  std::unordered_map<std::string, std::vector<double>> result;
  for (const auto &[ion, lines] : LINES) {
    std::vector<double> mults;
    mults.reserve(lines.size());
    for (const auto &line : lines) {
      mults.push_back(line.stark_mult);
    }
    result[ion] = std::move(mults);
  }
  return result;
}

std::unordered_map<std::string, std::vector<double>> get_aki_accuracies() {
  std::unordered_map<std::string, std::vector<double>> result;
  for (const auto &[ion, lines] : LINES) {
    std::vector<double> accs;
    accs.reserve(lines.size());
    for (const auto &line : lines) {
      accs.push_back(line.acc);
    }
    result[ion] = std::move(accs);
  }
  return result;
}

// Return the *current* in-memory A_ji values (after any multipliers).
std::unordered_map<std::string, std::vector<double>> get_aki_values() {
  std::unordered_map<std::string, std::vector<double>> result;
  for (const auto &[ion, lines] : LINES) {
    std::vector<double> aki;
    aki.reserve(lines.size());
    for (const auto &line : lines) {
      aki.push_back(line.A_ji);
    }
    result[ion] = std::move(aki);
  }
  return result;
}

// Accuracy management functions
void set_accuracy(double value) {
    accuracy = value;
}

double get_accuracy() {
    return accuracy;
}

void set_fine_samples_per_fwhm(double samples) {
    fine_samples_per_fwhm = samples;
}

double get_fine_samples_per_fwhm() {
    return fine_samples_per_fwhm;
}

} // namespace libskit
