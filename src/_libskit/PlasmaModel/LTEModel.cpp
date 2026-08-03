/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include "LTEModel.h"
#include "physics_kernel.h"
#include <limits>
#include <stdexcept>

namespace libskit {

LTEModel::LTEModel(std::map<std::string, double> _element_ratio,
                   double _temperature, double _electron_density, double _length)
    : element_ratio(std::move(_element_ratio)), temperature(_temperature),
      electron_density(_electron_density), length(_length) {}

std::shared_ptr<PlasmaModel> LTEModel::clone() const {
  return std::make_shared<LTEModel>(*this);
}

double &LTEModel::get_parameter(const std::string &parameter_name) {

  if (parameter_name == "temperature") {
    return temperature;
  } else if (parameter_name == "electron_density") {
    return electron_density;
  } else if (parameter_name == "length") {
    return length;
  } else {
    return element_ratio[parameter_name];
  }
}

void LTEModel::set_parameter(const std::string &parameter_name, double value) {

  if (parameter_name == "temperature") {
    temperature = value;
  } else if (parameter_name == "electron_density") {
    electron_density = value;
  } else if (parameter_name == "length") {
    length = value;
  } else if (element_ratio.count(parameter_name) > 0) {
    element_ratio[parameter_name] = value;
  } else {
    throw std::invalid_argument(
        "LTEModel::set_parameter: unknown parameter name '" + parameter_name +
        "'. Valid names are 'temperature', 'electron_density', 'length', or "
        "any element symbol present in element_ratio.");
  }
}

std::vector<std::string> LTEModel::get_parameter_names() const {
  std::vector<std::string> parameter_names;
  // Main plasma parameters first, followed by per-element mass ratios.
  parameter_names.push_back("temperature");
  parameter_names.push_back("electron_density");
  parameter_names.push_back("length");
  for (const auto &[element, ratio] : element_ratio) {
    parameter_names.push_back(element);
  }
  return parameter_names;
}

double LTEModel::calculate_f_sum(
    const std::map<std::string, Eigen::VectorXd> &population_fractions) const {
  double f_sum = 0.0;
  for (const auto &[element_, ratio] : element_ratio) {
    double ion_sum = 0;

    const unsigned int maximum_ion_level =
        get_maximum_ionization_level(element_);

    // sum over ionization states --> 1*Fe_II + 2*Fe_III + ...
    // be careful here! we only sum over the ionized populations!
    for (unsigned int i = 0; i < maximum_ion_level - 1; i++) {
      ion_sum += (i + 1) * population_fractions.at(element_)[i + 1];
    }
    f_sum += element_ratio.at(element_) / ATOM_INFO.at(element_).atomic_weight *
             ion_sum;
  }

  return f_sum;
}

std::map<std::string, Eigen::VectorXd>
LTEModel::get_population_fractions() const {
  std::map<std::string, Eigen::VectorXd> population_fractions;
  for (const auto &[element, ratio] : element_ratio) {
    population_fractions[element] =
        calculate_population_levels(element, temperature, electron_density);
  }
  return population_fractions;
}

std::map<std::string, double> LTEModel::_calculate_element_fractions() const {

  std::map<std::string, Eigen::VectorXd> population_fractions;
  for (const auto &[element, ratio] : element_ratio) {
    population_fractions[element] =
        calculate_population_levels(element, temperature, electron_density);
  }

  // f_sum couples the LTE ionization balance to the requested electron density.
  const double f_sum = calculate_f_sum(population_fractions);

  std::map<std::string, double> fractions;
  for (const auto &[element, ratio] : element_ratio) {
    const double n_k = electron_density *                    // n_e
                       element_ratio.at(element) /           // c_k
                       ATOM_INFO.at(element).atomic_weight / // M_k
                       f_sum;

    fractions[element] = n_k;
  }

  return fractions;
}

double LTEModel::calculate_emission_for_one_line(
    const unsigned int ionization_level, const double n_k,
    const Eigen::VectorXd &population_fraction,
    const double partition_function, const EmissionLine &line_data) const {

  const double m_kzj = line_data.g_j /      // g_kzj
                       partition_function * // Z_kz
                       exp(-line_data.E_j / (BOLTZMANN_IN_EV_K * temperature));

  const double n_kzj =
      n_k *                                   // n_k
      population_fraction[ionization_level] * // beta_{k,z}
      m_kzj;                                  // m_kzi

  const double emission = ELEMENTARY_CHARGE / (4.0 * std::numbers::pi) *
                          (line_data.E_j - line_data.E_i) * line_data.A_ji *
                          n_kzj;

  return emission;
}

std::pair<EmissionData<double>, double> LTEModel::get_emission_coefficient_data(
    const Eigen::VectorXd &wavelengths) const {
  // Population levels are reused across all lines, so compute them once.
  std::map<std::string, Eigen::VectorXd> population_fractions;
  for (const auto &[element, ratio] : element_ratio) {
    population_fractions[element] =
        calculate_population_levels(element, temperature, electron_density);
  }

  // f_sum is constant for the whole plasma state.
  const double f_sum = calculate_f_sum(population_fractions);

  EmissionData<double> emission_data;

  double max_irradiance_peak = 0.0;
  for (const auto &[element, ratio] : element_ratio) {

    // Element number density for this species.
    const double n_k = electron_density *                    // n_e
                       element_ratio.at(element) /           // c_k
                       ATOM_INFO.at(element).atomic_weight / // M_k
                       f_sum;

    const unsigned int maximum_ionization_level =
        get_maximum_ionization_level(element);

    for (unsigned int ionization_level = 0;
         ionization_level < maximum_ionization_level; ionization_level++) {

      const std::string ion = element + " " + to_roman(ionization_level + 1);

      // Some elements have gaps in the tabulated ion stages.
      if (!LINES.count(ion)) {
        continue;
      }
      const std::vector<EmissionLine> ion_lines = LINES.at(ion);

      const double partition_function =
          calculate_partition_function(ion, temperature);

      for (const auto &line : ion_lines) {
        // LINES are wavelength-sorted, so we can early-skip / early-break.
        if (line.wavelength < wavelengths[0]) {
          continue;
        }
        if (line.wavelength > wavelengths[wavelengths.size() - 1]) {
          break;
        }
        const double emission = calculate_emission_for_one_line(
            ionization_level, n_k, population_fractions.at(element),
            partition_function, line);

        const double sigma =
            get_doppler_standard_deviation(ion, line, temperature);
        const double gamma =
            0.5 * get_stark_broadening_fwhm(line, electron_density, temperature);
        const double eps_peak =
            emission * 1.0e9 * voigt_profile_approx(0.0, sigma, gamma);

        // Convert the line-center emission coefficient into an irradiance peak
        // through the LTE radiative transfer equation, which naturally caps
        // strong lines at the blackbody level.
        const double bb = blackbody_radiation(line.wavelength, temperature);
        const double irradiance_peak =
            bb * (1.0 - exp(-eps_peak / bb * length));

        if (irradiance_peak > max_irradiance_peak) {
          max_irradiance_peak = irradiance_peak;
        }

        emission_data[ion].push_back(std::pair(line, emission));
      }
    }
  }

  EmissionData<double> emission_data_sorted;

  const double accuracy = libskit::get_accuracy();
  for (const auto &[ion, lines] : emission_data) {
    for (const auto &[line, emission] : lines) {
      const double sigma =
          get_doppler_standard_deviation(ion, line, temperature);
      const double gamma =
          0.5 * get_stark_broadening_fwhm(line, electron_density, temperature);
      const double eps_peak =
          emission * 1.0e9 * voigt_profile_approx(0.0, sigma, gamma);
      const double bb = blackbody_radiation(line.wavelength, temperature);
      const double irradiance_peak =
          bb * (1.0 - exp(-eps_peak / bb * length));
      if (irradiance_peak > accuracy * max_irradiance_peak) {
        emission_data_sorted[ion].push_back(std::pair(line, emission));
      }
    }
  }

  return std::make_pair(emission_data_sorted, max_irradiance_peak);
}

Eigen::VectorXd LTEModel::calculate_emission_coefficient(
    const Eigen::VectorXd &wavelengths) const {
  // Delegate the line-broadening kernel to the shared physics implementation
  // in physics_kernel.h (single source of truth, also used by Enzyme).
  std::vector<std::string> element_names;
  std::vector<double> ratios;
  element_names.reserve(element_ratio.size());
  ratios.reserve(element_ratio.size());
  for (const auto &[name, r] : element_ratio) {
    element_names.push_back(name);
    ratios.push_back(r);
  }

  const auto data = physics::build_data_from_elements(element_names,
                                                      libskit::get_accuracy());

  std::vector<double> ec_vec;
  physics::compute_emission_coeff(temperature, electron_density, length,
                                  ratios.data(),
                                  static_cast<int>(element_names.size()),
                                  &data, wavelengths.data(),
                                  wavelengths.size(), ec_vec);

  Eigen::VectorXd emission_coefficient =
      Eigen::Map<Eigen::VectorXd>(ec_vec.data(), wavelengths.size());

  // Add free-free and free-bound continuum on top of the broadened line spectrum.
  if (include_continuum) {
    add_continuum_emission(wavelengths, emission_coefficient);
  }

  return emission_coefficient;
}

void LTEModel::add_continuum_emission(
    const Eigen::VectorXd &wavelengths,
    Eigen::VectorXd &emission_coefficient) const {
  auto [ff, fb] = calculate_continuum_emission(wavelengths);
  emission_coefficient += ff + fb;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
LTEModel::calculate_continuum_emission(
    const Eigen::VectorXd &wavelengths) const {

  const long N = wavelengths.size();
  Eigen::VectorXd eps_ff_vec = Eigen::VectorXd::Zero(N);
  Eigen::VectorXd eps_fb_vec = Eigen::VectorXd::Zero(N);

  // Collect ion densities N_a^z for all charged states and precompute
  // sum(z^2 * N_a^z) for the free-free term.
  auto pop_fractions = get_population_fractions();
  const double f_sum = calculate_f_sum(pop_fractions);

  struct IonInfo {
    unsigned int z;
    double density;
  };
  std::vector<IonInfo> ions;

  double z2_sum = 0.0;

  for (const auto &[element, ratio] : element_ratio) {
    const double n_k = electron_density * element_ratio.at(element) /
                       ATOM_INFO.at(element).atomic_weight / f_sum;

    const unsigned int max_ion = get_maximum_ionization_level(element);

    const auto &pop = pop_fractions.at(element);

    for (unsigned int ion_idx = 1; ion_idx < max_ion; ion_idx++) {
      const unsigned int z = ion_idx;
      const double N_az = n_k * pop[ion_idx];
      if (N_az > 0.0) {
        ions.push_back({z, N_az});
        z2_sum += z * z * N_az;
      }
    }
  }

  const double kT_J = BOLTZMANN_IN_J_K * temperature;
  const double EH_over_kT = RYDBERG_ENERGY_J / kT_J;

  const double alpha_a0_cubed =
      pow(FINE_STRUCTURE_CONSTANT * BOHR_RADIUS, 3);
  const double prefactor_ff =
      (128.0 * std::numbers::pi * std::numbers::pi / 3.0) * alpha_a0_cubed *
      RYDBERG_ENERGY_J / sqrt(3.0 * std::numbers::pi) * sqrt(EH_over_kT) *
      electron_density;

  const double fb_factor = (2.0 / 3.0) * EH_over_kT;

  constexpr int N_MAX = 30;

  for (long wi = 0; wi < wavelengths.size(); wi++) {
    const double lambda_m = wavelengths[wi] * 1e-9;

    // Free-free prefactor at this wavelength.
    const double hc_over_lambda_kT =
        PLANCK_CONSTANT * SPEED_OF_LIGHT / (lambda_m * kT_J);
    const double C_ff = prefactor_ff * SPEED_OF_LIGHT / (lambda_m * lambda_m) *
                        exp(-hc_over_lambda_kT);

    const double eps_ff = C_ff * z2_sum;

    const double C_fb = C_ff * fb_factor;
    double eps_fb = 0.0;

    const double photon_energy_J =
        PLANCK_CONSTANT * SPEED_OF_LIGHT / lambda_m;

    for (const auto &ion : ions) {
      const unsigned int z = ion.z;

      const double n_star_real =
          z * sqrt(RYDBERG_ENERGY_J / photon_energy_J);
      const int n_star = std::max(1, (int)std::ceil(n_star_real));

      if (n_star > N_MAX)
        continue;

      for (int n = n_star; n <= N_MAX; n++) {
        // z^4 / n^3 * exp(-z^2 E_H / (n^2 k_B T)) * N_a^z
        const double zd = static_cast<double>(z);
        const double nd = static_cast<double>(n);
        eps_fb += (zd * zd * zd * zd) / (nd * nd * nd) *
                  exp(-zd * zd * EH_over_kT / (nd * nd)) * ion.density;
      }
    }
    eps_fb *= C_fb;

    eps_ff_vec[wi] = eps_ff;
    eps_fb_vec[wi] = eps_fb;
  }

  return {eps_ff_vec, eps_fb_vec};
}

Eigen::VectorXd LTEModel::calculate_irradiance(
    const Eigen::VectorXd &wavelengths) const {
  // First compute emission coefficients, then apply the single-layer RTE.
  const Eigen::VectorXd emission_coefficient =
      calculate_emission_coefficient(wavelengths);

  const long n = wavelengths.size();
  std::vector<double> ec_vec(emission_coefficient.data(),
                             emission_coefficient.data() + n);
  std::vector<double> irr_vec;
  physics::apply_rte_single(wavelengths.data(), n, ec_vec, temperature, length,
                            irr_vec);
  return Eigen::Map<Eigen::VectorXd>(irr_vec.data(), n);
}

std::vector<std::tuple<std::string, double, double, double, double>>
LTEModel::get_lines(const Eigen::VectorXd &wavelengths) const {
  const auto [emission_coefficient_data, max_irradiance_peak] =
      get_emission_coefficient_data(wavelengths);

  std::vector<std::tuple<std::string, double, double, double, double>> result;

  for (const auto &[ion, data] : emission_coefficient_data) {
    for (const auto &[line, emission] : data) {
      // Gaussian Doppler and Lorentzian Stark components combine into an
      // approximate Voigt FWHM.
      const double sigma_doppler =
          get_doppler_standard_deviation(ion, line, temperature);
      const double fwhm_gauss = 2.0 * sqrt(2.0 * log(2.0)) * sigma_doppler;

      const double fwhm_lorentz =
          get_stark_broadening_fwhm(line, electron_density, temperature);

      const double fwhm_voigt =
          0.5343 * fwhm_lorentz +
          sqrt(0.2169 * fwhm_lorentz * fwhm_lorentz + fwhm_gauss * fwhm_gauss);

      result.push_back(
          std::make_tuple(ion, line.wavelength, emission, fwhm_voigt, line.acc));
    }
  }

  return result;
}

double LTEModel::get_minimum_voigt_fwhm(
    const Eigen::VectorXd &wavelengths) const {
  auto lines_data = get_lines(wavelengths);

  double min_fwhm = std::numeric_limits<double>::max();
  for (const auto &[ion, wl, emission, fwhm, acc] : lines_data) {
    if (fwhm > 0.0 && fwhm < min_fwhm) {
      min_fwhm = fwhm;
    }
  }

  return min_fwhm;
}

} // namespace libskit
