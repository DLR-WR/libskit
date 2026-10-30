/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */
#include <iomanip>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include "libskit.h"
#include "enzyme_gradients.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

std::vector<double> create_range(double start, double stop, double step) {
  std::vector<double> range;

  for (double i = start; i < stop; i += step) {
    range.push_back(i);
  };

  return range;
};

double calc(std::vector<double> &vec) { return vec.back() - vec.front(); };

namespace py = pybind11;

PYBIND11_MODULE(_libskit, m) {
  m.doc() = R"pbdoc(
        LIBSkit C++ Extension Module
        ============================

        This module provides the core C++ functionality for LIBSkit, a library
        for simulating and analyzing Laser-Induced Breakdown Spectroscopy (LIBS)
        spectra.

        Example
        -------
        >>> import libskit
        >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
        >>> instrument = libskit.GenericSpectrometer(0.05, 0.01)
        >>> wavelengths = np.linspace(200, 900, 5000)
        >>> spectrum = instrument.calculate_spectrum(wavelengths, model)
    )pbdoc";

  /*
   * Spectrum class
   */
  py::class_<libskit::Spectrum>(m, "Spectrum",
      R"pbdoc(
          Container for spectroscopic data with wavelength-intensity pairs.

          A Spectrum object stores matched arrays of wavelength and intensity
          values, representing either measured or simulated spectroscopic data.

          Parameters
          ----------
          wavelengths : numpy.ndarray
              1D array of wavelength values in nanometers. Must be the same
              length as irradiance.
          irradiance : numpy.ndarray
              1D array of intensity/irradiance values corresponding to each
              wavelength point.

          Attributes
          ----------
          wavelengths : numpy.ndarray
              Wavelength array in nanometers.
          irradiance : numpy.ndarray
              Intensity/irradiance array.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>> wl = np.linspace(200, 900, 1000)
          >>> intensity = np.random.rand(1000)
          >>> spectrum = libskit.Spectrum(wl, intensity)
      )pbdoc")
      .def(py::init([](py::array_t<double> wl, py::array_t<double> irr) {
          auto wl_buf = wl.request();
          auto irr_buf = irr.request();

          if (wl_buf.ndim != 1 || irr_buf.ndim != 1) {
              throw std::runtime_error("Arrays must be 1-dimensional");
          }

          if (wl_buf.size != irr_buf.size) {
              throw std::runtime_error("Wavelength and irradiance arrays must have the same size");
          }

          Eigen::VectorXd wavelengths = Eigen::Map<const Eigen::VectorXd>(
              static_cast<const double*>(wl_buf.ptr), wl_buf.size);
          Eigen::VectorXd irradiance = Eigen::Map<const Eigen::VectorXd>(
              static_cast<const double*>(irr_buf.ptr), irr_buf.size);

          return libskit::Spectrum(wavelengths, irradiance);
      }))
      .def_property("wavelengths",
          [](const libskit::Spectrum &s) {
              return py::cast(s.wavelengths);
          },
          [](libskit::Spectrum &s, py::array_t<double> arr) {
              auto buf = arr.request();
              if (buf.ndim != 1) {
                  throw std::runtime_error("Array must be 1-dimensional");
              }
              s.wavelengths = Eigen::Map<const Eigen::VectorXd>(
                  static_cast<const double*>(buf.ptr), buf.size);
          })
      .def_property("irradiance",
          [](const libskit::Spectrum &s) {
              return py::cast(s.irradiance);
          },
          [](libskit::Spectrum &s, py::array_t<double> arr) {
              auto buf = arr.request();
              if (buf.ndim != 1) {
                  throw std::runtime_error("Array must be 1-dimensional");
              }
              s.irradiance = Eigen::Map<const Eigen::VectorXd>(
                  static_cast<const double*>(buf.ptr), buf.size);
          });

  /*
   * Instrument classes
   */
  py::class_<libskit::Instrument>(m, "Instrument",
      R"pbdoc(
          Abstract base class for spectroscopic instrument models.

          This class defines the interface for all instrument models in LIBSkit.
          Instruments simulate how plasma spectra would be measured by real
          spectrometers, including effects such as instrumental line broadening.
      )pbdoc");
  py::class_<libskit::GenericSpectrometer, libskit::Instrument>(
      m, "GenericSpectrometer",
      R"pbdoc(
          Generic spectrometer model with configurable Voigt line broadening.

          This class simulates how a spectrometer would measure plasma emission,
          applying instrumental broadening using a Voigt profile (combination of
          Gaussian and Lorentzian line shapes). The broadening is performed
          efficiently using FFT convolution.

          Parameters
          ----------
          sigma : float
              Gaussian broadening parameter (standard deviation) in nanometers.
              Represents the Doppler-like instrumental broadening component.
          gamma : float
              Lorentzian broadening parameter (half-width at half-maximum) in
              nanometers. Represents pressure-like broadening effects.

          Attributes
          ----------
          sigma : float
              Gaussian broadening parameter in nm.
          gamma : float
              Lorentzian broadening parameter in nm.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>> instrument = libskit.GenericSpectrometer(sigma=0.05, gamma=0.01)
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> wavelengths = np.linspace(200, 900, 5000)
          >>> spectrum = instrument.calculate_spectrum(wavelengths, model)
      )pbdoc")
      .def(py::init<double, double>())
      .def_readwrite("sigma", &libskit::GenericSpectrometer::sigma)
      .def_readwrite("gamma", &libskit::GenericSpectrometer::gamma)
      .def("calculate_spectrum",
           &libskit::GenericSpectrometer::calculate_spectrum);
  py::class_<libskit::Aryelle, libskit::Instrument>(
      m, "Aryelle",
      R"pbdoc(
          Aryelle spectrometer model with instrument-specific characteristics.

          This class implements the specific instrumental response function of
          the LTB Aryelle spectrometer.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>> aryelle = libskit.Aryelle()
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> wavelengths = np.linspace(200, 900, 5000)
          >>> spectrum = aryelle.calculate_spectrum(wavelengths, model)
      )pbdoc")
      .def(py::init<>())
      .def("calculate_spectrum",
           &libskit::Aryelle::calculate_spectrum);
  py::class_<libskit::ChemCam, libskit::Instrument>(
      m, "ChemCam",
      R"pbdoc(
          ChemCam spectrometer model for the Mars Science Laboratory rover.

          Models the ChemCam LIBS instrument which consists of three separate
          CCD spectrometers with different spectral resolutions:

          - UV:     240.1-342.2 nm,  FWHM = 0.15 nm
          - VIO:    382.1-469.3 nm,  FWHM = 0.20 nm
          - VISNIR: 474.0-906.5 nm,  FWHM = 0.61 nm

          Instrumental broadening is a pure Gaussian for each range.
          Wavelengths in the inter-detector gaps (342.2-382.1 nm, 469.3-474.0 nm)
          receive zero intensity.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>> chemcam = libskit.ChemCam()
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> wavelengths = np.linspace(240, 840, 10000)
          >>> spectrum = chemcam.calculate_spectrum(wavelengths, model)
      )pbdoc")
      .def(py::init<>())
      .def("calculate_spectrum",
           &libskit::ChemCam::calculate_spectrum)
      .def_readwrite("scale_vio", &libskit::ChemCam::scale_vio,
                     "Relative scale factor for the VIO CCD (382.1–469.3 nm). "
                     "UV is the reference (implicitly 1.0).")
      .def_readwrite("scale_visnir", &libskit::ChemCam::scale_visnir,
                     "Relative scale factor for the VISNIR CCD (474.0–906.5 nm). "
                     "UV is the reference (implicitly 1.0).");
  py::class_<libskit::SlopingSpectrometer, libskit::Instrument>(
      m, "SlopingSpectrometer",
      R"pbdoc(
          Spectrometer with wavelength-dependent Gaussian broadening.

          Models instrumental broadening as a Gaussian whose FWHM grows
          linearly with wavelength: FWHM(λ) = m·λ + b.

          Uses a change-of-variable trick to convert the position-dependent
          convolution into a shift-invariant one, enabling efficient FFT-based
          computation in O(N log N) instead of O(N²).

          Parameters
          ----------
          fwhm_slope : float
              Slope m of FWHM(λ) = m·λ + b  (nm/nm, dimensionless).
          fwhm_intercept : float
              Intercept b of FWHM(λ) = m·λ + b  (nm).

          Attributes
          ----------
          fwhm_slope : float
              Slope of the linear FWHM model.
          fwhm_intercept : float
              Intercept of the linear FWHM model.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>> spec = libskit.SlopingSpectrometer(fwhm_slope=1.5e-4, fwhm_intercept=7.8e-3)
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> wavelengths = np.linspace(200, 900, 5000)
          >>> spectrum = spec.calculate_spectrum(wavelengths, model)
      )pbdoc")
      .def(py::init<double, double>(),
           py::arg("fwhm_slope"), py::arg("fwhm_intercept"))
      .def_readwrite("fwhm_slope",
                     &libskit::SlopingSpectrometer::fwhm_slope)
      .def_readwrite("fwhm_intercept",
                     &libskit::SlopingSpectrometer::fwhm_intercept)
      .def("calculate_spectrum",
           &libskit::SlopingSpectrometer::calculate_spectrum);

  /*
   * Loss type
   */
  py::enum_<libskit::LossType>(m, "LossType",
      R"pbdoc(
          Enumeration of available loss function types for optimization.

          Different loss functions can be used depending on the characteristics
          of the data and the desired optimization behavior.

          Attributes
          ----------
          L2Loss
              Standard L2 (least squares) loss function. Minimizes the sum of
              squared residuals between measured and simulated spectra.
          IntensityWeightedLoss
              Custom loss function designed for LIBS data that is more robust
              to outliers and baseline variations.
          L1Loss
              L1 (sum of absolute residuals) loss. More robust to outliers
              than L2 — large residuals are penalised linearly, not
              quadratically.
          LogL2Loss
              L2 loss applied to log(1 + spectrum) transformed data.
              The logarithmic compression amplifies the contribution of
              small peaks relative to dominant ones, improving recovery
              of minor/trace species.
          WeightedL2Loss
              Per-pixel weighted L2 loss
              Σ w_i (y_sim_i - y_exp_i)² / Σ w_i.
              The weight vector is set on the Experiment via the
              ``loss_weights`` attribute (length must equal the
              spectrum length). Useful for downweighting spectral
              regions with high transition-probability uncertainty.
          HuberLoss
              Robust loss, quadratic for |residual| <= threshold and
              linear beyond it. The threshold is set on the Experiment
              via the ``robust_threshold`` attribute. Dampens (but does
              not eliminate) the influence of persistently mismatched
              lines, e.g. from self-absorption or missing database
              entries.
          TukeyBiweightLoss
              Redescending robust loss: saturates to a constant for
              |residual| >= threshold, giving exactly zero gradient
              contribution from points beyond it (full outlier
              rejection, not just dampening). The threshold is set on
              the Experiment via the ``robust_threshold`` attribute.
      )pbdoc")
      .value("L2Loss", libskit::LossType::L2Loss)
      .value("IntensityWeightedLoss", libskit::LossType::IntensityWeightedLoss)
      .value("L1Loss", libskit::LossType::L1Loss)
      .value("LogL2Loss", libskit::LossType::LogL2Loss)
      .value("WeightedL2Loss", libskit::LossType::WeightedL2Loss)
      .value("HuberLoss", libskit::LossType::HuberLoss)
      .value("TukeyBiweightLoss", libskit::LossType::TukeyBiweightLoss)
      .export_values();

  /*
   * Normalization type
   */
  py::enum_<libskit::NormalizationType>(m, "NormalizationType",
      R"pbdoc(
          Enumeration of normalization strategies for spectral fitting.

          Controls how simulated and experimental spectra are brought to a
          common scale before computing the loss function.

          Attributes
          ----------
          MaxNorm
              Legacy normalization: independently divide each spectrum by its
              maximum value. Simple but can distort relative peak heights.
          AnalyticNorm
              Analytically optimal scale factor. Computes c = (y_sim · y_exp) /
              (y_exp · y_exp) that minimizes ||y_sim - c * y_exp||². Fully
              differentiable and better for gradient-based optimization.
          OptimizedScaleNorm
              Free scale factor optimized via gradient descent. Same structure
              as AnalyticNorm but `c` is a free parameter rather than
              analytically computed. Initialize via Optimizer which sets
              scale_factor from a first analytic estimate.
      )pbdoc")
      .value("MaxNorm", libskit::NormalizationType::MaxNorm)
      .value("AnalyticNorm", libskit::NormalizationType::AnalyticNorm)
      .value("OptimizedScaleNorm", libskit::NormalizationType::OptimizedScaleNorm)
      .export_values();

  /*
   * Plasma model classes
   */
  py::class_<libskit::PlasmaModel>(m, "PlasmaModel",
      R"pbdoc(
          Abstract base class for plasma models.

          This class defines the interface for all plasma models in LIBSkit.
          It provides methods for calculating plasma properties and spectral
          characteristics.
      )pbdoc")
      .def("get_fine_grid_spacing",
           &libskit::PlasmaModel::get_fine_grid_spacing,
           R"pbdoc(
               Get the fine grid spacing determined by the narrowest emission line.

               Returns the wavelength step size (in nm) needed to adequately
               sample all emission lines: min_voigt_fwhm / fine_samples_per_fwhm.

               Parameters
               ----------
               wavelengths : numpy.ndarray
                   Wavelength range to consider for determining the narrowest line.

               Returns
               -------
               float
                   Fine grid spacing in nm.

               Example
               -------
               >>> dl = model.get_fine_grid_spacing(wavelengths)
               >>> print(f"Fine grid spacing: {dl:.5f} nm")
           )pbdoc")
      .def("build_fine_wavelength_grid",
           &libskit::PlasmaModel::build_fine_wavelength_grid,
           R"pbdoc(
               Build a uniform fine wavelength grid for accurate spectral computation.

               Constructs a grid spanning the same range as the input wavelengths,
               with spacing determined by the narrowest emission line's FWHM.
               If the input grid is already fine enough, the spacing matches
               the input.

               Parameters
               ----------
               wavelengths : numpy.ndarray
                   Wavelength range defining the desired extent.

               Returns
               -------
               numpy.ndarray
                   Fine uniform wavelength grid.

               Example
               -------
               >>> wl_fine = model.build_fine_wavelength_grid(wavelengths)
               >>> irr_fine = model.calculate_irradiance(wl_fine)
           )pbdoc")
      .def("get_minimum_voigt_fwhm",
           &libskit::PlasmaModel::get_minimum_voigt_fwhm,
           R"pbdoc(
               Get the minimum Voigt FWHM across all emission lines in the range.

               Parameters
               ----------
               wavelengths : numpy.ndarray
                   Wavelength range to consider.

               Returns
               -------
               float
                   Minimum Voigt FWHM in nm.
           )pbdoc");
  py::class_<libskit::LTEModel, libskit::PlasmaModel>(
      m, "LTEModel",
      R"pbdoc(
          Local Thermodynamic Equilibrium (LTE) plasma model.

          This class models a plasma in local thermodynamic equilibrium,
          characterized by its element mass ratios, temperature, electron
          density, and physical length. It provides methods for calculating
          population fractions, emission line intensities, and radiative
          transfer.

          Parameters
          ----------
          element_ratio : dict
              Dictionary mapping element symbols (str) to mass ratios (float).
              The ratios should sum to 1.0 for proper normalization.
              Example: {"Fe": 0.7, "Si": 0.2, "Ca": 0.1}
          temperature : float
              Plasma temperature in Kelvin. Typical LIBS values range from
              8,000 K to 20,000 K.
          electron_density : float
              Electron number density in m^-3. Typical LIBS values range from
              1e22 to 1e24 m^-3.
          length : float
              Physical length of the plasma column in meters. Typical LIBS
              values range from 1e-4 to 1e-2 m.

          Attributes
          ----------
          temperature : float
              Plasma temperature in Kelvin.
          electron_density : float
              Electron number density in m^-3.
          length : float
              Physical length of plasma column in meters.
          element_ratio : dict
              Dictionary of element mass ratios.

          Methods
          -------
          calculate_irradiance(wavelengths)
              Calculate spectral irradiance at given wavelengths.
          get_lines()
              Get emission lines with wavelength, intensity, and FWHM.
          calculate_element_fractions()
              Calculate element number density fractions.
          get_population_fractions()
              Get ionization state populations for each element.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>> model = libskit.LTEModel(
          ...     {"Fe": 0.7, "Ca": 0.3},
          ...     temperature=10000.0,
          ...     electron_density=1e23,
          ...     length=1e-3
          ... )
          >>> wavelengths = np.linspace(200, 900, 1000)
          >>> irradiance = model.calculate_irradiance(wavelengths)
          >>> print(f"Temperature: {model.temperature} K")
          Temperature: 10000.0 K

          Notes
          -----
          The LTE assumption requires that the plasma is sufficiently dense
          and collisionally dominated. The model uses the Saha-Boltzmann
          equations to calculate ionization equilibrium and level populations.
      )pbdoc")
      .def(py::init<std::map<std::string, double>, double, double, double>())
      .def_property("element_ratio",
          [](libskit::LTEModel& model) -> std::map<std::string, double> {
              return model.element_ratio;  // Return copy for reading
          },
          [](libskit::LTEModel& model, const std::map<std::string, double>& new_ratios) {
              model.element_ratio = new_ratios;  // Allow full replacement
          })
      // Add dict-like methods for element_ratio access
      .def("__getitem__",
          [](libskit::LTEModel& model, const std::string& element) -> double {
              auto it = model.element_ratio.find(element);
              if (it == model.element_ratio.end()) {
                  throw py::key_error("Element '" + element + "' not found");
              }
              return it->second;
          })
      .def("__setitem__",
          [](libskit::LTEModel& model, const std::string& element, double value) {
              model.element_ratio[element] = value;
          })
      .def("__contains__",
          [](libskit::LTEModel& model, const std::string& element) -> bool {
              return model.element_ratio.find(element) != model.element_ratio.end();
          })
      .def("keys",
          [](libskit::LTEModel& model) {
              py::list keys;
              for (const auto& [key, value] : model.element_ratio) {
                  keys.append(key);
              }
              return keys;
          })
      .def("items",
          [](libskit::LTEModel& model) {
              py::list items;
              for (const auto& [key, value] : model.element_ratio) {
                  items.append(py::make_tuple(key, value));
              }
              return items;
          })
      .def("set_element_ratio",
          [](libskit::LTEModel& model, const std::string& element, double value) {
              model.element_ratio[element] = value;
          },
          "Set the ratio for a specific element")
      .def_readwrite("temperature", &libskit::LTEModel::temperature)
      .def_readwrite("electron_density",
                     &libskit::LTEModel::electron_density)
      .def_readwrite("length", &libskit::LTEModel::length)
      .def_readwrite("include_continuum",
                     &libskit::LTEModel::include_continuum,
                     R"pbdoc(
          Whether to include continuum emission (bremsstrahlung + recombination).

          When True, free-free (bremsstrahlung) and free-bound (recombination)
          continuum radiation is added to the emission coefficient before solving
          the radiative transfer equation. Uses the hydrogen approximation
          (Gaunt factor = 1).

          Default is False for backward compatibility.

          Example
          -------
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> model.include_continuum = True
          >>> irradiance = model.calculate_irradiance(wavelengths)
      )pbdoc")
      .def("calculate_continuum_emission",
           &libskit::LTEModel::calculate_continuum_emission,
           py::arg("wavelengths"),
           R"pbdoc(
          Calculate free-free and free-bound continuum emission separately.

          Computes bremsstrahlung (free-free) and recombination (free-bound)
          continuum emission in the hydrogen approximation (Gaunt factor = 1).
          Reference: Noll (2012), Eqs. 10.14-10.18.

          Parameters
          ----------
          wavelengths : numpy.ndarray
              Wavelength grid in nanometers.

          Returns
          -------
          tuple[numpy.ndarray, numpy.ndarray]
              A tuple (free_free, free_bound), each an array of emission
              coefficients in W/m⁴ at the given wavelengths.

          Example
          -------
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> ff, fb = model.calculate_continuum_emission(wavelengths)
      )pbdoc")
      .def("__repr__",
           [](const libskit::LTEModel &a) {
             std::stringstream ss;
             ss << "LTEModel(";
             ss << "T = " << a.temperature << ", ";
             ss << "n_e = " << std::scientific << std::setprecision(5)
                << a.electron_density << ", ";
             ss << "L = " << std::fixed << a.length << ", ";

             ss << "elements = {";
             for (const auto &[element, ratio] : a.element_ratio) {
               ss << element << ": " << ratio << ", ";
             };

             // Remove the trailing ", " from the serialized element-ratio map.
             auto temp = ss.str();
             temp.pop_back();
             temp.pop_back();

             std::stringstream ss2;
             ss2 << temp << "}";

             ss2 << ")";
             return ss2.str();
           })
      .def("calculate_irradiance",
           &libskit::LTEModel::calculate_irradiance)
      .def("calculate_element_fractions",
           &libskit::LTEModel::_calculate_element_fractions)
      .def("get_population_fractions",
           &libskit::LTEModel::get_population_fractions)
      .def("get_lines",
           &libskit::LTEModel::get_lines,
           "Get emission lines with ion, wavelength, peak emission, FWHM, and accuracy fraction")
      .def("get_parameter_names", &libskit::LTEModel::get_parameter_names,
           "Get list of all parameter names (temperature, electron_density, length, elements)")
      .def("get_parameter", &libskit::LTEModel::get_parameter,
           py::return_value_policy::reference,
           "Get a parameter value by name")
      .def("set_parameter", &libskit::LTEModel::set_parameter,
           "Set a parameter value by name");

    /*
     * LTELayers class - Multi-layer plasma model.
     *
     * Each layer has its own composition and plasma state.
     */
  py::class_<libskit::LTELayers, libskit::PlasmaModel>(
      m, "LTELayers",
      R"pbdoc(
          Multi-layer plasma model for spatially varying plasmas.

          This class represents a plasma with multiple discrete layers, each
          having its own elemental composition, temperature, electron density,
          and length. It is useful for modeling:
          - Inhomogeneous plasmas with temperature gradients
          - Core-shell plasma structures

          Parameters
          ----------
          element_ratios : list of dict
              List of element ratio dictionaries, one per layer.
          temperatures : list of float
              List of temperatures in Kelvin, one per layer.
          electron_densities : list of float
              List of electron densities in m^-3, one per layer.
          lengths : list of float
              List of physical lengths in meters, one per layer.

          Example
          -------
          >>> # Two-layer plasma with different conditions
          >>> layers = libskit.LTELayers(
          ...     [{"Fe": 1.0}, {"Fe": 1.0}],  # Element ratios per layer
          ...     [12000.0, 8000.0],             # Temperatures (K)
          ...     [1e24, 1e23],                  # Electron densities (m^-3)
          ...     [1e-4, 1e-3]                   # Lengths (m)
          ... )
          >>> len(layers)  # Number of layers
          2
          >>> layers[0].temperature  # Access first layer
          12000.0
      )pbdoc")
      .def(py::init([](py::list element_ratios, py::list temperatures,
                      py::list electron_densities, py::list lengths) {
          // Convert Python lists to the C++ vectors expected by LTELayers.
          std::vector<std::map<std::string, double>> cpp_ratios;
          std::vector<double> cpp_temps;
          std::vector<double> cpp_densities;
          std::vector<double> cpp_lengths;

          // Process per-layer element-ratio dictionaries.
          for (auto item : element_ratios) {
              std::map<std::string, double> ratio_map = item.cast<std::map<std::string, double>>();
              cpp_ratios.push_back(ratio_map);
          }

          // Process the remaining per-layer plasma parameters.
          for (auto item : temperatures) {
              cpp_temps.push_back(item.cast<double>());
          }
          for (auto item : electron_densities) {
              cpp_densities.push_back(item.cast<double>());
          }
          for (auto item : lengths) {
              cpp_lengths.push_back(item.cast<double>());
          }

          return libskit::LTELayers(cpp_ratios, cpp_temps, cpp_densities, cpp_lengths);
      }))
      .def("get_parameter", &libskit::LTELayers::get_parameter,
           py::return_value_policy::reference)
      .def("set_parameter", &libskit::LTELayers::set_parameter)
      .def("get_parameter_names", &libskit::LTELayers::get_parameter_names)
      .def("calculate_irradiance", &libskit::LTELayers::calculate_irradiance)
      .def("__repr__",
           [](const libskit::LTELayers &layers) {
             std::stringstream ss;
             ss << "LTELayers(" << layers.lte_layers.size() << " layers)";
             return ss.str();
           })
      .def("__len__", [](const libskit::LTELayers& layers) {
          return layers.lte_layers.size();
      })
      .def("__getitem__",
          [](const libskit::LTELayers& layers, size_t index) -> const libskit::LTEModel& {
              if (index >= layers.lte_layers.size()) {
                  throw py::index_error("Layer index out of range");
              }
              return layers.lte_layers[index];
          }, py::return_value_policy::reference_internal);

    /*
     * LTESharedLayers class - Multi-layer plasma model with shared composition.
     */
  py::class_<libskit::LTESharedLayers, libskit::PlasmaModel>(
      m, "LTESharedLayers",
      R"pbdoc(
          Multi-layer LTE plasma model with one shared elemental composition.

          Like :class:`LTELayers`, each layer has its own temperature, electron
          density, and length, but the elemental mass-ratio composition is
          shared across all layers. This is the natural model for inverse
          problems where the only spatially varying quantities are the plasma
          state.

          Parameters
          ----------
          element_ratio : dict
              Single composition dictionary (e.g. ``{"Fe": 1.0}``) used for
              every layer.
          temperatures : list of float
              Per-layer temperatures in Kelvin.
          electron_densities : list of float
              Per-layer electron densities in m^-3.
          lengths : list of float
              Per-layer physical lengths in meters.

          Example
          -------
          >>> layers = libskit.LTESharedLayers(
          ...     {"Fe": 1.0},
          ...     [12000.0, 8000.0],
          ...     [1e24, 1e23],
          ...     [1e-4, 1e-3],
          ... )
          >>> len(layers)
          2
      )pbdoc")
      .def(py::init([](py::dict element_ratio, py::list temperatures,
                       py::list electron_densities, py::list lengths) {
          std::map<std::string, double> cpp_ratio =
              element_ratio.cast<std::map<std::string, double>>();
          std::vector<double> cpp_temps;
          std::vector<double> cpp_densities;
          std::vector<double> cpp_lengths;
          for (auto item : temperatures)
              cpp_temps.push_back(item.cast<double>());
          for (auto item : electron_densities)
              cpp_densities.push_back(item.cast<double>());
          for (auto item : lengths)
              cpp_lengths.push_back(item.cast<double>());
          return libskit::LTESharedLayers(
              cpp_ratio, cpp_temps, cpp_densities, cpp_lengths);
      }))
      .def("get_parameter", &libskit::LTESharedLayers::get_parameter,
           py::return_value_policy::reference)
      .def("set_parameter", &libskit::LTESharedLayers::set_parameter)
      .def("get_parameter_names",
           &libskit::LTESharedLayers::get_parameter_names)
      .def("calculate_irradiance",
           &libskit::LTESharedLayers::calculate_irradiance)
      .def("__repr__",
           [](const libskit::LTESharedLayers &m) {
             std::stringstream ss;
             ss << "LTESharedLayers(" << m.n_layers() << " layers, "
                << m.element_ratio.size() << " elements)";
             return ss.str();
           })
      .def("__len__", [](const libskit::LTESharedLayers &m) {
          return m.n_layers();
      });

  /*
   * LTETimeIntegrated class - Time-integrated plasma model
   */
  py::class_<libskit::LTETimeIntegrated, libskit::PlasmaModel>(
      m, "LTETimeIntegrated",
      R"pbdoc(
          Time-integrated plasma model.

          Models the time evolution of a LIBS plasma where temperature
          and electron density decay asymptotically toward final values,
          and the plasma length grows with saturation:

            T(t)  = T_f  + (T_0  - T_f)  * exp(-t / tau_T)
            ne(t) = ne_f + (ne_0 - ne_f) * exp(-t / tau_ne)
            L(t)  = L_f  + (L_0  - L_f)  * exp(-t / tau_L)

          The resulting spectrum is integrated over the specified time window.

          Parameters
          ----------
          element_ratio : dict
              Dictionary mapping element symbols to mass ratios.
          start_temperature : float
              Initial temperature T_0 at t=0 in Kelvin.
          end_temperature : float
              Asymptotic final temperature T_f in Kelvin.
          tau_temperature : float
              Temperature decay time constant in seconds.
          start_electron_density : float
              Initial electron density ne_0 at t=0 in m^-3.
          end_electron_density : float
              Asymptotic final electron density ne_f in m^-3.
          tau_electron_density : float
              Electron density decay time constant in seconds.
          start_length : float
              Initial plasma length L_0 at t=0 in meters.
          end_length : float
              Asymptotic final plasma length L_f in meters.
          tau_length : float
              Length evolution time constant in seconds.
          integration_time : float
              Total integration time in seconds.
          integration_points : int, optional
              Number of time points for the weighted sum (default: 5).

          Example
          -------
          >>> model = libskit.LTETimeIntegrated(
          ...     {"Fe": 1.0},
          ...     start_temperature=13000.0,
          ...     end_temperature=6500.0,
          ...     tau_temperature=195e-9,
          ...     start_electron_density=4e23,
          ...     end_electron_density=0.8e23,
          ...     tau_electron_density=90e-9,
          ...     start_length=0.5e-3,
          ...     end_length=3.0e-3,
          ...     tau_length=109e-9,
          ...     integration_time=1990e-9,
          ...     integration_points=10
          ... )
      )pbdoc")
      .def(py::init<std::map<std::string, double>, double, double, double, double, double, double, double, double, double, double, int>(),
           py::arg("element_ratio"), py::arg("start_temperature"), py::arg("end_temperature"),
           py::arg("tau_temperature"), py::arg("start_electron_density"), py::arg("end_electron_density"),
           py::arg("tau_electron_density"), py::arg("start_length"), py::arg("end_length"),
           py::arg("tau_length"), py::arg("integration_time"),
           py::arg("integration_points") = 5)
      .def("get_parameter", &libskit::LTETimeIntegrated::get_parameter,
           py::return_value_policy::reference)
      .def("set_parameter", &libskit::LTETimeIntegrated::set_parameter)
      .def("get_parameter_names", &libskit::LTETimeIntegrated::get_parameter_names)
      .def("calculate_irradiance", &libskit::LTETimeIntegrated::calculate_irradiance)
      .def_readonly("start_temperature", &libskit::LTETimeIntegrated::start_temperature)
      .def_readonly("end_temperature", &libskit::LTETimeIntegrated::end_temperature)
      .def_readonly("tau_temperature", &libskit::LTETimeIntegrated::tau_temperature)
      .def_readonly("start_electron_density", &libskit::LTETimeIntegrated::start_electron_density)
      .def_readonly("end_electron_density", &libskit::LTETimeIntegrated::end_electron_density)
      .def_readonly("tau_electron_density", &libskit::LTETimeIntegrated::tau_electron_density)
      .def_readonly("start_length", &libskit::LTETimeIntegrated::start_length)
      .def_readonly("end_length", &libskit::LTETimeIntegrated::end_length)
      .def_readonly("tau_length", &libskit::LTETimeIntegrated::tau_length)
      .def_readonly("integration_time", &libskit::LTETimeIntegrated::integration_time)
      .def_readonly("integration_points", &libskit::LTETimeIntegrated::integration_points)
      .def("__repr__",
           [](const libskit::LTETimeIntegrated &model) {
             std::stringstream ss;
             ss << "LTETimeIntegrated(T₀: " << model.start_temperature
                << "K → " << model.end_temperature << "K, τ_T: "
                << model.tau_temperature << "s, ne₀: "
                << model.start_electron_density/1e18 << "e18 → "
                << model.end_electron_density/1e18 << "e18 m⁻³, τ_ne: "
                << model.tau_electron_density << "s, t_int: " << model.integration_time << "s)";
             return ss.str();
           });

  /*
   * LTETwoTime class - Two-time surrogate of a time-integrated plasma
   */
    py::class_<libskit::LTETwoTime, libskit::PlasmaModel>(
      m, "LTETwoTime",
      R"pbdoc(
          Two-time surrogate plasma model.

          Approximates a time-integrated LIBS plasma as a weighted sum of
          two static LTE plasmas (early + late) sharing the same elemental
          composition:

            I(λ) = w · I_early(λ) + (1 − w) · I_late(λ)

          where ``w = weight_early`` ∈ [0, 1] and the late weight is
          implicitly ``1 - w``.

          The late state is not stored directly; it is derived from the
          early state and three non-negative "delta" parameters that
          encode the physically expected direction of plasma evolution:

            late_temperature      = early_temperature      - delta_temperature
            late_electron_density = early_electron_density - delta_electron_density
            late_length            = early_length            + delta_length

          Only early_*, delta_* and weight_early are fittable parameters
          (see ``get_parameter_names()``); late_* remain readable as
          computed, read-only attributes. Use :meth:`from_late_state` to
          construct a model directly from absolute early/late states.

          Parameters
          ----------
          element_ratio : dict
              Element symbol → mass ratio (shared between both times).
          early_temperature, early_electron_density, early_length : float
              Plasma parameters of the early state.
          delta_temperature, delta_electron_density, delta_length : float
              Non-negative early-to-late deltas (see above).
          weight_early : float
              Weight of the early contribution in [0, 1].
      )pbdoc")
      .def(py::init<std::map<std::string, double>, double, double, double,
                    double, double, double, double>(),
           py::arg("element_ratio"),
           py::arg("early_temperature"), py::arg("early_electron_density"),
           py::arg("early_length"),
           py::arg("delta_temperature"), py::arg("delta_electron_density"),
           py::arg("delta_length"),
           py::arg("weight_early") = 0.5)
     .def_static("from_late_state", &libskit::LTETwoTime::from_late_state,
           py::arg("element_ratio"),
           py::arg("early_temperature"), py::arg("early_electron_density"),
           py::arg("early_length"),
           py::arg("late_temperature"), py::arg("late_electron_density"),
           py::arg("late_length"),
           py::arg("weight_early") = 0.5,
           R"pbdoc(
               Construct an LTETwoTime model directly from early and late
               plasma states, converting late_* into the internal delta_*
               representation.
           )pbdoc")
     .def("get_parameter", &libskit::LTETwoTime::get_parameter,
           py::return_value_policy::reference)
     .def("set_parameter", &libskit::LTETwoTime::set_parameter)
     .def("get_parameter_names", &libskit::LTETwoTime::get_parameter_names)
     .def("calculate_irradiance", &libskit::LTETwoTime::calculate_irradiance)
     .def_readwrite("early_temperature", &libskit::LTETwoTime::early_temperature)
     .def_readwrite("early_electron_density", &libskit::LTETwoTime::early_electron_density)
     .def_readwrite("early_length", &libskit::LTETwoTime::early_length)
     .def_readwrite("delta_temperature", &libskit::LTETwoTime::delta_temperature)
     .def_readwrite("delta_electron_density", &libskit::LTETwoTime::delta_electron_density)
     .def_readwrite("delta_length", &libskit::LTETwoTime::delta_length)
     .def_readwrite("weight_early", &libskit::LTETwoTime::weight_early)
     .def_property_readonly("late_temperature", &libskit::LTETwoTime::late_temperature)
     .def_property_readonly("late_electron_density", &libskit::LTETwoTime::late_electron_density)
     .def_property_readonly("late_length", &libskit::LTETwoTime::late_length)
      .def("__repr__",
         [](const libskit::LTETwoTime &model) {
             std::stringstream ss;
             ss << "LTETwoTime(T: " << model.early_temperature
                << "K → " << model.late_temperature()
                << "K, ne: " << model.early_electron_density / 1e18
                << "e18 → " << model.late_electron_density() / 1e18
                << "e18 m⁻³, L: " << model.early_length
                << "m → " << model.late_length()
                << "m, w_early: " << model.weight_early << ")";
             return ss.str();
           });


  /*
   * Experiment class
   */
  py::class_<libskit::Experiment>(m, "Experiment",
      R"pbdoc(
          Experimental configuration for gradient-based optimization.

          This class encapsulates all components needed for computing gradients
          of loss functions with respect to plasma model parameters. It combines:
          - The plasma model (with parameters to be optimized)
          - The instrument model (for forward simulation)
          - The experimental data (target spectrum)
          - The loss function type

          Parameters
          ----------
          loss_type : LossType
              Type of loss function to use (L2Loss or IntensityWeightedLoss).
          plasma_model : PlasmaModel
              Initial plasma model with parameters to optimize.
          instrument : Instrument
              Spectrometer model for simulating measurements.
          true_spectrum : Spectrum
              Target experimental spectrum to fit.

          Attributes
          ----------
          loss_type : LossType
              The loss function type.
          plasma_model : PlasmaModel
              The plasma model being optimized.
          instrument : Instrument
              The instrument model.
          true_spectrum : Spectrum
              The target spectrum.

          Example
          -------
          >>> import libskit
          >>> import numpy as np
          >>>
          >>> # Create components
          >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
          >>> instrument = libskit.GenericSpectrometer(0.05, 0.01)
          >>> wavelengths = np.linspace(200, 900, 5000)
          >>> intensities = instrument.calculate_spectrum(wavelengths, model)
          >>> spectrum = libskit.Spectrum(wavelengths, intensities)
          >>>
          >>> # Create experiment
          >>> experiment = libskit.Experiment(
          ...     libskit.LossType.L2Loss,
          ...     model,
          ...     instrument,
          ...     spectrum
          ... )
          >>>
          >>> # Calculate loss and gradients
          >>> loss_value = libskit.loss(experiment)
          >>> gradients = libskit.get_gradients(experiment)
      )pbdoc")
      .def(py::init<enum libskit::LossType, const libskit::PlasmaModel&,
                    const libskit::Instrument&, libskit::Spectrum,
                    enum libskit::NormalizationType>(),
           py::arg("loss_type"), py::arg("plasma_model"),
           py::arg("instrument"), py::arg("true_spectrum"),
           py::arg("normalization_type") = libskit::NormalizationType::MaxNorm)
      .def_readwrite("loss_type", &libskit::Experiment::loss_type)
      .def_readwrite("normalization_type", &libskit::Experiment::normalization_type)
      .def_property("plasma_model",
          [](libskit::Experiment& exp) -> libskit::PlasmaModel& {
              return exp.get_plasma_model();
          },
          [](libskit::Experiment& exp, const libskit::PlasmaModel& model) {
              // This is a simplified setter - in practice you might want to clone
              exp.plasma_model_ptr = model.clone();
          })
      .def_property("instrument",
          [](libskit::Experiment& exp) -> libskit::Instrument& {
              return exp.get_instrument();
          },
          [](libskit::Experiment& exp, const libskit::Instrument& instr) {
              exp.instrument_ptr = instr.clone();
          })
      .def_readwrite("true_spectrum", &libskit::Experiment::true_spectrum)
      .def_readwrite("scale_factor", &libskit::Experiment::scale_factor)
      .def_property("loss_weights",
          [](const libskit::Experiment& exp) {
              return Eigen::VectorXd(exp.loss_weights);
          },
          [](libskit::Experiment& exp, const Eigen::VectorXd& w) {
              exp.loss_weights = w;
          },
          "Per-pixel weights for WeightedL2Loss (length = spectrum length, "
          "or empty for uniform weights).")
      .def_readwrite("robust_threshold", &libskit::Experiment::robust_threshold,
          "Robust-loss threshold: the Huber 'delta' or Tukey biweight 'c' "
          "constant (same units as the normalized residuals). Only used "
          "when loss_type is HuberLoss or TukeyBiweightLoss.")
      .def("get_plasma_model", &libskit::Experiment::get_plasma_model,
           py::return_value_policy::reference_internal)
      .def("get_instrument", &libskit::Experiment::get_instrument,
           py::return_value_policy::reference_internal);

  m.def("calculate_irradiance_test", &libskit::calculate_irradiance_test,
        R"pbdoc(
           Reference irradiance calculation used by the test suite.

           Evaluates the analytical Planck blackbody irradiance plus a
           narrow Gaussian emission line on a fixed wavelength grid; the
           Python tests compare the result against ``blackbody_irradiance``
           plus a numerical convolution to detect regressions in the
           wavelength-grid handling.

           Returns:
               numpy.ndarray: Irradiance in W m\u207b\u00b2 nm\u207b\u00b9 sr\u207b\u00b9.
       )pbdoc");
  m.def("optimal_scale_factor", &libskit::optimal_scale_factor,
        R"pbdoc(
           Compute the analytically optimal scale factor between two spectra.

           Finds c that minimizes ||y_sim / c - y_ref||² (i.e. the scale
           factor such that y_sim / c ≈ y_ref):
             c_opt = (y_sim · y_sim) / (y_sim · y_ref)

           Args:
               y_sim: Simulated spectrum intensities (numpy array).
               y_ref: Reference (experimental) spectrum intensities (numpy array).

           Returns:
               float: Optimal scale factor c.
       )pbdoc");
  m.def("get_gradients", &libskit::get_gradients,
        R"pbdoc(
           Calculate gradients using Enzyme reverse-mode AD (recommended default).

           Computes all gradients of the loss function w.r.t. plasma model
           same result in a single reverse-mode pass regardless of parameter
           count.

           Args:
               experiment: An Experiment object (double precision).

           Returns:
               dict: Mapping of parameter names to gradient values.
       )pbdoc");
  m.def("compute_loss", &libskit::compute_loss,
        R"pbdoc(
           Compute loss using the Enzyme numerical pipeline.

           Useful for verifying that the reimplemented pipeline
           matches the standard loss() function.

           Args:
               experiment: An Experiment object (double precision).

           Returns:
               float: The computed loss value.
       )pbdoc");
  m.def("get_jacobian", &libskit::get_jacobian,
        R"pbdoc(
           Compute the Jacobian matrix of the forward model output w.r.t. parameters.

           Returns the matrix J where J[i,j] = dI(λ_i) / d(param_j) using
           n_params forward-mode AD passes (one column per parameter).
           Cost is O(n_params × cost_of_one_forward_pass), which is much faster
           than a reverse-mode row-loop whenever n_wavelengths >> n_params
           (the typical LIBS case).

           Args:
               experiment: An Experiment object.

           Returns:
               numpy.ndarray: Shape (n_wavelengths, n_params).

           Example:
               >>> J = libskit.get_jacobian(experiment)
               >>> print(J.shape)  # (5000, 13) for 5000 wavelengths, 13 params
       )pbdoc");
  m.def("get_aki_gradient", &libskit::get_aki_gradient,
        R"pbdoc(
           Gradient of the loss w.r.t. every per-line Einstein coefficient A_ji.

           Returns a numpy array of length ``n_aki`` indexed in the database
           iteration order: walk active ion-line groups in build_const_data
           order, then ``.lines`` in order. The same ordering is accepted by
           ``spectrum_vjp_aki``.

           Args:
               experiment: An Experiment object (double precision).

           Returns:
               numpy.ndarray: Shape (n_aki,) with values ∂L/∂A_ji.
       )pbdoc");
  m.def("spectrum_vjp_aki", &libskit::spectrum_vjp_aki,
        R"pbdoc(
           Vector-Jacobian product against the Aki Jacobian:
           returns ``J_A^T @ adjoint`` in one Enzyme reverse pass.

           J_A has shape (n_out, n_aki) with J_A[i, l] = ∂y_sim_i / ∂A_l.

           Args:
               experiment: An Experiment object.
               adjoint: Adjoint covector of length n_out (output spectrum grid).

           Returns:
               numpy.ndarray: Shape (n_aki,) with (J_A)^T @ adjoint.
       )pbdoc");
  m.attr("ENZYME_ENABLED") = true;
  m.def("loss", &libskit::loss_func,
        R"pbdoc(
           Calculate the loss function value for an experiment.

           Takes an Experiment object and returns the scalar loss value
           computed using the experiment's specified loss function type
           (L2Loss or IntensityWeightedLoss).

           Args:
               experiment: An Experiment object containing the plasma model,
                          instrument, target spectrum, and loss type.

           Returns:
               float: The computed loss value.

           Example:
               >>> loss_value = libskit.loss(experiment)
       )pbdoc");
  m.def("blackbody_irradiance", &libskit::blackbody_radiation<double>,
        R"pbdoc(
           Spectral radiance of a blackbody at temperature ``T``.

           Evaluates Planck's law for the supplied wavelength grid,
           returning the per-wavelength spectral radiance in SI units.

           Args:
               wavelengths: Wavelengths in nm (numpy array).
               T: Blackbody temperature in K.

           Returns:
               numpy.ndarray: Spectral radiance B\u03bb(T) in W m\u207b\u00b3 sr\u207b\u00b9.
       )pbdoc");

  m.def("calculate_partition_function", &libskit::calculate_partition_function<double>,
        R"pbdoc(
           Calculate the partition function for a given ion and temperature.

           Uses the full energy level data from the NIST database.
           U(T) = Σ g_i * exp(-E_i / (k_B * T))

           Args:
               ion (str): The ion species, e.g. "Ca I", "Fe II"
               temperature (float): Temperature in Kelvin

           Returns:
               float: The partition function value

           Example:
               >>> libskit.calculate_partition_function("Ca I", 10000.0)
       )pbdoc");

  m.def("calculate_saha_function", &libskit::calculate_saha_function<double>,
        R"pbdoc(
           Calculate the Saha function for a given element and ionization level.

           Args:
               element (str): Element symbol, e.g. "Ca", "Fe"
               ionization_level (int): Ionization level (0 = neutral to singly ionized)
               temperature (float): Temperature in Kelvin

           Returns:
               float: The Saha function value in m^-3

           Example:
               >>> libskit.calculate_saha_function("Ca", 0, 10000.0)
       )pbdoc");

  // Database path management functions
  m.def("set_database_path", &libskit::set_database_path,
        R"pbdoc(
           Set the path to the atomic databases directory.

           Args:
               path (str): Path to directory containing the CSV database files
                          (lines_processed_all.csv, ionization_energies_processed.csv,
                           levels_processed.csv)

           Example:
               libskit.set_database_path("/path/to/my/databases")
       )pbdoc");

  m.def("get_database_path", &libskit::get_database_path,
        R"pbdoc(
           Get the current path to the atomic databases directory.

           Returns:
               str: Current database path
       )pbdoc");

  m.def("load_databases", &libskit::load_databases, py::arg("verbose") = 1,
        R"pbdoc(
           Manually trigger loading of atomic databases.

           This is usually called automatically when needed, but can be called
           explicitly to preload databases or check for loading errors.

           Parameters:
               verbose: Verbosity level (0=silent, 1=normal). Default: 1
       )pbdoc");

  m.def("reload_databases", &libskit::reload_databases,
        R"pbdoc(
           Force reload of atomic databases from current database path.

           Useful after changing the database path with set_database_path().
       )pbdoc");

  // Aki modification API for Monte Carlo perturbation studies
  m.def("set_aki_multipliers", &libskit::set_aki_multipliers,
        R"pbdoc(
           Multiply each emission line's Einstein A coefficient by a factor.

           For each ion key in the dictionary, the corresponding list of
           factors is applied element-wise to the lines of that ion.  The
           factor list must have the same length as the number of lines
           for that ion.

           Use ``reset_aki()`` to restore the original values.

           Args:
               multipliers (dict[str, list[float]]): Mapping from ion name
                   (e.g. ``"Ca II"``) to a list of multiplicative factors.

           Raises:
               RuntimeError: If an ion name is not found or the factor list
                   has the wrong length.

           Example:
               >>> factors = {"Ca II": [1.05, 0.97, 1.02, ...]}
               >>> libskit.set_aki_multipliers(factors)
       )pbdoc");

  m.def("reset_aki", &libskit::reset_aki,
        R"pbdoc(
           Restore all Einstein A coefficients to their original database values.

           Undoes any modifications made by ``set_aki_multipliers()``.
       )pbdoc");

  m.def("get_aki_accuracies", &libskit::get_aki_accuracies,
        R"pbdoc(
           Get NIST accuracy fractions for all Einstein A coefficients.

           Returns a dict mapping ion names to lists of accuracy fractions,
           one per emission line in the database (in the same order expected
           by ``set_aki_multipliers``).

           Returns:
               dict[str, list[float]]: {ion: [acc_fraction, ...]}
       )pbdoc");

  m.def("get_aki_values", &libskit::get_aki_values,
        R"pbdoc(
           Get the current in-memory A_ji values for every emission line.

           Returns a dict mapping ion names to lists of Einstein A
           coefficients (s^-1), in the same per-ion ordering as
           ``set_aki_multipliers`` and ``get_aki_accuracies``.  Reflects any
           active ``set_aki_multipliers`` override.

           Returns:
               dict[str, list[float]]: {ion: [A_ji, ...]}
       )pbdoc");

  // Stark-FWHM modification API for Monte Carlo perturbation studies
  m.def("set_stark_multipliers", &libskit::set_stark_multipliers,
        R"pbdoc(
           Set per-line multipliers on the Stark FWHM (Lorentzian width in
           the Voigt profile).  Absolute assignment: ``line.stark_mult :=
           factor`` (default 1.0).  Stark *shift* is untouched.

           For each ion key in the dictionary, the corresponding list of
           factors is stored element-wise for the lines of that ion.  The
           factor list must have the same length as the number of lines
           for that ion.  Use ``reset_stark()`` to restore all multipliers
           back to 1.0.

           Args:
               multipliers (dict[str, list[float]]): {ion: [factor, ...]}.

           Raises:
               RuntimeError: If an ion name is not found or the factor list
                   has the wrong length.
       )pbdoc");

  m.def("reset_stark", &libskit::reset_stark,
        R"pbdoc(
           Restore all Stark FWHM multipliers to 1.0.

           Undoes any modifications made by ``set_stark_multipliers()``.
       )pbdoc");

  m.def("get_stark_multipliers", &libskit::get_stark_multipliers,
        R"pbdoc(
           Get the current per-line Stark FWHM multipliers, keyed by ion.

           Returns:
               dict[str, list[float]]: {ion: [stark_mult, ...]}
       )pbdoc");

  m.def("get_stark_gradient", &libskit::get_stark_gradient,
        R"pbdoc(
           Gradient of the loss w.r.t. every per-line Stark FWHM multiplier
           via Enzyme reverse-mode AD.

           Returns a numpy array of length ``n_aki`` (same layout as
           ``get_aki_gradient``).

           Args:
               experiment: An Experiment object (double precision).

           Returns:
               numpy.ndarray: Shape (n_aki,) with values ∂L/∂m_l.
       )pbdoc");

  m.def("spectrum_vjp_stark", &libskit::spectrum_vjp_stark,
        R"pbdoc(
           Vector-Jacobian product against the Stark-multiplier Jacobian:
           returns ``J_m^T @ adjoint`` in one Enzyme reverse pass, where
           J_m[i, l] = ∂y_sim_i / ∂m_l.

           Args:
               experiment: An Experiment object.
               adjoint: Adjoint covector of length n_out.

           Returns:
               numpy.ndarray: Shape (n_aki,) with (J_m)^T @ adjoint.
       )pbdoc");

  // Accuracy management
  m.def("set_accuracy", &libskit::set_accuracy,
        R"pbdoc(
           Set the accuracy threshold as a fraction of the maximum peak height.

           This single parameter controls both:
           - Which emission lines are included (lines whose peak height is
             below accuracy * max_peak_height are discarded)
           - Where broadening wings are truncated (broadening stops when the
             spectral density drops below accuracy * max_peak_height)

           Args:
               accuracy (float): Fraction of max peak height (default: 1e-5)

           Example:
               libskit.set_accuracy(1e-3)  # Noise floor at 0.1% of tallest peak
       )pbdoc");

  m.def("get_accuracy", &libskit::get_accuracy,
        R"pbdoc(
           Get the current accuracy threshold.

           Returns:
               float: Fraction of max peak height used as noise floor
       )pbdoc");

  m.def("set_fine_samples_per_fwhm", &libskit::set_fine_samples_per_fwhm,
        R"pbdoc(
           Set the number of fine grid samples per narrowest line FWHM.

           Controls the resolution of the internal fine grid used for
           FFT-based instrumental broadening. Higher values give more
           accurate convolution at the cost of computation time.

           Args:
               samples (float): Number of samples per FWHM (default: 5). Values < 1 undersample the narrowest line.

           Example:
               libskit.set_fine_samples_per_fwhm(10)  # Higher accuracy
       )pbdoc");

  m.def("get_fine_samples_per_fwhm", &libskit::get_fine_samples_per_fwhm,
        R"pbdoc(
           Get the current number of fine grid samples per narrowest line FWHM.

           Returns:
               float: Current fine grid samples per FWHM
       )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
