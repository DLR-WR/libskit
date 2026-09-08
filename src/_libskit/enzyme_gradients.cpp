/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file enzyme_gradients.cpp
 * @brief Enzyme reverse-mode AD implementation for libskit loss gradients.
 *
 * All string operations (to_roman, ion name construction, map lookups by
 * string key) are performed OUTSIDE the Enzyme-differentiated function.
 * The differentiated function receives only numeric arrays and indices.
 *
 * Compile with:
 *   clang++-18 -fplugin=ClangEnzyme-18.so -O2
 *              -mllvm -enzyme-globals-default-inactive=1
 */

#include "enzyme_gradients.h"
#include "AtomicData/atom_info.h"
#include "AtomicData/constants.h"
#include "AtomicData/elemental_data.h"
#include "Instrument/GenericSpectrometer.h"
#include "Instrument/SlopingSpectrometer.h"
#include "Instrument/ChemCam.h"
#include "Instrument/Aryelle.h"
#include "PlasmaModel/LTEModel.h"
#include "PlasmaModel/LTELayers.h"
#include "PlasmaModel/LTESharedLayers.h"
#include "PlasmaModel/LTETimeIntegrated.h"
#include "PlasmaModel/LTETwoTime.h"
#include "functions.h"
#include "gradients.h"
#include "physics_kernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <numbers>
#include <string>
#include <vector>

// ── Enzyme declarations ──────────────────────────────────────────────
extern int enzyme_dup;
extern int enzyme_const;
extern int enzyme_out;

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);

template <typename return_type, typename... T>
return_type __enzyme_fwddiff(void *, T...);

namespace libskit {

// =====================================================================
// Enum types for model/instrument dispatch inside Enzyme
// =====================================================================
enum EnzymeModelType {
  MODEL_LTE = 0,
  MODEL_LTE_LAYERS = 2,
  MODEL_LTE_TIME_INTEGRATED = 3,
  MODEL_LTE_TWO_TIME = 4,
  MODEL_LTE_SHARED_LAYERS = 5,
};

enum EnzymeInstrumentType {
  INST_GENERIC = 0,
  INST_SLOPING = 1,
  INST_CHEMCAM = 2,
  INST_ARYELLE = 3,
};

// =====================================================================
// All constant data passed to the Enzyme function as enzyme_const.
//
// Inherits the per-element / per-line plasma data layout from
// physics::PlasmaData so the kernel functions in physics_kernel.h work
// directly with an EnzymeConstData* (implicit pointer upcast).
// =====================================================================
struct EnzymeConstData : physics::PlasmaData {
  int n_params;

  // Model/instrument dispatch
  int model_type;      // EnzymeModelType
  int instrument_type;  // EnzymeInstrumentType

  // Wavelength grids (accessed via .data() inside Enzyme)
  std::vector<double> fine_wavelengths;
  long n_fine;
  double fine_dl;
  std::vector<double> output_wavelengths;
  long n_out;
  std::vector<double> reference_irradiance;

  // Per-pixel weights for WeightedL2Loss. Empty ⇒ uniform weights.
  std::vector<double> loss_weights;

  // Pre-computed instrument kernel FFT (GENERIC, SLOPING)
  std::vector<std::complex<double>> kernel_fft;
  long N_fft;

  // Config
  int loss_type;
  int normalization_type;
  bool has_scale_factor;

  // ── LTE_LAYERS specific ──
  int n_layers;  // number of layers (0 if not LTE_LAYERS)

  // ── LTE_TIME_INTEGRATED specific ──
  int integration_points;  // number of time steps (0 if not TIME_INTEGRATED)

  // ── SLOPING specific ──
  // Pre-computed x-grid and lambda-grid for log-transform FFT
  std::vector<double> sloping_x_grid;    // uniform x-grid
  std::vector<double> sloping_lam_grid;  // corresponding λ-grid
  long sloping_n_signal;
  double sloping_dx;
  double sloping_alpha;
  double sloping_beta;

  // ── CHEMCAM specific ──
  // 3 CCD ranges: [UV, VIO, VISNIR]
  struct CCDRange {
    long i_start;       // start index in output wavelengths
    long n_sub;         // number of wavelengths in this CCD
    long n_fine_sub;    // number of fine grid points for this CCD
    double fine_dl_sub; // fine grid spacing for this CCD
    std::vector<double> fine_wavelengths_sub;
    std::vector<double> output_wavelengths_sub;
    std::vector<std::complex<double>> kernel_fft_sub;
    long N_fft_sub;
  };
  std::vector<CCDRange> ccd_ranges;
  // scale_vio and scale_visnir are in params (differentiated)

  // ── ARYELLE specific ──
  // Wavelength-dependent sigma: σ(λ) = aryelle_sigma_slope * λ + aryelle_sigma_intercept
  double aryelle_sigma_slope;
  double aryelle_sigma_intercept;

  // ── Aki (Einstein A_ji) per-line baseline values ──
  // Flat, line-ordered: iterate ion_line_groups in order, then .lines in order,
  // and push back A_ji for ALL lines (including those outside any grid). This
  // is the ordering accepted by physics::compute_emission_coeff's
  // `aki_overrides` parameter and is also the ordering of the Enzyme adjoint
  // returned by reverse-mode AD w.r.t. this buffer.
  std::vector<double> aki_active;
  long n_aki = 0;

  // ── Stark-FWHM per-line multipliers ──
  // Same layout / same length as `aki_active`. Seeded from `line.stark_mult`
  // in build_const_data, so any `set_stark_multipliers` in force at experiment-
  // construction time is picked up. Threaded through the same physics-kernel
  // hook as `aki_overrides` (see physics_kernel.h) so Enzyme can treat these
  // multipliers as first-class differentiable inputs.
  std::vector<double> stark_active;
};

// =====================================================================
// Iterative in-place FFT (no recursion, no dynamic allocation)
// Uses POSITIVE angle convention (matching FFT.cpp)
// =====================================================================
static void ad_fft(std::complex<double> *a, long n) {
  // Bit-reversal permutation
  for (long i = 1, j = 0; i < n; i++) {
    long bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  // Cooley-Tukey butterfly (positive angle, matching FFT.cpp convention)
  for (long len = 2; len <= n; len <<= 1) {
    double ang = 2.0 * std::numbers::pi / static_cast<double>(len);
    std::complex<double> wn(cos(ang), sin(ang));
    for (long i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (long j = 0; j < len / 2; j++) {
        auto u = a[i + j];
        auto v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wn;
      }
    }
  }
}

// Iterative IFFT (negative angle, divide by N — matching FFT.cpp convention)
static void ad_ifft(std::complex<double> *a, long n) {
  // Bit-reversal permutation
  for (long i = 1, j = 0; i < n; i++) {
    long bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  // Cooley-Tukey butterfly (negative angle)
  for (long len = 2; len <= n; len <<= 1) {
    double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
    std::complex<double> wn(cos(ang), sin(ang));
    for (long i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (long j = 0; j < len / 2; j++) {
        auto u = a[i + j];
        auto v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wn;
      }
    }
  }
  // Divide by N
  double inv_n = 1.0 / static_cast<double>(n);
  for (long i = 0; i < n; i++)
    a[i] *= inv_n;
}

// =====================================================================
// All physics primitives (Voigt, blackbody, partition function, Saha,
// population levels, Doppler/Stark broadening) and the line-broadening
// + RTE kernels live in physics_kernel.h (namespace libskit::physics).
// They are shared verbatim with the high-level LTEModel / LTELayers /
// LTESharedLayers paths so there is only ONE implementation of the
// LTE physics in the codebase.
// =====================================================================

// =====================================================================
// Helper: FFT convolution with pre-computed kernel
// =====================================================================
static void ad_convolve_fft(
    const std::vector<double> &irradiance,
    const std::vector<std::complex<double>> &kernel_fft, long N_fft,
    std::vector<double> &convolved) {

  long n_fine = static_cast<long>(irradiance.size());
  std::vector<std::complex<double>> signal_c(N_fft, {0.0, 0.0});
  for (long i = 0; i < n_fine; i++)
    signal_c[i] = {irradiance[i], 0.0};

  ad_fft(signal_c.data(), N_fft);

  for (long i = 0; i < N_fft; i++)
    signal_c[i] *= kernel_fft[i];

  ad_ifft(signal_c.data(), N_fft);

  convolved.resize(n_fine);
  for (long i = 0; i < n_fine; i++) {
    double val = signal_c[i].real();
    convolved[i] = (val > 0.0) ? val : 0.0;
  }
}

// =====================================================================
// Helper: interpolate from fine grid to output grid
// =====================================================================
static void ad_interpolate(
    const std::vector<double> &convolved,
    double dl, double wl_start,
    const double *out_wl, long n_out,
    std::vector<double> &y_sim) {

  long n_fine = static_cast<long>(convolved.size());
  for (long i = 0; i < n_out; i++) {
    double pos = (out_wl[i] - wl_start) / dl;
    long idx = static_cast<long>(pos);
    if (idx < 0)
      idx = 0;
    if (idx >= n_fine - 1)
      idx = n_fine - 2;
    double frac = pos - idx;
    y_sim[i] = convolved[idx] * (1.0 - frac) + convolved[idx + 1] * frac;
  }
}

// =====================================================================
// Instrument convolution: GenericSpectrometer
// FFT convolve on fine grid → interpolate to output grid
// =====================================================================
static void apply_inst_generic(
    const std::vector<double> &irradiance,
    const EnzymeConstData *data,
    const double *out_wl, long n_out,
    std::vector<double> &y_sim) {

  std::vector<double> convolved;
  ad_convolve_fft(irradiance, data->kernel_fft, data->N_fft, convolved);
  ad_interpolate(convolved,
                     data->fine_dl,
                     data->fine_wavelengths[0],
                     out_wl, n_out, y_sim);
}

// =====================================================================
// Instrument convolution: SlopingSpectrometer
// Jacobian-weighted FFT in log-transformed x-space
// =====================================================================
static void apply_inst_sloping(
    const std::vector<double> &irradiance,
    const EnzymeConstData *data,
    const double *out_wl, long n_out,
    std::vector<double> &y_sim) {

  const long n_signal = data->sloping_n_signal;
  const double *lam_grid = data->sloping_lam_grid.data();
  const double alpha = data->sloping_alpha;
  const double beta = data->sloping_beta;
  const long N_fft = data->N_fft;

  // Weight by Jacobian: f(x) = I(λ(x)) * (α*λ + β)
  std::vector<std::complex<double>> signal_c(N_fft, {0.0, 0.0});
  for (long i = 0; i < n_signal; i++) {
    double jacobian = alpha * lam_grid[i] + beta;
    signal_c[i] = {irradiance[i] * jacobian, 0.0};
  }

  ad_fft(signal_c.data(), N_fft);

  const std::complex<double> *kern = data->kernel_fft.data();
  for (long i = 0; i < N_fft; i++)
    signal_c[i] *= kern[i];

  ad_ifft(signal_c.data(), N_fft);

  // fftshift
  std::vector<std::complex<double>> shifted(N_fft);
  for (long i = 0; i < N_fft / 2; i++) {
    shifted[i] = signal_c[i + N_fft / 2];
    shifted[i + N_fft / 2] = signal_c[i];
  }

  // Remove Jacobian, clamp negatives
  std::vector<double> broadened(n_signal);
  for (long i = 0; i < n_signal; i++) {
    double jacobian = alpha * lam_grid[i] + beta;
    double val = shifted[i].real() / jacobian;
    broadened[i] = (val > 0.0) ? val : 0.0;
  }

  // Interpolate back to output grid
  int j = 0;
  for (long i = 0; i < n_out; i++) {
    double lam = out_wl[i];
    while (j < n_signal - 2 && lam_grid[j + 1] < lam)
      j++;
    if (j >= n_signal - 1) {
      y_sim[i] = broadened[n_signal - 1];
    } else {
      double t = (lam - lam_grid[j]) / (lam_grid[j + 1] - lam_grid[j]);
      y_sim[i] = broadened[j] * (1.0 - t) + broadened[j + 1] * t;
    }
  }
}

// =====================================================================
// Instrument convolution: Aryelle
// Direct O(N²) trapezoid integration with λ-dependent Voigt.
// irradiance is on the OUTPUT grid (not a fine grid).
// =====================================================================
static void apply_inst_aryelle(
    const std::vector<double> &irradiance,
    const EnzymeConstData *data,
    const double *out_wl, long n_out,
    std::vector<double> &y_sim) {

  for (long i = 0; i < n_out - 1; i++) {
    double sigma = data->aryelle_sigma_slope * out_wl[i] +
                   data->aryelle_sigma_intercept;
    double gamma_inst = 0.0;

    double search_range = 4.0 * sigma + 300.0 * gamma_inst;
    long lower_idx = 0;
    for (long k = 0; k < n_out; k++) {
      if (out_wl[k] > out_wl[i] - search_range) {
        lower_idx = k;
        break;
      }
    }
    long upper_idx = n_out;
    for (long k = n_out - 1; k >= 0; k--) {
      if (out_wl[k] < out_wl[i] + search_range) {
        upper_idx = k + 1;
        break;
      }
    }

    double sum = 0.0;
    for (long j_idx = lower_idx + 1; j_idx < upper_idx; j_idx++) {
      sum += 0.5 * (out_wl[j_idx] - out_wl[j_idx - 1]) *
             (irradiance[j_idx - 1] *
                  physics::voigt(out_wl[i] - out_wl[j_idx - 1], sigma,
                               gamma_inst) +
              irradiance[j_idx] *
                  physics::voigt(out_wl[i] - out_wl[j_idx], sigma,
                               gamma_inst));
    }
    y_sim[i] = sum;
  }
  y_sim[n_out - 1] = 0.0;
}

// =====================================================================
// Helper: normalize and compute loss
// =====================================================================
static double normalize_and_loss(
    const double *y_sim, long n_out,
    const double *y_exp,
    int normalization_type, int loss_type,
    double scale_factor_param,
    const double *loss_weights, long n_loss_weights) {

  double y_exp_max = 0.0;
  for (long i = 0; i < n_out; i++)
    if (y_exp[i] > y_exp_max)
      y_exp_max = y_exp[i];

  std::vector<double> vec1(n_out), vec2(n_out);

  switch (normalization_type) {
  case AnalyticNorm: {
    // Analytic minimiser of the applied L2 loss ‖y_sim/c − y_exp‖² wrt c:
    //   d/dc ‖y_sim/c − y_exp‖² = 0  ⇒  c* = ⟨y_sim, y_sim⟩ / ⟨y_sim, y_exp⟩.
    // (See paper Sec. 2.2 / Eq. (34); this differs from the naïve
    //  ⟨y_sim, y_exp⟩/⟨y_exp, y_exp⟩, which minimises ‖y_sim − c·y_exp‖²
    //  — the *other* direction of the projection.)
    double num = 0.0, den = 0.0;
    for (long i = 0; i < n_out; i++) {
      num += y_sim[i] * y_sim[i];
      den += y_sim[i] * y_exp[i];
    }
    den += 1e-30;
    double c = num / den;
    double scale = c * y_exp_max;
    for (long i = 0; i < n_out; i++) {
      vec1[i] = y_sim[i] / scale;
      vec2[i] = y_exp[i] / y_exp_max;
    }
    break;
  }
  case OptimizedScaleNorm: {
    double scale = scale_factor_param * y_exp_max;
    for (long i = 0; i < n_out; i++) {
      vec1[i] = y_sim[i] / scale;
      vec2[i] = y_exp[i] / y_exp_max;
    }
    break;
  }
  case MaxNorm:
  default: {
    double sim_max = 0.0;
    for (long i = 0; i < n_out; i++)
      if (y_sim[i] > sim_max)
        sim_max = y_sim[i];
    if (sim_max <= 0.0)
      sim_max = 1.0;
    for (long i = 0; i < n_out; i++) {
      vec1[i] = y_sim[i] / sim_max;
      vec2[i] = y_exp[i] / y_exp_max;
    }
    break;
  }
  }

  double loss = 0.0;
  switch (loss_type) {
  case L2Loss:
    for (long i = 0; i < n_out; i++) {
      double difference = vec1[i] - vec2[i];
      loss += difference * difference;
    }
    break;
  case IntensityWeightedLoss:
    for (long i = 0; i < n_out; i++) {
      double difference = vec1[i] - vec2[i];
      loss += difference * difference / (0.05 + vec2[i] * vec2[i]);
    }
    break;
  case L1Loss:
    for (long i = 0; i < n_out; i++)
      loss += std::abs(vec1[i] - vec2[i]);
    break;
  case LogL2Loss:
    for (long i = 0; i < n_out; i++) {
      double log_difference = log(1.0 + vec1[i]) - log(1.0 + vec2[i]);
      loss += log_difference * log_difference;
    }
    break;
  case WeightedL2Loss: {
    if (n_loss_weights == n_out && loss_weights != nullptr) {
      double wsum = 0.0;
      for (long i = 0; i < n_out; i++) {
        const double w = loss_weights[i];
        const double difference = vec1[i] - vec2[i];
        loss += w * difference * difference;
        wsum += w;
      }
      loss /= (wsum + 1e-30);
    } else {
      // Fallback: uniform weights == plain L2
      for (long i = 0; i < n_out; i++) {
        const double difference = vec1[i] - vec2[i];
        loss += difference * difference;
      }
    }
    break;
  }
  default:
    for (long i = 0; i < n_out; i++) {
      double difference = vec1[i] - vec2[i];
      loss += difference * difference;
    }
    break;
  }

  return loss;
}

// =====================================================================
// Model traits — one struct per plasma model.
//
// Each trait provides:
//   - extra_offset(data): number of "model parameters" before the
//     optional scale_factor / scale_vio / scale_visnir tail.
//   - compute_irradiance(params, data, wl, n_wl, out): writes the
//     model's emitted irradiance on the supplied wavelength grid.
//
// The signature `compute_irradiance` is identical for every model so
// that an Instrument trait can call any model on any grid (this is
// what ChemCam needs: it loops over 3 sub-grids).
// =====================================================================

struct LTEModelTraits {
  static int extra_offset(const EnzymeConstData &constant_data) {
    return 3 + constant_data.n_elements;
  }
  static void compute_irradiance(double *params, const EnzymeConstData *data,
                                 const double *wl, long n_wl,
                                 std::vector<double> &irradiance,
                                 const double *aki_overrides,
                                 const double *stark_overrides) {
    physics::compute_irradiance_lte(params[0], params[1], params[2], &params[3],
                                  data->n_elements, data, wl, n_wl,
                                  irradiance, aki_overrides, stark_overrides);
  }
};

// LTEModel param layout traits
// (the scale fields are unused — Enzyme differentiates physical params).

struct LTELayersTraits {
  static int extra_offset(const EnzymeConstData &constant_data) {
    return constant_data.n_layers * (3 + constant_data.n_elements);
  }
  static void compute_irradiance(double *params, const EnzymeConstData *data,
                                 const double *wl, long n_wl,
                                 std::vector<double> &irradiance,
                                 const double *aki_overrides,
                                 const double *stark_overrides) {
    const int n_elements = data->n_elements;
    const int ppl = 3 + n_elements;
    irradiance.assign(n_wl, 0.0);
    std::vector<double> emission_coefficients;
    for (int layer = 0; layer < data->n_layers; layer++) {
      const int base = layer * ppl;
      physics::compute_emission_coeff(params[base], params[base + 1],
                                    params[base + 2], &params[base + 3],
                                    n_elements, data, wl, n_wl,
                                    emission_coefficients, aki_overrides,
                                    stark_overrides);
      physics::apply_rte_multilayer(wl, n_wl, emission_coefficients, params[base],
                                  params[base + 2], irradiance);
    }
  }
};

struct LTESharedLayersTraits {
  static int extra_offset(const EnzymeConstData &constant_data) {
    return 3 * constant_data.n_layers + constant_data.n_elements;
  }
  static void compute_irradiance(double *params, const EnzymeConstData *data,
                                 const double *wl, long n_wl,
                                 std::vector<double> &irradiance,
                                 const double *aki_overrides,
                                 const double *stark_overrides) {
    const int n_elements = data->n_elements;
    const double *element_ratios = &params[3 * data->n_layers];
    irradiance.assign(n_wl, 0.0);
    std::vector<double> emission_coefficients;
    for (int layer = 0; layer < data->n_layers; layer++) {
      const int base = layer * 3;
      physics::compute_emission_coeff(params[base], params[base + 1],
                                    params[base + 2], element_ratios,
                                    n_elements, data, wl, n_wl,
                                    emission_coefficients, aki_overrides,
                                    stark_overrides);
      physics::apply_rte_multilayer(wl, n_wl, emission_coefficients, params[base],
                                  params[base + 2], irradiance);
    }
  }
};

struct LTETimeIntegratedTraits {
  static int extra_offset(const EnzymeConstData &constant_data) {
    return 3 + constant_data.n_elements + 10; // 10 time-evolution params
  }
  static void compute_irradiance(double *params, const EnzymeConstData *data,
                                 const double *wl, long n_wl,
                                 std::vector<double> &irradiance,
                                 const double *aki_overrides,
                                 const double *stark_overrides) {
    const int n_elements = data->n_elements;
    const double *element_ratios = &params[3];
    const int tb = 3 + n_elements;
    const double start_T = params[tb + 0], end_T = params[tb + 1],
                 tau_T = params[tb + 2];
    const double start_ne = params[tb + 3], end_ne = params[tb + 4],
                 tau_ne = params[tb + 5];
    const double start_L = params[tb + 6], end_L = params[tb + 7],
                 tau_L = params[tb + 8];
    const double int_time = params[tb + 9];

    const int n_steps = data->integration_points;
    const double dt = int_time / n_steps;
    const double w = dt / int_time;

    irradiance.assign(n_wl, 0.0);
    std::vector<double> irr_t;
    for (int step = 0; step < n_steps; step++) {
      const double t = (step + 0.5) * dt;
      const double T_t = end_T + (start_T - end_T) * exp(-t / tau_T);
      const double ne_t = end_ne + (start_ne - end_ne) * exp(-t / tau_ne);
      const double L_t = end_L + (start_L - end_L) * exp(-t / tau_L);
      physics::compute_irradiance_lte(T_t, ne_t, L_t, element_ratios, n_elements,
                                    data, wl, n_wl, irr_t, aki_overrides,
                                    stark_overrides);
      for (long i = 0; i < n_wl; i++)
        irradiance[i] += irr_t[i] * w;
    }
  }
};

struct LTETwoTimeTraits {
  // No leading temperature/electron_density/length dummy slots: unlike the
  // other model traits, LTETwoTime's fittable params are early_*/delta_*/
  // weight_early followed by the elements (matches get_parameter_names()).
  static int extra_offset(const EnzymeConstData &constant_data) {
    return 7 + constant_data.n_elements;
  }
  static void compute_irradiance(double *params, const EnzymeConstData *data,
                                 const double *wl, long n_wl,
                                 std::vector<double> &irradiance,
                                 const double *aki_overrides,
                                 const double *stark_overrides) {
    const int n_elements = data->n_elements;
    const double *element_ratios = &params[7];

    const double early_temperature = params[0];
    const double early_electron_density = params[1];
    const double early_length = params[2];
    const double delta_temperature = params[3];
    const double delta_electron_density = params[4];
    const double delta_length = params[5];
    const double early_weight = params[6];
    const double late_weight = 1.0 - early_weight;

    const double late_temperature = early_temperature - delta_temperature;
    const double late_electron_density = early_electron_density - delta_electron_density;
    const double late_length = early_length + delta_length;

    std::vector<double> early_irradiance;
    std::vector<double> late_irradiance;

    physics::compute_irradiance_lte(
        early_temperature, early_electron_density, early_length,
        element_ratios, n_elements, data, wl, n_wl, early_irradiance,
        aki_overrides, stark_overrides);
    physics::compute_irradiance_lte(
        late_temperature, late_electron_density, late_length,
        element_ratios, n_elements, data, wl, n_wl, late_irradiance,
        aki_overrides, stark_overrides);

    irradiance.resize(n_wl);
    for (long wavelength_index = 0; wavelength_index < n_wl;
         wavelength_index++) {
      irradiance[wavelength_index] =
          early_weight * early_irradiance[wavelength_index] +
          late_weight * late_irradiance[wavelength_index];
    }
  }
};

// =====================================================================
// Instrument traits — one struct per instrument.
//
// Each trait provides:
//   - compute_y_sim<Model>(params, data, y_sim): drives the full
//     forward model on the right grid(s) for this instrument and
//     writes the simulated, instrument-broadened spectrum into y_sim.
//
// The Model parameter lets the same Instrument trait work with any
// plasma model — crucial for ChemCam, which evaluates the model once
// per CCD sub-grid.
// =====================================================================

struct GenericInstrumentTraits {
  template <typename Model>
  static void compute_y_sim(double *params, const double *aki_overrides,
                            const double *stark_overrides,
                            const EnzymeConstData *data,
                            std::vector<double> &y_sim) {
    std::vector<double> irradiance;
    Model::compute_irradiance(params, data, data->fine_wavelengths.data(),
                              data->n_fine, irradiance, aki_overrides,
                              stark_overrides);
    apply_inst_generic(irradiance, data,
                              data->output_wavelengths.data(), data->n_out,
                              y_sim);
  }
};

struct SlopingInstrumentTraits {
  template <typename Model>
  static void compute_y_sim(double *params, const double *aki_overrides,
                            const double *stark_overrides,
                            const EnzymeConstData *data,
                            std::vector<double> &y_sim) {
    std::vector<double> irradiance;
    // SLOPING evaluates the model on the log-transform λ-grid, which
    // build_const_data stores in fine_wavelengths (n_fine == n_signal).
    Model::compute_irradiance(params, data, data->fine_wavelengths.data(),
                              data->n_fine, irradiance, aki_overrides,
                              stark_overrides);
    apply_inst_sloping(irradiance, data,
                              data->output_wavelengths.data(), data->n_out,
                              y_sim);
  }
};

struct AryelleInstrumentTraits {
  template <typename Model>
  static void compute_y_sim(double *params, const double *aki_overrides,
                            const double *stark_overrides,
                            const EnzymeConstData *data,
                            std::vector<double> &y_sim) {
    std::vector<double> irradiance;
    // ARYELLE evaluates the model directly on the output grid.
    Model::compute_irradiance(params, data, data->output_wavelengths.data(),
                              data->n_out, irradiance, aki_overrides,
                              stark_overrides);
    apply_inst_aryelle(irradiance, data,
                              data->output_wavelengths.data(), data->n_out,
                              y_sim);
  }
};

struct ChemCamInstrumentTraits {
  template <typename Model>
  static void compute_y_sim(double *params, const double *aki_overrides,
                            const double *stark_overrides,
                            const EnzymeConstData *data,
                            std::vector<double> &y_sim) {
    const int sf_base =
        Model::extra_offset(*data) + (data->has_scale_factor ? 1 : 0);
    const double scales[3] = {1.0, params[sf_base], params[sf_base + 1]};

    std::vector<double> irr_sub, conv_sub;
    const int n_ccds = static_cast<int>(data->ccd_ranges.size());
    for (int c = 0; c < n_ccds; c++) {
      const auto &ccd = data->ccd_ranges[c];
      if (ccd.n_sub <= 0)
        continue;

      Model::compute_irradiance(params, data, ccd.fine_wavelengths_sub.data(),
                                ccd.n_fine_sub, irr_sub, aki_overrides,
                                stark_overrides);
      ad_convolve_fft(irr_sub, ccd.kernel_fft_sub, ccd.N_fft_sub,
                          conv_sub);

      const double *owls = ccd.output_wavelengths_sub.data();
      const double scale = scales[c];
      for (long i = 0; i < ccd.n_sub; i++) {
        double pos = (owls[i] - ccd.fine_wavelengths_sub[0]) / ccd.fine_dl_sub;
        long idx = static_cast<long>(pos);
        if (idx < 0)
          idx = 0;
        if (idx >= ccd.n_fine_sub - 1)
          idx = ccd.n_fine_sub - 2;
        const double frac = pos - idx;
        y_sim[ccd.i_start + i] =
            (conv_sub[idx] * (1.0 - frac) + conv_sub[idx + 1] * frac) * scale;
      }
    }
  }
};

// =====================================================================
// Core loss function — Enzyme differentiates each instantiation of
// this template.  The body is ~10 lines: dispatch is all compile-time.
//
// params layout depends on Model::extra_offset(data); the tail is:
//   [..., (optional) scale_factor,
//         (optional, CHEMCAM only) scale_vio, scale_visnir]
// =====================================================================
template <typename Model, typename Instrument>
static double __attribute__((noinline))
loss_impl(double *params, const double *aki, const double *stark,
                 const EnzymeConstData *data) {

  std::vector<double> y_sim(data->n_out, 0.0);
  Instrument::template compute_y_sim<Model>(params, aki, stark, data, y_sim);

  const double scale_factor_param =
      data->has_scale_factor ? params[Model::extra_offset(*data)] : 0.0;

  return normalize_and_loss(
      y_sim.data(), data->n_out, data->reference_irradiance.data(),
      data->normalization_type, data->loss_type, scale_factor_param,
      data->loss_weights.empty() ? nullptr : data->loss_weights.data(),
      static_cast<long>(data->loss_weights.size()));
}

// =====================================================================
// Dispatch: pick the right (Model, Instrument) specialization.
//
// IMPORTANT: Enzyme needs to see the target function as a direct symbol
// at the __enzyme_autodiff call site — it cannot follow an indirect
// function pointer.  Even an in-source `switch` over function pointers
// gets folded by the optimizer into a single indirect call.
//
// Workaround: wrap each __enzyme_autodiff invocation in its own
// `noinline` function (one per Model × Instrument pair).  Each wrapper
// has a hard-coded, direct call to a specific loss_impl
// specialization, so Enzyme can resolve the symbol.  The runtime
// dispatch (the switch) then just calls the right wrapper.
// =====================================================================
using EnzymeLossFn = double (*)(double *, const double *, const double *,
                                 const EnzymeConstData *);

template <typename Model>
static EnzymeLossFn pick_instrument(int instrument_type) {
  switch (instrument_type) {
  case INST_GENERIC:
    return &loss_impl<Model, GenericInstrumentTraits>;
  case INST_SLOPING:
    return &loss_impl<Model, SlopingInstrumentTraits>;
  case INST_CHEMCAM:
    return &loss_impl<Model, ChemCamInstrumentTraits>;
  case INST_ARYELLE:
    return &loss_impl<Model, AryelleInstrumentTraits>;
  }
  return nullptr;
}

static EnzymeLossFn pick_loss_fn(int model_type, int instrument_type) {
  switch (model_type) {
  case MODEL_LTE:
    return pick_instrument<LTEModelTraits>(instrument_type);
  case MODEL_LTE_LAYERS:
    return pick_instrument<LTELayersTraits>(instrument_type);
  case MODEL_LTE_SHARED_LAYERS:
    return pick_instrument<LTESharedLayersTraits>(instrument_type);
  case MODEL_LTE_TIME_INTEGRATED:
    return pick_instrument<LTETimeIntegratedTraits>(instrument_type);
  case MODEL_LTE_TWO_TIME:
    return pick_instrument<LTETwoTimeTraits>(instrument_type);
  }
  return nullptr;
}

// One reverse-mode-pass wrapper per (Model, Instrument).  Marked
// noinline so the call to __enzyme_autodiff stays at a distinct call
// site with a directly-resolvable function symbol.
template <typename Model, typename Instrument>
static __attribute__((noinline)) void
grad_pass(double *params, double *grads, const double *aki,
                 const double *stark, const EnzymeConstData *data) {
  __enzyme_autodiff<void>(
      reinterpret_cast<void *>(&loss_impl<Model, Instrument>),
      enzyme_dup, params, grads,
      enzyme_const, aki,
      enzyme_const, stark,
      enzyme_const, data);
}

// Reverse-mode pass that differentiates the loss w.r.t. the per-line A_ji
// values instead of the regular parameter vector.  Parameters are frozen
// (enzyme_const); `aki` is the active duplicated input and `daki` is the
// shadow buffer receiving ∂L/∂A_ji.
template <typename Model, typename Instrument>
static __attribute__((noinline)) void
aki_grad_pass(double *params, const double *aki, double *daki,
                     const double *stark, const EnzymeConstData *data) {
  __enzyme_autodiff<void>(
      reinterpret_cast<void *>(&loss_impl<Model, Instrument>),
      enzyme_const, params,
      enzyme_dup, aki, daki,
      enzyme_const, stark,
      enzyme_const, data);
}

// Reverse-mode pass that differentiates the loss w.r.t. the per-line Stark
// FWHM multipliers.  Parameters and Aki are frozen; `stark` is the active
// duplicated input; `dstark` receives ∂L/∂m_l.
template <typename Model, typename Instrument>
static __attribute__((noinline)) void
stark_grad_pass(double *params, const double *aki,
                       const double *stark, double *dstark,
                       const EnzymeConstData *data) {
  __enzyme_autodiff<void>(
      reinterpret_cast<void *>(&loss_impl<Model, Instrument>),
      enzyme_const, params,
      enzyme_const, aki,
      enzyme_dup, stark, dstark,
      enzyme_const, data);
}

template <typename Model>
static void dispatch_grad_instrument(int instrument_type, double *params,
                                     double *grads, const double *aki,
                                     const double *stark,
                                     const EnzymeConstData *data) {
  switch (instrument_type) {
  case INST_GENERIC:
    grad_pass<Model, GenericInstrumentTraits>(params, grads, aki, stark, data);
    break;
  case INST_SLOPING:
    grad_pass<Model, SlopingInstrumentTraits>(params, grads, aki, stark, data);
    break;
  case INST_CHEMCAM:
    grad_pass<Model, ChemCamInstrumentTraits>(params, grads, aki, stark, data);
    break;
  case INST_ARYELLE:
    grad_pass<Model, AryelleInstrumentTraits>(params, grads, aki, stark, data);
    break;
  }
}

static void dispatch_grad(int model_type, int instrument_type, double *params,
                          double *grads, const double *aki,
                          const double *stark,
                          const EnzymeConstData *data) {
  switch (model_type) {
  case MODEL_LTE:
    dispatch_grad_instrument<LTEModelTraits>(instrument_type, params, grads,
                                             aki, stark, data);
    break;
  case MODEL_LTE_LAYERS:
    dispatch_grad_instrument<LTELayersTraits>(instrument_type, params, grads,
                                              aki, stark, data);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    dispatch_grad_instrument<LTESharedLayersTraits>(instrument_type, params,
                                                    grads, aki, stark, data);
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    dispatch_grad_instrument<LTETimeIntegratedTraits>(instrument_type, params,
                                                      grads, aki, stark, data);
    break;
  case MODEL_LTE_TWO_TIME:
    dispatch_grad_instrument<LTETwoTimeTraits>(instrument_type, params, grads,
                                               aki, stark, data);
    break;
  }
}

// Dispatchers for the Aki-gradient pass (∂L/∂A_ji).
template <typename Model>
static void dispatch_aki_grad_instrument(int instrument_type, double *params,
                                         const double *aki, double *daki,
                                         const double *stark,
                                         const EnzymeConstData *data) {
  switch (instrument_type) {
  case INST_GENERIC:
    aki_grad_pass<Model, GenericInstrumentTraits>(params, aki, daki, stark, data);
    break;
  case INST_SLOPING:
    aki_grad_pass<Model, SlopingInstrumentTraits>(params, aki, daki, stark, data);
    break;
  case INST_CHEMCAM:
    aki_grad_pass<Model, ChemCamInstrumentTraits>(params, aki, daki, stark, data);
    break;
  case INST_ARYELLE:
    aki_grad_pass<Model, AryelleInstrumentTraits>(params, aki, daki, stark, data);
    break;
  }
}

static void dispatch_aki_grad(int model_type, int instrument_type,
                              double *params, const double *aki, double *daki,
                              const double *stark,
                              const EnzymeConstData *data) {
  switch (model_type) {
  case MODEL_LTE:
    dispatch_aki_grad_instrument<LTEModelTraits>(instrument_type, params, aki,
                                                 daki, stark, data);
    break;
  case MODEL_LTE_LAYERS:
    dispatch_aki_grad_instrument<LTELayersTraits>(instrument_type, params, aki,
                                                  daki, stark, data);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    dispatch_aki_grad_instrument<LTESharedLayersTraits>(instrument_type, params,
                                                        aki, daki, stark, data);
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    dispatch_aki_grad_instrument<LTETimeIntegratedTraits>(instrument_type,
                                                          params, aki, daki,
                                                          stark, data);
    break;
  case MODEL_LTE_TWO_TIME:
    dispatch_aki_grad_instrument<LTETwoTimeTraits>(instrument_type, params, aki,
                                                   daki, stark, data);
    break;
  }
}

// Dispatchers for the Stark-gradient pass (∂L/∂m_l).
template <typename Model>
static void dispatch_stark_grad_instrument(int instrument_type, double *params,
                                           const double *aki,
                                           const double *stark, double *dstark,
                                           const EnzymeConstData *data) {
  switch (instrument_type) {
  case INST_GENERIC:
    stark_grad_pass<Model, GenericInstrumentTraits>(params, aki, stark, dstark, data);
    break;
  case INST_SLOPING:
    stark_grad_pass<Model, SlopingInstrumentTraits>(params, aki, stark, dstark, data);
    break;
  case INST_CHEMCAM:
    stark_grad_pass<Model, ChemCamInstrumentTraits>(params, aki, stark, dstark, data);
    break;
  case INST_ARYELLE:
    stark_grad_pass<Model, AryelleInstrumentTraits>(params, aki, stark, dstark, data);
    break;
  }
}

static void dispatch_stark_grad(int model_type, int instrument_type,
                                double *params, const double *aki,
                                const double *stark, double *dstark,
                                const EnzymeConstData *data) {
  switch (model_type) {
  case MODEL_LTE:
    dispatch_stark_grad_instrument<LTEModelTraits>(instrument_type, params, aki,
                                                   stark, dstark, data);
    break;
  case MODEL_LTE_LAYERS:
    dispatch_stark_grad_instrument<LTELayersTraits>(instrument_type, params,
                                                    aki, stark, dstark, data);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    dispatch_stark_grad_instrument<LTESharedLayersTraits>(
        instrument_type, params, aki, stark, dstark, data);
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    dispatch_stark_grad_instrument<LTETimeIntegratedTraits>(
        instrument_type, params, aki, stark, dstark, data);
    break;
  case MODEL_LTE_TWO_TIME:
    dispatch_stark_grad_instrument<LTETwoTimeTraits>(instrument_type, params,
                                                     aki, stark, dstark, data);
    break;
  }
}

// =====================================================================
// Helper: get element names from any model (LTEModel is base for all)
// =====================================================================
static std::vector<std::string>
get_element_names(const Experiment &experiment) {
  std::vector<std::string> names;
  // All model types inherit from LTEModel or contain LTEModel layers
  const auto *layers = dynamic_cast<const LTELayers *>(
      experiment.plasma_model_ptr.get());
  const auto *shared = dynamic_cast<const LTESharedLayers *>(
      experiment.plasma_model_ptr.get());
  if (layers) {
    for (const auto &[name, _] : layers->lte_layers[0].element_ratio)
      names.push_back(name);
  } else if (shared) {
    for (const auto &[name, _] : shared->element_ratio)
      names.push_back(name);
  } else {
    const auto *lte = dynamic_cast<const LTEModel *>(
        experiment.plasma_model_ptr.get());
    for (const auto &[name, _] : lte->element_ratio)
      names.push_back(name);
  }
  return names;
}

// =====================================================================
// Helper: pre-compute FFT kernel for a GenericSpectrometer-style
//         instrument on a given fine grid
// =====================================================================
static void build_kernel_fft(
    double inst_sigma, double inst_gamma, double accuracy,
    long n_fine, double fine_dl,
    std::vector<std::complex<double>> &kernel_fft_out, long &N_fft_out) {

  double kernel_extent =
      compute_kernel_extent(inst_sigma, inst_gamma, accuracy);
  long half = static_cast<long>(std::ceil(kernel_extent / fine_dl));
  long n_kernel = 2 * half + 1;

  std::vector<double> kernel(n_kernel);
  for (long i = 0; i < n_kernel; i++)
    kernel[i] = voigt_profile_approx<double>(
        double((i - half) * fine_dl), inst_sigma, inst_gamma);
  double ksum = 0.0;
  for (long i = 0; i < n_kernel; i++)
    ksum += kernel[i];
  ksum *= fine_dl;
  for (long i = 0; i < n_kernel; i++)
    kernel[i] = kernel[i] * fine_dl / ksum;

  long conv_len = n_fine + n_kernel - 1;
  long N = 1;
  while (N < conv_len)
    N <<= 1;
  N_fft_out = N;

  kernel_fft_out.resize(N, {0.0, 0.0});
  for (long i = 0; i <= half; i++)
    kernel_fft_out[i] = {kernel[half + i], 0.0};
  for (long i = 1; i <= half; i++)
    kernel_fft_out[N - i] = {kernel[half - i], 0.0};

  ad_fft(kernel_fft_out.data(), N);
}

// =====================================================================
// Build EnzymeConstData — all string operations happen here.
// Detects model and instrument type via dynamic_cast.
// =====================================================================
static EnzymeConstData
build_const_data(const Experiment &experiment) {
  EnzymeConstData constant_data;

  // ── Detect model type ──────────────────────────────────────────────
  const auto *layers = dynamic_cast<const LTELayers *>(
      experiment.plasma_model_ptr.get());
  const auto *shared = dynamic_cast<const LTESharedLayers *>(
      experiment.plasma_model_ptr.get());
  const auto *time_int = dynamic_cast<const LTETimeIntegrated *>(
      experiment.plasma_model_ptr.get());
  const auto *two_time = dynamic_cast<const LTETwoTime *>(
      experiment.plasma_model_ptr.get());
  // NB: LTETimeIntegrated and LTETwoTime inherit from LTEModel,
  // so check them BEFORE LTEModel.

  if (layers) {
    constant_data.model_type = MODEL_LTE_LAYERS;
  } else if (shared) {
    constant_data.model_type = MODEL_LTE_SHARED_LAYERS;
  } else if (time_int) {
    constant_data.model_type = MODEL_LTE_TIME_INTEGRATED;
  } else if (two_time) {
    constant_data.model_type = MODEL_LTE_TWO_TIME;
  } else {
    constant_data.model_type = MODEL_LTE;
  }

  // ── Detect instrument type ─────────────────────────────────────────
  const auto *gs = dynamic_cast<const GenericSpectrometer *>(
      experiment.instrument_ptr.get());
  const auto *sloping = dynamic_cast<const SlopingSpectrometer *>(
      experiment.instrument_ptr.get());
  const auto *chemcam = dynamic_cast<const ChemCam *>(
      experiment.instrument_ptr.get());
  const auto *aryelle = dynamic_cast<const Aryelle *>(
      experiment.instrument_ptr.get());

  if (chemcam) {
    constant_data.instrument_type = INST_CHEMCAM;
  } else if (sloping) {
    constant_data.instrument_type = INST_SLOPING;
  } else if (aryelle) {
    constant_data.instrument_type = INST_ARYELLE;
  } else {
    constant_data.instrument_type = INST_GENERIC;
  }

  // ── Element data (common to all models) ────────────────────────────
  auto element_names = get_element_names(experiment);
  constant_data.n_elements = static_cast<int>(element_names.size());

  constant_data.atomic_weights.resize(constant_data.n_elements);
  constant_data.max_ions.resize(constant_data.n_elements);
  constant_data.element_offset.resize(constant_data.n_elements);

  int total_ion_slots = 0;
  unsigned int max_max = 0;
  for (int e = 0; e < constant_data.n_elements; e++) {
    constant_data.atomic_weights[e] = ATOM_INFO.at(element_names[e]).atomic_weight;
    constant_data.max_ions[e] = get_maximum_ionization_level(element_names[e]);
    if (constant_data.max_ions[e] > max_max) max_max = constant_data.max_ions[e];
    constant_data.element_offset[e] = total_ion_slots;
    total_ion_slots += constant_data.max_ions[e] + 1;
  }
  constant_data.max_max_ions = static_cast<int>(max_max);

  constant_data.ion_levels_flat.resize(total_ion_slots);
  constant_data.saha_ion_energies.resize(total_ion_slots, 1e30);

  for (int e = 0; e < constant_data.n_elements; e++) {
    int offset = constant_data.element_offset[e];
    for (unsigned int il = 0; il <= constant_data.max_ions[e]; il++) {
      std::string ion = element_names[e] + " " + to_roman(il + 1);
      physics::IonLevels &ion_levels = constant_data.ion_levels_flat[offset + il];

      if (LEVELS.count(ion) && IONIZATION_ENERGIES.count(ion)) {
        const auto &lev = LEVELS.at(ion);
        ion_levels.ionization_energy = IONIZATION_ENERGIES.at(ion);
        ion_levels.n_levels = static_cast<int>(lev.size());
        ion_levels.g_values.resize(ion_levels.n_levels);
        ion_levels.energies.resize(ion_levels.n_levels);
        for (int j = 0; j < ion_levels.n_levels; j++) {
          ion_levels.g_values[j] = lev[j].g;
          ion_levels.energies[j] = lev[j].energy;
        }
      } else {
        ion_levels.n_levels = 0;
        ion_levels.ionization_energy = 1e30;
      }

      if (il < constant_data.max_ions[e]) {
        constant_data.saha_ion_energies[offset + il] =
            IONIZATION_ENERGIES.count(ion) ? IONIZATION_ENERGIES.at(ion)
                                           : 1e30;
      }
    }
  }

  // Pre-gather emission lines
  for (int e = 0; e < constant_data.n_elements; e++) {
    for (unsigned int il = 0; il < constant_data.max_ions[e]; il++) {
      std::string ion = element_names[e] + " " + to_roman(il + 1);
      if (!LINES.count(ion))
        continue;
      physics::IonLineData ion_line_group;
      ion_line_group.element_idx = e;
      ion_line_group.ion_level = il;
      ion_line_group.atomic_weight = constant_data.atomic_weights[e];
      ion_line_group.lines = LINES.at(ion);
      constant_data.ion_line_groups.push_back(std::move(ion_line_group));
    }
  }

  // Flatten A_ji values into a dense, line-ordered buffer. The ordering must
  // match `compute_emission_coeff`'s `aki_overrides[group_line_offset+line_index]`
  // (counts ALL lines, including those outside the active wavelength range).
  for (const auto &group : constant_data.ion_line_groups) {
    for (const auto &line : group.lines)
      constant_data.aki_active.push_back(line.A_ji);
  }
  constant_data.n_aki = static_cast<long>(constant_data.aki_active.size());

  // Flatten stark multipliers in the same order (used by `stark_overrides`).
  // Seeded from `line.stark_mult` so an active `set_stark_multipliers` in the
  // database is picked up at experiment build time.
  constant_data.stark_active.reserve(constant_data.n_aki);
  for (const auto &group : constant_data.ion_line_groups) {
    for (const auto &line : group.lines)
      constant_data.stark_active.push_back(line.stark_mult);
  }

  // ── Wavelength grids and reference spectrum ────────────────────────
  Eigen::VectorXd out_wl = experiment.true_spectrum.wavelengths;
  constant_data.n_out = out_wl.size();
  constant_data.output_wavelengths.assign(out_wl.data(), out_wl.data() + constant_data.n_out);

  Eigen::VectorXd ref = experiment.true_spectrum.irradiance;
  constant_data.reference_irradiance.assign(ref.data(), ref.data() + constant_data.n_out);

  if (experiment.loss_weights.size() > 0) {
    if (experiment.loss_weights.size() != constant_data.n_out) {
      throw std::runtime_error(
          "Experiment.loss_weights length (" +
          std::to_string(experiment.loss_weights.size()) +
          ") must match true_spectrum length (" +
          std::to_string(constant_data.n_out) + ")");
    }
    constant_data.loss_weights.assign(experiment.loss_weights.data(),
                          experiment.loss_weights.data() + constant_data.n_out);
  }

  constant_data.accuracy = get_accuracy();
  constant_data.loss_type = static_cast<int>(experiment.loss_type);
  constant_data.normalization_type = static_cast<int>(experiment.normalization_type);
  constant_data.has_scale_factor = (experiment.normalization_type == OptimizedScaleNorm);

  // ── Model-specific data ────────────────────────────────────────────
  constant_data.n_layers = 0;
  constant_data.integration_points = 0;

  if (constant_data.model_type == MODEL_LTE_LAYERS) {
    constant_data.n_layers = static_cast<int>(layers->lte_layers.size());
  } else if (constant_data.model_type == MODEL_LTE_SHARED_LAYERS) {
    constant_data.n_layers = static_cast<int>(shared->n_layers());
  } else if (constant_data.model_type == MODEL_LTE_TIME_INTEGRATED) {
    constant_data.integration_points = time_int->integration_points;
  }

  // ── Instrument-specific data ───────────────────────────────────────
  constant_data.aryelle_sigma_slope = 0.0;
  constant_data.aryelle_sigma_intercept = 0.0;
  constant_data.sloping_n_signal = 0;
  constant_data.sloping_dx = 0.0;
  constant_data.sloping_alpha = 0.0;
  constant_data.sloping_beta = 0.0;

  // Fine wavelength grid (needed by most instruments)
  Eigen::VectorXd fine =
      experiment.plasma_model_ptr->build_fine_wavelength_grid(out_wl);
  constant_data.n_fine = fine.size();
  constant_data.fine_dl = fine[1] - fine[0];
  constant_data.fine_wavelengths.assign(fine.data(), fine.data() + constant_data.n_fine);

  switch (constant_data.instrument_type) {
  case INST_GENERIC: {
    double inst_sigma = static_cast<double>(gs->sigma);
    double inst_gamma = static_cast<double>(gs->gamma);
    build_kernel_fft(inst_sigma, inst_gamma, constant_data.accuracy,
                     constant_data.n_fine, constant_data.fine_dl, constant_data.kernel_fft, constant_data.N_fft);
    break;
  }

  case INST_SLOPING: {
    const double fwhm_to_sigma = 2.0 * std::sqrt(2.0 * std::log(2.0));
    constant_data.sloping_alpha = sloping->fwhm_slope / fwhm_to_sigma;
    constant_data.sloping_beta = sloping->fwhm_intercept / fwhm_to_sigma;

    double lam_min = out_wl[0];
    double lam_max = out_wl[constant_data.n_out - 1];
    double x_min = std::log(constant_data.sloping_alpha * lam_min + constant_data.sloping_beta) / constant_data.sloping_alpha;
    double x_max = std::log(constant_data.sloping_alpha * lam_max + constant_data.sloping_beta) / constant_data.sloping_alpha;

    constant_data.sloping_n_signal = constant_data.n_out;  // same count as output
    constant_data.sloping_dx = (x_max - x_min) / (constant_data.sloping_n_signal - 1);
    double sigma_pix = 1.0 / constant_data.sloping_dx;

    // Build x-grid and lambda-grid
    constant_data.sloping_x_grid.resize(constant_data.sloping_n_signal);
    constant_data.sloping_lam_grid.resize(constant_data.sloping_n_signal);
    for (long i = 0; i < constant_data.sloping_n_signal; i++) {
      constant_data.sloping_x_grid[i] = x_min + i * constant_data.sloping_dx;
      constant_data.sloping_lam_grid[i] =
          (std::exp(constant_data.sloping_alpha * constant_data.sloping_x_grid[i]) - constant_data.sloping_beta) /
          constant_data.sloping_alpha;
    }

    // Also update the fine grid to use the sloping lambda grid
    // (the irradiance will be computed on this non-uniform grid)
    constant_data.n_fine = constant_data.sloping_n_signal;
    constant_data.fine_dl = constant_data.sloping_lam_grid[1] - constant_data.sloping_lam_grid[0]; // approximate
    constant_data.fine_wavelengths.assign(constant_data.sloping_lam_grid.begin(), constant_data.sloping_lam_grid.end());

    // Build kernel: Gaussian with σ_x = sigma_pix in x-space, padded
    long N_signal = constant_data.sloping_n_signal;
    int next_pow2 = static_cast<int>(std::ceil(std::log2(N_signal)));
    long N = 1L << next_pow2;
    constant_data.N_fft = N;

    // Kernel centred at N/2,  using LinSpaced(N, -N/2, N/2) semantics
    constant_data.kernel_fft.resize(N, {0.0, 0.0});
    double ksum = 0.0;
    for (long i = 0; i < N; i++) {
      double x_k = -N / 2.0 + i * (static_cast<double>(N) / (N - 1));
      double val = std::exp(-0.5 * (x_k / sigma_pix) * (x_k / sigma_pix));
      constant_data.kernel_fft[i] = {val, 0.0};
      ksum += val;
    }
    for (long i = 0; i < N; i++)
      constant_data.kernel_fft[i] /= ksum;
    ad_fft(constant_data.kernel_fft.data(), N);
    break;
  }

  case INST_CHEMCAM: {
    // 3 CCD ranges
    struct RangeDef {
      double wl_min, wl_max, fwhm;
    };
    constexpr double FWHM_TO_SIGMA =
        1.0 / (2.0 * 1.17741002251547469);
    const RangeDef ranges[] = {
      {240.1, 342.2, 0.15},   // UV
      {382.1, 469.3, 0.20},   // VIO
      {474.0, 906.5, 0.61},   // VISNIR
    };

    constant_data.ccd_ranges.resize(3);
    for (int c = 0; c < 3; c++) {
      auto &ccd = constant_data.ccd_ranges[c];
      const auto &rng = ranges[c];

      // Find output indices in this CCD range
      long i_start = -1, i_end = -1;
      for (long i = 0; i < constant_data.n_out; i++) {
        double wl = constant_data.output_wavelengths[i];
        if (wl >= rng.wl_min && wl <= rng.wl_max) {
          if (i_start < 0) i_start = i;
          i_end = i;
        }
      }

      if (i_start < 0) {
        ccd.i_start = 0;
        ccd.n_sub = 0;
        ccd.n_fine_sub = 0;
        ccd.fine_dl_sub = 0.0;
        ccd.N_fft_sub = 0;
        continue;
      }

      ccd.i_start = i_start;
      ccd.n_sub = i_end - i_start + 1;

      // Sub output wavelengths
      ccd.output_wavelengths_sub.assign(
          constant_data.output_wavelengths.begin() + i_start,
          constant_data.output_wavelengths.begin() + i_end + 1);

      // Build fine grid for this CCD sub-range
      Eigen::VectorXd wl_sub =
          out_wl.segment(i_start, ccd.n_sub);
      Eigen::VectorXd fine_sub =
          experiment.plasma_model_ptr->build_fine_wavelength_grid(wl_sub);
      ccd.n_fine_sub = fine_sub.size();
      ccd.fine_dl_sub = fine_sub[1] - fine_sub[0];
      ccd.fine_wavelengths_sub.assign(fine_sub.data(),
                                      fine_sub.data() + ccd.n_fine_sub);

      // Kernel FFT for this CCD
      double sigma = rng.fwhm * FWHM_TO_SIGMA;
      build_kernel_fft(sigma, 0.0, constant_data.accuracy,
                       ccd.n_fine_sub, ccd.fine_dl_sub,
                       ccd.kernel_fft_sub, ccd.N_fft_sub);
    }
    break;
  }

  case INST_ARYELLE: {
    constant_data.aryelle_sigma_slope = 6.3984e-05;
    constant_data.aryelle_sigma_intercept = 3.3232e-03;
    // Aryelle uses the main fine grid; no separate kernel needed
    // (direct O(N²) integration inside loss_impl)
    constant_data.N_fft = 0;
    break;
  }
  } // end instrument switch

  // ── Compute n_params ───────────────────────────────────────────────
  int base_params = 0;
  switch (constant_data.model_type) {
  case MODEL_LTE:
    base_params = 3 + constant_data.n_elements;
    break;
  case MODEL_LTE_LAYERS:
    base_params = constant_data.n_layers * (3 + constant_data.n_elements);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    base_params = constant_data.n_layers * 3 + constant_data.n_elements;
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    base_params = 3 + constant_data.n_elements + 10; // 10 time params
    break;
  case MODEL_LTE_TWO_TIME:
    base_params = 7 + constant_data.n_elements; // 6 plasma + 1 weight
    break;
  }
  int extra = 0;
  if (constant_data.has_scale_factor) extra += 1;
  if (constant_data.instrument_type == INST_CHEMCAM) extra += 2; // scale_vio, scale_visnir
  constant_data.n_params = base_params + extra;

  return constant_data;
}

// =====================================================================
// Build parameter name list (matches forward-mode get_parameter_names)
// =====================================================================
static std::vector<std::string>
build_param_names(const Experiment &experiment,
                  const EnzymeConstData &data) {
  auto element_names = get_element_names(experiment);
  std::vector<std::string> names;

  switch (data.model_type) {
  case MODEL_LTE:
    names.push_back("temperature");
    names.push_back("electron_density");
    names.push_back("length");
    for (const auto &el : element_names)
      names.push_back(el);
    break;

  case MODEL_LTE_LAYERS:
    for (int layer = 0; layer < data.n_layers; layer++) {
      std::string prefix = "layer_" + std::to_string(layer) + "_";
      names.push_back(prefix + "temperature");
      names.push_back(prefix + "electron_density");
      names.push_back(prefix + "length");
      for (const auto &el : element_names)
        names.push_back(prefix + el);
    }
    break;

  case MODEL_LTE_SHARED_LAYERS:
    for (int layer = 0; layer < data.n_layers; layer++) {
      std::string prefix = "layer_" + std::to_string(layer) + "_";
      names.push_back(prefix + "temperature");
      names.push_back(prefix + "electron_density");
      names.push_back(prefix + "length");
    }
    for (const auto &el : element_names)
      names.push_back(el);
    break;

  case MODEL_LTE_TIME_INTEGRATED:
    names.push_back("temperature");
    names.push_back("electron_density");
    names.push_back("length");
    for (const auto &el : element_names)
      names.push_back(el);
    names.push_back("start_temperature");
    names.push_back("end_temperature");
    names.push_back("tau_temperature");
    names.push_back("start_electron_density");
    names.push_back("end_electron_density");
    names.push_back("tau_electron_density");
    names.push_back("start_length");
    names.push_back("end_length");
    names.push_back("tau_length");
    names.push_back("integration_time");
    break;

  case MODEL_LTE_TWO_TIME:
    names.push_back("early_temperature");
    names.push_back("early_electron_density");
    names.push_back("early_length");
    names.push_back("delta_temperature");
    names.push_back("delta_electron_density");
    names.push_back("delta_length");
    names.push_back("weight_early");
    for (const auto &el : element_names)
      names.push_back(el);
    break;
  }

  if (data.has_scale_factor)
    names.push_back("scale_factor");
  if (data.instrument_type == INST_CHEMCAM) {
    names.push_back("scale_vio");
    names.push_back("scale_visnir");
  }

  return names;
}

// =====================================================================
// Fill flat parameter array from Experiment
// =====================================================================
static std::vector<double>
fill_params(const Experiment &experiment,
            const EnzymeConstData &data) {
  auto element_names = get_element_names(experiment);
  std::vector<double> params(data.n_params);

  int idx = 0;
  switch (data.model_type) {
  case MODEL_LTE: {
    const auto *lte = dynamic_cast<const LTEModel *>(
        experiment.plasma_model_ptr.get());
    params[idx++] = lte->temperature;
    params[idx++] = lte->electron_density;
    params[idx++] = lte->length;
    for (int e = 0; e < data.n_elements; e++)
      params[idx++] = lte->element_ratio.at(element_names[e]);
    break;
  }

  case MODEL_LTE_LAYERS: {
    const auto *lay = dynamic_cast<const LTELayers *>(
        experiment.plasma_model_ptr.get());
    for (int layer = 0; layer < data.n_layers; layer++) {
      const auto &l = lay->lte_layers[layer];
      params[idx++] = static_cast<double>(l.temperature);
      params[idx++] = static_cast<double>(l.electron_density);
      params[idx++] = static_cast<double>(l.length);
      for (int e = 0; e < data.n_elements; e++)
        params[idx++] = static_cast<double>(l.element_ratio.at(element_names[e]));
    }
    break;
  }

  case MODEL_LTE_SHARED_LAYERS: {
    const auto *shared = dynamic_cast<const LTESharedLayers *>(
        experiment.plasma_model_ptr.get());
    for (int layer = 0; layer < data.n_layers; layer++) {
      params[idx++] = static_cast<double>(shared->temperatures[layer]);
      params[idx++] = static_cast<double>(shared->electron_densities[layer]);
      params[idx++] = static_cast<double>(shared->lengths[layer]);
    }
    for (int e = 0; e < data.n_elements; e++)
      params[idx++] =
          static_cast<double>(shared->element_ratio.at(element_names[e]));
    break;
  }

  case MODEL_LTE_TIME_INTEGRATED: {
    const auto *ti = dynamic_cast<const LTETimeIntegrated *>(
        experiment.plasma_model_ptr.get());
    params[idx++] = static_cast<double>(ti->temperature);
    params[idx++] = static_cast<double>(ti->electron_density);
    params[idx++] = static_cast<double>(ti->length);
    for (int e = 0; e < data.n_elements; e++)
      params[idx++] = static_cast<double>(ti->element_ratio.at(element_names[e]));
    params[idx++] = static_cast<double>(ti->start_temperature);
    params[idx++] = static_cast<double>(ti->end_temperature);
    params[idx++] = static_cast<double>(ti->tau_temperature);
    params[idx++] = static_cast<double>(ti->start_electron_density);
    params[idx++] = static_cast<double>(ti->end_electron_density);
    params[idx++] = static_cast<double>(ti->tau_electron_density);
    params[idx++] = static_cast<double>(ti->start_length);
    params[idx++] = static_cast<double>(ti->end_length);
    params[idx++] = static_cast<double>(ti->tau_length);
    params[idx++] = static_cast<double>(ti->integration_time);
    break;
  }

  case MODEL_LTE_TWO_TIME: {
    const auto *tt = dynamic_cast<const LTETwoTime *>(
        experiment.plasma_model_ptr.get());
    params[idx++] = static_cast<double>(tt->early_temperature);
    params[idx++] = static_cast<double>(tt->early_electron_density);
    params[idx++] = static_cast<double>(tt->early_length);
    params[idx++] = static_cast<double>(tt->delta_temperature);
    params[idx++] = static_cast<double>(tt->delta_electron_density);
    params[idx++] = static_cast<double>(tt->delta_length);
    params[idx++] = static_cast<double>(tt->weight_early);
    for (int e = 0; e < data.n_elements; e++)
      params[idx++] = static_cast<double>(tt->element_ratio.at(element_names[e]));
    break;
  }
  }

  if (data.has_scale_factor)
    params[idx++] = static_cast<double>(experiment.scale_factor);

  if (data.instrument_type == INST_CHEMCAM) {
    const auto *cc = dynamic_cast<const ChemCam *>(
        experiment.instrument_ptr.get());
    params[idx++] = static_cast<double>(cc->scale_vio);
    params[idx++] = static_cast<double>(cc->scale_visnir);
  }

  return params;
}

// =====================================================================
// Weighted forward function: g(params, v) = Σ_k v[k] * y_sim_k(params)
// One reverse-mode pass computes J^T * v where J(i,j) = ∂y_sim_i/∂param_j
// =====================================================================
template <typename Model, typename Instrument>
static double __attribute__((noinline))
weighted_forward_impl(double *params, const double *aki,
                             const double *stark,
                             const EnzymeConstData *data,
                             const double *weights) {
  std::vector<double> y_sim(data->n_out, 0.0);
  Instrument::template compute_y_sim<Model>(params, aki, stark, data, y_sim);

  double result = 0.0;
  for (long i = 0; i < data->n_out; i++)
    result += weights[i] * y_sim[i];
  return result;
}

// Pickers for the weighted forward function
using EnzymeWeightedForwardFn = double (*)(double *, const double *,
                                            const double *,
                                            const EnzymeConstData *,
                                            const double *);

template <typename Model>
static EnzymeWeightedForwardFn
pick_weighted_instrument(int instrument_type) {
  switch (instrument_type) {
  case INST_GENERIC:
    return &weighted_forward_impl<Model, GenericInstrumentTraits>;
  case INST_SLOPING:
    return &weighted_forward_impl<Model, SlopingInstrumentTraits>;
  case INST_CHEMCAM:
    return &weighted_forward_impl<Model, ChemCamInstrumentTraits>;
  case INST_ARYELLE:
    return &weighted_forward_impl<Model, AryelleInstrumentTraits>;
  }
  return nullptr;
}

static EnzymeWeightedForwardFn
pick_weighted_fn(int model_type, int instrument_type) {
  switch (model_type) {
  case MODEL_LTE:
    return pick_weighted_instrument<LTEModelTraits>(instrument_type);
  case MODEL_LTE_LAYERS:
    return pick_weighted_instrument<LTELayersTraits>(instrument_type);
  case MODEL_LTE_SHARED_LAYERS:
    return pick_weighted_instrument<LTESharedLayersTraits>(instrument_type);
  case MODEL_LTE_TIME_INTEGRATED:
    return pick_weighted_instrument<LTETimeIntegratedTraits>(instrument_type);
  case MODEL_LTE_TWO_TIME:
    return pick_weighted_instrument<LTETwoTimeTraits>(instrument_type);
  }
  return nullptr;
}

// Reverse-mode pass that differentiates g(params, aki, v) = Σ_k v[k] * y_sim_k
// w.r.t. `aki` instead of `params`.  One pass with v = adjoint returns J_A^T v,
// where J_A(i, l) = ∂y_sim_i / ∂A_l.
template <typename Model, typename Instrument>
static __attribute__((noinline)) void
aki_jvp_pass(double *params, const double *aki, double *daki,
                    const double *stark, const EnzymeConstData *data,
                    const double *weights, double *weights_shadow) {
  __enzyme_autodiff<void>(
      reinterpret_cast<void *>(&weighted_forward_impl<Model, Instrument>),
      enzyme_const, params,
      enzyme_dup, aki, daki,
      enzyme_const, stark,
      enzyme_const, data,
      enzyme_dup, weights, weights_shadow);
}

// Reverse-mode pass that differentiates g w.r.t. the Stark multipliers.
// One pass with v = adjoint returns (J_m^T v)_l, where J_m(i, l) =
// ∂y_sim_i / ∂m_l.
template <typename Model, typename Instrument>
static __attribute__((noinline)) void
stark_jvp_pass(double *params, const double *aki,
                      const double *stark, double *dstark,
                      const EnzymeConstData *data,
                      const double *weights, double *weights_shadow) {
  __enzyme_autodiff<void>(
      reinterpret_cast<void *>(&weighted_forward_impl<Model, Instrument>),
      enzyme_const, params,
      enzyme_const, aki,
      enzyme_dup, stark, dstark,
      enzyme_const, data,
      enzyme_dup, weights, weights_shadow);
}

// Dispatchers for the Aki-VJP: with v = adjoint, gives Σ_k v[k] ∂y_sim_k/∂A_l
// = (J_A^T v)_l, where J_A(i, l) = ∂y_sim_i/∂A_l.
template <typename Model>
static void dispatch_aki_jvp_instrument(int instrument_type, double *params,
                                        const double *aki, double *daki,
                                        const double *stark,
                                        const EnzymeConstData *data,
                                        const double *weights,
                                        double *weights_shadow) {
  switch (instrument_type) {
  case INST_GENERIC:
    aki_jvp_pass<Model, GenericInstrumentTraits>(params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case INST_SLOPING:
    aki_jvp_pass<Model, SlopingInstrumentTraits>(params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case INST_CHEMCAM:
    aki_jvp_pass<Model, ChemCamInstrumentTraits>(params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case INST_ARYELLE:
    aki_jvp_pass<Model, AryelleInstrumentTraits>(params, aki, daki, stark, data, weights, weights_shadow);
    break;
  }
}

static void dispatch_aki_jvp(int model_type, int instrument_type,
                             double *params, const double *aki, double *daki,
                             const double *stark,
                             const EnzymeConstData *data,
                             const double *weights, double *weights_shadow) {
  switch (model_type) {
  case MODEL_LTE:
    dispatch_aki_jvp_instrument<LTEModelTraits>(instrument_type, params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_LAYERS:
    dispatch_aki_jvp_instrument<LTELayersTraits>(instrument_type, params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    dispatch_aki_jvp_instrument<LTESharedLayersTraits>(instrument_type, params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    dispatch_aki_jvp_instrument<LTETimeIntegratedTraits>(instrument_type, params, aki, daki, stark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_TWO_TIME:
    dispatch_aki_jvp_instrument<LTETwoTimeTraits>(instrument_type, params, aki, daki, stark, data, weights, weights_shadow);
    break;
  }
}

// Dispatchers for the Stark-VJP: (J_m^T v)_l where J_m(i, l) = ∂y_sim_i/∂m_l.
template <typename Model>
static void dispatch_stark_jvp_instrument(int instrument_type, double *params,
                                          const double *aki,
                                          const double *stark, double *dstark,
                                          const EnzymeConstData *data,
                                          const double *weights,
                                          double *weights_shadow) {
  switch (instrument_type) {
  case INST_GENERIC:
    stark_jvp_pass<Model, GenericInstrumentTraits>(params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case INST_SLOPING:
    stark_jvp_pass<Model, SlopingInstrumentTraits>(params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case INST_CHEMCAM:
    stark_jvp_pass<Model, ChemCamInstrumentTraits>(params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case INST_ARYELLE:
    stark_jvp_pass<Model, AryelleInstrumentTraits>(params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  }
}

static void dispatch_stark_jvp(int model_type, int instrument_type,
                               double *params, const double *aki,
                               const double *stark, double *dstark,
                               const EnzymeConstData *data,
                               const double *weights,
                               double *weights_shadow) {
  switch (model_type) {
  case MODEL_LTE:
    dispatch_stark_jvp_instrument<LTEModelTraits>(instrument_type, params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_LAYERS:
    dispatch_stark_jvp_instrument<LTELayersTraits>(instrument_type, params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    dispatch_stark_jvp_instrument<LTESharedLayersTraits>(instrument_type, params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    dispatch_stark_jvp_instrument<LTETimeIntegratedTraits>(instrument_type, params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  case MODEL_LTE_TWO_TIME:
    dispatch_stark_jvp_instrument<LTETwoTimeTraits>(instrument_type, params, aki, stark, dstark, data, weights, weights_shadow);
    break;
  }
}

// =====================================================================
// Forward-mode Jacobian: one __enzyme_fwddiff pass per parameter.
// Cost = n_params forward passes (vs n_out reverse passes in the
// reverse-mode variant above), which is the right asymptotic when
// n_out >> n_params -- the typical LIBS case.
// =====================================================================

// Spectrum-producing kernel that Enzyme can forward-differentiate.
// Writes the simulated spectrum into y_sim_out[0..n_out).
template <typename Model, typename Instrument>
static void __attribute__((noinline))
spectrum_impl(double *params, const double *aki, const double *stark,
                     const EnzymeConstData *data, double *y_sim_out) {
  std::vector<double> y_sim(data->n_out, 0.0);
  Instrument::template compute_y_sim<Model>(params, aki, stark, data, y_sim);
  for (long i = 0; i < data->n_out; i++)
    y_sim_out[i] = y_sim[i];
}

// One forward-mode pass: pushes the tangent `dparams` through the
// simulator and writes dy_sim = J * dparams.
template <typename Model, typename Instrument>
static __attribute__((noinline)) void
fwd_pass(double *params, double *dparams, const double *aki,
                const double *stark, const EnzymeConstData *data,
                double *y_sim, double *dy_sim) {
  __enzyme_fwddiff<void>(
      reinterpret_cast<void *>(&spectrum_impl<Model, Instrument>),
      enzyme_dup, params, dparams,
      enzyme_const, aki,
      enzyme_const, stark,
      enzyme_const, data,
      enzyme_dup, y_sim, dy_sim);
}

template <typename Model>
static void dispatch_fwd_instrument(int instrument_type, double *params,
                                    double *dparams, const double *aki,
                                    const double *stark,
                                    const EnzymeConstData *data,
                                    double *y_sim, double *dy_sim) {
  switch (instrument_type) {
  case INST_GENERIC:
    fwd_pass<Model, GenericInstrumentTraits>(
        params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case INST_SLOPING:
    fwd_pass<Model, SlopingInstrumentTraits>(
        params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case INST_CHEMCAM:
    fwd_pass<Model, ChemCamInstrumentTraits>(
        params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case INST_ARYELLE:
    fwd_pass<Model, AryelleInstrumentTraits>(
        params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  }
}

static void dispatch_fwd(int model_type, int instrument_type,
                         double *params, double *dparams, const double *aki,
                         const double *stark,
                         const EnzymeConstData *data,
                         double *y_sim, double *dy_sim) {
  switch (model_type) {
  case MODEL_LTE:
    dispatch_fwd_instrument<LTEModelTraits>(
        instrument_type, params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case MODEL_LTE_LAYERS:
    dispatch_fwd_instrument<LTELayersTraits>(
        instrument_type, params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case MODEL_LTE_SHARED_LAYERS:
    dispatch_fwd_instrument<LTESharedLayersTraits>(
        instrument_type, params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case MODEL_LTE_TIME_INTEGRATED:
    dispatch_fwd_instrument<LTETimeIntegratedTraits>(
        instrument_type, params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  case MODEL_LTE_TWO_TIME:
    dispatch_fwd_instrument<LTETwoTimeTraits>(
        instrument_type, params, dparams, aki, stark, data, y_sim, dy_sim);
    break;
  }
}

Eigen::MatrixXd
get_jacobian(const Experiment &experiment_double) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto params = fill_params(experiment_double, data);

  std::vector<double> dparams(data.n_params, 0.0);
  std::vector<double> y_sim(data.n_out, 0.0);
  std::vector<double> dy_sim(data.n_out, 0.0);

  Eigen::MatrixXd J(data.n_out, data.n_params);

  for (int j = 0; j < data.n_params; j++) {
    std::fill(dparams.begin(), dparams.end(), 0.0);
    std::fill(y_sim.begin(),   y_sim.end(),   0.0);
    std::fill(dy_sim.begin(),  dy_sim.end(),  0.0);
    dparams[j] = 1.0;
    dispatch_fwd(data.model_type, data.instrument_type,
                 params.data(), dparams.data(),
                 data.aki_active.data(), data.stark_active.data(), &data,
                 y_sim.data(), dy_sim.data());
    for (long i = 0; i < data.n_out; i++)
      J(i, j) = dy_sim[i];
  }

  return J;
}

// =====================================================================
// Public API: compute loss (for validation against loss_func<double>)
// =====================================================================
double compute_loss(const Experiment &experiment_double) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto params = fill_params(experiment_double, data);
  EnzymeLossFn loss_fn = pick_loss_fn(data.model_type, data.instrument_type);
  return loss_fn(params.data(), data.aki_active.data(),
                 data.stark_active.data(), &data);
}

// =====================================================================
// Public API: reverse-mode gradients via Enzyme
// =====================================================================
std::map<std::string, double>
get_gradients(const Experiment &experiment_double) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto param_names = build_param_names(experiment_double, data);
  auto params = fill_params(experiment_double, data);
  std::vector<double> grads(data.n_params, 0.0);

  // Single reverse-mode pass → all gradients.
  // Dispatch must use a direct symbol (Enzyme limitation) — see
  // dispatch_grad above.
  dispatch_grad(data.model_type, data.instrument_type, params.data(),
                grads.data(), data.aki_active.data(),
                data.stark_active.data(), &data);

  std::map<std::string, double> gradient_map;
  for (int i = 0; i < data.n_params; i++)
    gradient_map[param_names[i]] = grads[i];

  return gradient_map;
}

// =====================================================================
// Public API: ∂L/∂A_ji — reverse-mode gradient of the loss w.r.t. the
// Einstein A coefficients of every emission line in the active database
// (in build_const_data's iteration order: ion_line_groups, then .lines).
// =====================================================================
std::vector<double>
get_aki_gradient_impl(const Experiment &experiment_double) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto params = fill_params(experiment_double, data);

  std::vector<double> aki = data.aki_active;  // mutable copy: Enzyme writes shadow
  std::vector<double> daki(aki.size(), 0.0);

  dispatch_aki_grad(data.model_type, data.instrument_type, params.data(),
                    aki.data(), daki.data(), data.stark_active.data(), &data);
  return daki;
}

Eigen::VectorXd
get_aki_gradient(const Experiment &experiment_double) {
  auto daki = get_aki_gradient_impl(experiment_double);
  return Eigen::Map<Eigen::VectorXd>(daki.data(),
                                      static_cast<Eigen::Index>(daki.size()))
      .eval();
}

// =====================================================================
// Public API: spectrum-VJP w.r.t. Aki — returns (J_A)^T @ adjoint where
//   J_A(i, l) = ∂ y_sim_i / ∂ A_l
// One Enzyme reverse pass uses g(params, aki, v) = Σ_k v[k] * y_sim_k as
// the seed; differentiating g w.r.t. aki yields (J_A)^T v.
// =====================================================================
Eigen::VectorXd
spectrum_vjp_aki(const Experiment &experiment_double,
                 const Eigen::VectorXd &adjoint) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto params = fill_params(experiment_double, data);

  if (static_cast<long>(adjoint.size()) != data.n_out) {
    throw std::runtime_error(
        "spectrum_vjp_aki: adjoint length (" + std::to_string(adjoint.size()) +
        ") must equal n_out (" + std::to_string(data.n_out) + ")");
  }

  std::vector<double> aki = data.aki_active;
  std::vector<double> daki(aki.size(), 0.0);
  std::vector<double> weights(adjoint.data(), adjoint.data() + adjoint.size());
  std::vector<double> weights_shadow(adjoint.size(), 0.0);

  dispatch_aki_jvp(data.model_type, data.instrument_type, params.data(),
                   aki.data(), daki.data(), data.stark_active.data(), &data,
                   weights.data(), weights_shadow.data());
  return Eigen::Map<Eigen::VectorXd>(daki.data(),
                                      static_cast<Eigen::Index>(daki.size()))
      .eval();
}

// =====================================================================
// Public API: ∂L/∂m_l — reverse-mode gradient of the loss w.r.t. the
// per-line Stark FWHM multipliers.  Same layout as the Aki gradient
// (build_const_data's iteration order).
// =====================================================================
Eigen::VectorXd
get_stark_gradient(const Experiment &experiment_double) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto params = fill_params(experiment_double, data);

  std::vector<double> stark = data.stark_active;  // mutable copy
  std::vector<double> dstark(stark.size(), 0.0);

  dispatch_stark_grad(data.model_type, data.instrument_type, params.data(),
                      data.aki_active.data(), stark.data(), dstark.data(),
                      &data);
  return Eigen::Map<Eigen::VectorXd>(dstark.data(),
                                      static_cast<Eigen::Index>(dstark.size()))
      .eval();
}

// =====================================================================
// Public API: spectrum-VJP w.r.t. Stark multipliers — returns
//   (J_m)^T @ adjoint  where  J_m(i, l) = ∂y_sim_i / ∂m_l
// =====================================================================
Eigen::VectorXd
spectrum_vjp_stark(const Experiment &experiment_double,
                   const Eigen::VectorXd &adjoint) {
  EnzymeConstData data = build_const_data(experiment_double);
  auto params = fill_params(experiment_double, data);

  if (static_cast<long>(adjoint.size()) != data.n_out) {
    throw std::runtime_error(
        "spectrum_vjp_stark: adjoint length (" +
        std::to_string(adjoint.size()) + ") must equal n_out (" +
        std::to_string(data.n_out) + ")");
  }

  std::vector<double> stark = data.stark_active;
  std::vector<double> dstark(stark.size(), 0.0);
  std::vector<double> weights(adjoint.data(), adjoint.data() + adjoint.size());
  std::vector<double> weights_shadow(adjoint.size(), 0.0);

  dispatch_stark_jvp(data.model_type, data.instrument_type, params.data(),
                     data.aki_active.data(), stark.data(), dstark.data(),
                     &data, weights.data(), weights_shadow.data());
  return Eigen::Map<Eigen::VectorXd>(dstark.data(),
                                      static_cast<Eigen::Index>(dstark.size()))
      .eval();
}

} // namespace libskit
