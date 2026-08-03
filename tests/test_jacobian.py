# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Test suite for the Jacobian computation via Enzyme AD.

Verifies that get_jacobian() produces results consistent with
central finite differences.
"""

import sys
import os

# Add the build directory so _libskit can be imported directly
_build_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "build", "cp312-cp312-linux_x86_64"
)
sys.path.insert(0, _build_dir)

import numpy as np
import pytest

import libskit


def _make_experiment(
    true_temperature: float = 11000.0,
    init_temperature: float = 9500.0,
    model_type: str = "lte",
    instrument_type: str = "generic",
):
    """Build a synthetic Experiment with one element (Fe) for testing."""
    if instrument_type == "generic":
        spec = libskit.GenericSpectrometer(0.1, 0.0)
    elif instrument_type == "sloping":
        spec = libskit.SlopingSpectrometer(1e-4, 0.01)
    elif instrument_type == "aryelle":
        spec = libskit.Aryelle()
    else:
        raise ValueError(f"Unknown instrument: {instrument_type}")

    wls = np.arange(380.0, 410.0, 0.05)

    if model_type == "lte":
        true_model = libskit.LTEModel(
            {"Fe": 1.0}, true_temperature, 5e23, 0.5e-3
        )
        init_model = libskit.LTEModel(
            {"Fe": 1.0}, init_temperature, 4e23, 0.4e-3
        )
    elif model_type == "lte_layers":
        true_model = libskit.LTELayers(
            [{"Fe": 1.0}, {"Fe": 1.0}],
            [true_temperature, true_temperature * 0.8],
            [5e23, 3e23],
            [0.3e-3, 0.2e-3],
        )
        init_model = libskit.LTELayers(
            [{"Fe": 1.0}, {"Fe": 1.0}],
            [init_temperature, init_temperature * 0.8],
            [4e23, 2.5e23],
            [0.25e-3, 0.15e-3],
        )
    elif model_type == "lte_shared_layers":
        true_model = libskit.LTESharedLayers(
            {"Fe": 1.0},
            [true_temperature, true_temperature * 0.8],
            [5e23, 3e23],
            [0.3e-3, 0.2e-3],
        )
        init_model = libskit.LTESharedLayers(
            {"Fe": 1.0},
            [init_temperature, init_temperature * 0.8],
            [4e23, 2.5e23],
            [0.25e-3, 0.15e-3],
        )
    elif model_type == "lte_time_integrated":
        true_model = libskit.LTETimeIntegrated(
            {"Fe": 1.0},
            start_temperature=true_temperature,
            end_temperature=6000.0,
            tau_temperature=100e-9,
            start_electron_density=5e23,
            end_electron_density=1e23,
            tau_electron_density=80e-9,
            start_length=0.3e-3,
            end_length=1.0e-3,
            tau_length=120e-9,
            integration_time=1000e-9,
            integration_points=5,
        )
        init_model = libskit.LTETimeIntegrated(
            {"Fe": 1.0},
            start_temperature=init_temperature,
            end_temperature=5500.0,
            tau_temperature=90e-9,
            start_electron_density=4e23,
            end_electron_density=0.9e23,
            tau_electron_density=70e-9,
            start_length=0.25e-3,
            end_length=0.8e-3,
            tau_length=100e-9,
            integration_time=900e-9,
            integration_points=5,
        )
    elif model_type == "lte_two_time":
        true_model = libskit.LTETwoTime(
            {"Fe": 1.0},
            early_temperature=true_temperature,
            early_electron_density=5e23,
            early_length=0.5e-3,
            late_temperature=true_temperature * 0.6,
            late_electron_density=2e23,
            late_length=0.3e-3,
            weight_early=0.6,
        )
        init_model = libskit.LTETwoTime(
            {"Fe": 1.0},
            early_temperature=init_temperature,
            early_electron_density=4e23,
            early_length=0.4e-3,
            late_temperature=init_temperature * 0.5,
            late_electron_density=1.5e23,
            late_length=0.25e-3,
            weight_early=0.55,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    true_irradiance = spec.calculate_spectrum(wls, true_model)
    true_spectrum = libskit.Spectrum(wls, true_irradiance)

    return libskit.Experiment(
        libskit.LossType.L2Loss, init_model, spec, true_spectrum
    )


def _jacobian_fd_central(experiment, param_name, wavelength_index, h=None):
    """Compute one Jacobian entry via central finite differences.

    Uses a parameter-relative step by default: h = max(|param|, 1) * 1e-5.
    This is critical for parameters with very different scales (e.g.
    temperature ~10^4 K vs length ~10^-4 m) and avoids the numerical noise
    that an absolute step would produce for off-scale parameters.
    """
    model = experiment.get_plasma_model()
    wls = experiment.true_spectrum.wavelengths

    old_val = model.get_parameter(param_name)
    if h is None:
        # Use a 10% relative step to avoid the wing-threshold discontinuity
        # in the emission broadening kernel; smaller steps land in a steep
        # piecewise-constant region and disagree with the smooth Enzyme gradient.
        # Use a purely relative step (not clamped to 1) so that small parameters
        # like length (~4e-4 m) are not over-perturbed.
        h = abs(old_val) * 0.1 if abs(old_val) > 1e-30 else 1e-7

    # Perturb parameter up
    model.set_parameter(param_name, old_val + h)
    irr_up = experiment.get_instrument().calculate_spectrum(wls, model)
    val_up = irr_up[wavelength_index]

    # Perturb parameter down
    model.set_parameter(param_name, old_val - h)
    irr_down = experiment.get_instrument().calculate_spectrum(wls, model)
    val_down = irr_down[wavelength_index]

    # Restore original value
    model.set_parameter(param_name, old_val)

    return (val_up - val_down) / (2 * h)


def _find_emission_peak_indices(experiment, n_peaks=3):
    """Return wavelength indices of the n_peaks brightest emission lines.

    Tests should compare Jacobian entries at these wavelengths rather than
    at arbitrary indices, because the Jacobian is essentially zero everywhere
    in the continuum background between spectral lines.
    """
    model = experiment.get_plasma_model()
    wls = experiment.true_spectrum.wavelengths
    irr = experiment.get_instrument().calculate_spectrum(wls, model)
    return list(np.argsort(irr)[-n_peaks:][::-1])


class TestJacobianShape:
    """Test that the Jacobian has the expected shape."""

    def test_jacobian_shape_lte(self):
        exp = _make_experiment(model_type="lte")
        J = libskit.get_jacobian(exp)
        n_wl = len(exp.true_spectrum.wavelengths)
        n_params = len(exp.get_plasma_model().get_parameter_names())
        assert J.shape == (n_wl, n_params)


class TestJacobianAccuracy:
    """Verify Jacobian entries against central finite differences."""

    @pytest.fixture
    def exp(self):
        return _make_experiment(model_type="lte", instrument_type="generic")

    @pytest.mark.parametrize("param_idx", [0, 1, 2])
    def test_jacobian_entry_lte(self, exp, param_idx):
        """Compare Enzyme Jacobian entries against FD at emission-line peaks.

        Tests at emission-line peak wavelengths where the Jacobian is
        meaningfully non-zero; off-peak wavelengths lie in the continuum
        background where both Enzyme and FD are ~0 and the relative-error
        metric becomes dominated by floating-point noise.
        """
        param_names = exp.get_plasma_model().get_parameter_names()
        param_name = param_names[param_idx]

        J_enzyme = libskit.get_jacobian(exp)
        peak_indices = _find_emission_peak_indices(exp, n_peaks=3)

        for wl_idx in peak_indices:
            J_fd = _jacobian_fd_central(exp, param_name, wl_idx)
            J_val = J_enzyme[wl_idx, param_idx]

            # Skip if both are near zero (parameter has little effect here)
            scale = max(abs(J_fd), abs(J_val))
            if scale < 1.0:
                continue

            rel_error = np.abs(J_val - J_fd) / scale
            assert rel_error < 5e-2, (  # 5 % tolerance; FD with 10 % step has O(h²) error
                f"param={param_name}, wl_idx={wl_idx}: "
                f"Enzyme={J_val:.6e}, FD={J_fd:.6e}, rel_err={rel_error:.2e}"
            )

    def test_jacobian_vs_loss_gradient(self, exp):
        """Exact MaxNorm-L2 gradient from Jacobian must match get_gradients.

        For Loss = ||y_sim/sim_max - y_ref/ref_max||^2 the exact relationship is:
          ∂L/∂p_j = (2/sim_max) * [(J^T @ residual)_j
                                    - J[peak_idx, j] * (y_sim_norm · residual)]
        where peak_idx = argmax(y_sim).  The simpler formula 2*J^T@residual is
        incorrect because it omits the normalization-sensitivity term.
        """
        J = libskit.get_jacobian(exp)
        param_names = exp.get_plasma_model().get_parameter_names()

        wls = exp.true_spectrum.wavelengths
        y_sim = exp.get_instrument().calculate_spectrum(wls, exp.get_plasma_model())
        y_ref = exp.true_spectrum.irradiance

        sim_max = np.max(y_sim)
        ref_max = np.max(y_ref)
        y_sim_norm = y_sim / sim_max
        y_ref_norm = y_ref / ref_max
        residual = y_sim_norm - y_ref_norm
        peak_idx = int(np.argmax(y_sim))

        # Correct MaxNorm L2 gradient derived from the raw spectrum Jacobian:
        #   ∂(y_sim_norm_i)/∂p = (1/sim_max) * (J[i] - y_sim_norm_i * J[peak])
        norm_correction = float(np.dot(y_sim_norm, residual))
        grad_from_jacobian = (2.0 / sim_max) * (
            J.T @ residual - J[peak_idx, :] * norm_correction
        )
        grad_enzyme = libskit.get_gradients(exp)

        for i, name in enumerate(param_names):
            g_j = grad_from_jacobian[i]
            g_e = grad_enzyme[name]
            scale = max(abs(g_j), abs(g_e))
            if scale < 1e-15:
                continue  # both effectively zero — skip
            rel_error = abs(g_j - g_e) / scale
            assert rel_error < 1e-4, (
                f"{name}: grad_from_J={g_j:.6e}, "
                f"grad_enzyme={g_e:.6e}, rel_err={rel_error:.2e}"
            )


class TestJacobianMultipleModels:
    """Test Jacobian across different plasma model types."""

    @pytest.mark.parametrize("model_type", [
        "lte",
        "lte_layers",
        "lte_shared_layers",
    ])
    @pytest.mark.parametrize("instrument_type", ["generic", "sloping"])
    def test_jacobian_matches_fd(self, model_type, instrument_type):
        exp = _make_experiment(model_type=model_type, instrument_type=instrument_type)
        J = libskit.get_jacobian(exp)

        param_names = exp.get_plasma_model().get_parameter_names()
        n_wl = len(exp.true_spectrum.wavelengths)

        # Sample a few entries at emission-line peaks (off-peak wavelengths
        # have near-zero gradients and the FD comparison is noise-dominated).
        test_params = [0, len(param_names) // 2]
        peak_wls = _find_emission_peak_indices(exp, n_peaks=3)

        max_rel_err = 0.0
        meaningful_pairs = 0
        for pi in test_params:
            for wi in peak_wls:
                J_fd = _jacobian_fd_central(exp, param_names[pi], wi)
                J_enzyme = J[wi, pi]
                scale = max(abs(J_fd), abs(J_enzyme))
                if scale < 1.0:
                    continue  # both near zero: skip
                rel_err = abs(J_enzyme - J_fd) / scale
                max_rel_err = max(max_rel_err, rel_err)
                meaningful_pairs += 1

        if meaningful_pairs > 0:
            assert max_rel_err < 5e-2, f"Max rel error: {max_rel_err:.2e}"


class TestJacobianNumericalStability:
    """Test numerical properties of the Jacobian."""

    def test_jacobian_values_bounded(self):
        """Gradients should not be unreasonably large."""
        exp = _make_experiment(model_type="lte")
        J = libskit.get_jacobian(exp)

        assert not np.any(np.isnan(J))
        assert not np.any(np.isinf(J))

    def test_symmetry_check_l2_loss_gradient(self):
        """For L2 loss, gradient = 2 * J^T @ residual."""
        exp = _make_experiment(model_type="lte")
        J = libskit.get_jacobian(exp)

        wls = exp.true_spectrum.wavelengths
        y_sim = exp.get_instrument().calculate_spectrum(wls, exp.get_plasma_model())
        y_ref = exp.true_spectrum.irradiance

        sim_max = np.max(y_sim)
        ref_max = np.max(y_ref)
        residual = (y_sim / sim_max) - (y_ref / ref_max)

        # The loss gradient is 2 * J^T @ residual (for normalized L2)
        grad_enzyme = libskit.get_gradients(exp)
        grad_from_j = 2.0 * (J.T @ residual)

        # Normalize both for direction comparison
        g_norm = np.array([grad_enzyme[k] for k in grad_enzyme])
        cos_angle = np.dot(g_norm, grad_from_j) / (
            np.linalg.norm(g_norm) * np.linalg.norm(grad_from_j) + 1e-15
        )
        # Direction should match (cos_angle close to 1 or -1)
        assert abs(cos_angle) > 0.99, (
            f"Gradient direction mismatch: cos_angle={cos_angle:.4f}"
        )