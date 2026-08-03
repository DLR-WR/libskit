# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""Tests for libskit.gradient_descent.custom_gradient_descent.

Locks in the post-refactor behaviour of the helpers introduced in B-Py3
(commit 165e2e8): _init_history, _record_state, _snapshot_parameters,
_apply_gradient_step, _relative_changes, _compute_quality_metrics.
"""

import numpy as np
import pytest

import libskit
from libskit.gradient_descent import custom_gradient_descent


def _make_experiment(true_temperature: float = 11000.0,
                     init_temperature: float = 9500.0):
    """Build a small synthetic Experiment with one element (Fe)."""
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    wls = np.arange(380.0, 410.0, 0.05)

    true_model = libskit.LTEModel({"Fe": 1.0}, true_temperature, 5e23, 0.5e-3)
    true_irradiance = spec.calculate_spectrum(wls, true_model)
    true_spectrum = libskit.Spectrum(wls, true_irradiance)

    init_model = libskit.LTEModel({"Fe": 1.0}, init_temperature, 4e23, 0.4e-3)
    return libskit.Experiment(libskit.LossType.L2Loss, init_model, spec, true_spectrum)


class TestHistoryShape:
    """The history dict structure is part of the public API surface."""

    def test_history_has_all_required_keys(self) -> None:
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=3,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
        )
        h = result["history"]
        for key in ("loss", "temperature", "electron_density", "length",
                    "learning_rate", "rmse", "chi_squared", "r_squared", "Fe"):
            assert key in h, f"missing history key: {key}"

    def test_history_lengths_match(self) -> None:
        """Every history list must have len == iterations + 1 (initial state included)."""
        exp = _make_experiment()
        n_iter = 5
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=n_iter,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
        )
        h = result["history"]
        expected_len = n_iter + 1
        for key in ("loss", "temperature", "electron_density", "length",
                    "rmse", "chi_squared", "r_squared", "Fe"):
            assert len(h[key]) == expected_len, (
                f"{key}: expected {expected_len}, got {len(h[key])}"
            )

    def test_iteration_count_matches_history(self) -> None:
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=4,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
        )
        assert result["iterations"] == len(result["history"]["loss"]) - 1


class TestParameterUpdates:
    """Verify the gradient step actually updates parameters."""

    def test_temperature_changes_after_iteration(self) -> None:
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=3,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
        )
        temps = result["history"]["temperature"]
        # Should have moved at least once
        assert any(t != temps[0] for t in temps[1:])

    def test_element_ratios_normalized(self) -> None:
        """For a single element the ratio must remain at 1.0 after normalisation."""
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=5,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
        )
        for fe_value in result["history"]["Fe"]:
            assert abs(fe_value - 1.0) < 1e-12

    def test_floors_respected(self) -> None:
        """Length is floored at 1e-7 m and electron density at 0."""
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=5,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
        )
        h = result["history"]
        assert all(L >= 1e-7 for L in h["length"])
        assert all(ne >= 0.0 for ne in h["electron_density"])


class TestConvergence:
    """Convergence detection logic."""

    def test_convergence_flag_for_tight_threshold(self) -> None:
        """With a permissive threshold, GD should declare convergence quickly."""
        exp = _make_experiment(true_temperature=10000.0, init_temperature=10000.0)
        result = custom_gradient_descent(
            exp, learning_rate=1e-7, max_iterations=20,
            parameter_change_threshold=1.0,  # very loose
            parameter_change_patience=2,
        )
        assert result["converged"] is True
        assert result["iterations"] <= 20

    def test_max_iterations_is_safety_limit(self) -> None:
        """With an unreachable threshold and patience, iter count == max_iterations."""
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=4,
            parameter_change_threshold=1e-30, parameter_change_patience=999,
        )
        assert result["converged"] is False
        assert result["iterations"] == 4


class TestLearningRateDecay:
    """Adaptive learning-rate decay tracking."""

    def test_learning_rate_recorded_per_iteration(self) -> None:
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=3,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
            lr_decay_factor=0.5, lr_decay_patience=2,
        )
        assert all(lr > 0 for lr in result["history"]["learning_rate"])

    def test_decay_does_not_increase_lr(self) -> None:
        """Learning rate is monotonically non-increasing across iterations."""
        exp = _make_experiment()
        result = custom_gradient_descent(
            exp, learning_rate=1e-4, max_iterations=10,
            parameter_change_threshold=1e-9, parameter_change_patience=999,
            lr_decay_factor=0.5, lr_decay_patience=2,
        )
        lrs = result["history"]["learning_rate"]
        for prev, curr in zip(lrs, lrs[1:]):
            assert curr <= prev + 1e-30
