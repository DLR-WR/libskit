#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Tests for libskit.optimizer.Optimizer (NLopt wrapper)."""

from __future__ import annotations

import nlopt
import numpy as np
import pytest

import libskit
from libskit import Parametrization
from libskit.optimizer import Optimizer, _infer_bounds, _is_element_param


def _make_experiment(true_T: float = 11000.0, init_T: float = 9500.0):
    spec = libskit.GenericSpectrometer(0.1, 0.0)
    wls = np.arange(380.0, 410.0, 0.05)
    true_model = libskit.LTEModel({"Fe": 1.0}, true_T, 5e23, 0.5e-3)
    true_irradiance = spec.calculate_spectrum(wls, true_model)
    true_spectrum = libskit.Spectrum(wls, true_irradiance)
    init_model = libskit.LTEModel({"Fe": 1.0}, init_T, 4e23, 0.4e-3)
    return libskit.Experiment(libskit.LossType.L2Loss, init_model, spec, true_spectrum)


class TestBoundsHelpers:
    @pytest.mark.parametrize(
        "name,expected_lo,expected_hi",
        [("temperature", 5_000.0, 50_000.0),
         ("electron_density", 1e22, 1e24),
         ("length", 1e-5, 1e-2),
         ("tau", 1e-9, 1e-3),
         ("integration_time", 1e-9, 1e-3)],
    )
    def test_infer_bounds_known_params(self, name: str,
                                       expected_lo: float, expected_hi: float) -> None:
        lo, hi = _infer_bounds(name)
        assert lo == expected_lo
        assert hi == expected_hi

    def test_infer_bounds_substring_match(self) -> None:
        # Substring 'temperature' in 'layer_temperature_0' triggers physics bounds
        lo, hi = _infer_bounds("layer_temperature_0")
        assert (lo, hi) == (5_000.0, 50_000.0)

    def test_infer_bounds_falls_back_to_element(self) -> None:
        lo, hi = _infer_bounds("Fe")
        assert lo == 1e-8
        assert hi == 1.0

    @pytest.mark.parametrize("name", ["Fe", "Ca", "Si", "Mg"])
    def test_is_element_param_true(self, name: str) -> None:
        assert _is_element_param(name) is True

    @pytest.mark.parametrize("name", ["temperature", "electron_density", "length",
                                       "layer_temperature_0", "scale_factor"])
    def test_is_element_param_false(self, name: str) -> None:
        assert _is_element_param(name) is False


class TestOptimizerConstruction:
    def test_discovers_parameters_from_model(self) -> None:
        opt = Optimizer(_make_experiment())
        # Single-element LTEModel: temperature, electron_density, length, Fe
        assert "temperature" in opt.param_names
        assert "electron_density" in opt.param_names
        assert "length" in opt.param_names
        assert "Fe" in opt.param_names
        assert opt.n_params == len(opt.param_names)

    def test_initial_x0_matches_model(self) -> None:
        opt = Optimizer(_make_experiment(init_T=9500.0))
        idx = opt.param_names.index("temperature")
        assert opt._x0[idx] == pytest.approx(9500.0)

    def test_bounds_default_for_physics_params(self) -> None:
        opt = Optimizer(_make_experiment())
        idx = opt.param_names.index("temperature")
        assert opt._lower[idx] == 5_000.0
        assert opt._upper[idx] == 50_000.0

    def test_user_bounds_override(self) -> None:
        opt = Optimizer(_make_experiment(),
                        bounds={"temperature": (8_000.0, 15_000.0)})
        idx = opt.param_names.index("temperature")
        assert opt._lower[idx] == 8_000.0
        assert opt._upper[idx] == 15_000.0

    def test_element_mask_identifies_ratios(self) -> None:
        opt = Optimizer(_make_experiment())
        for i, name in enumerate(opt.param_names):
            assert opt._element_mask[i] == _is_element_param(name)

    def test_scaling_initial_value_is_unity_for_physics(self) -> None:
        """``LINEAR_RAW`` strategy: x0 * scaling ≈ 1 for physics params."""
        opt = Optimizer(_make_experiment(),
                        parametrization=Parametrization.LINEAR_RAW)
        scaled_x0 = opt._x0 * opt.scaling
        for i, name in enumerate(opt.param_names):
            if not opt._element_mask[i]:
                assert scaled_x0[i] == pytest.approx(1.0, rel=1e-9)


class TestOptimizerRun:
    def test_optimize_reduces_loss(self) -> None:
        from libskit._libskit import loss

        exp = _make_experiment(true_T=11000.0, init_T=9500.0)
        initial_loss = float(loss(exp))
        opt = Optimizer(exp)
        result = opt.optimize(algorithm=nlopt.LD_LBFGS, max_eval=20, verbose=False)
        assert result["status"] in ("success", "failed")
        assert result["final_loss"] <= initial_loss + 1e-12
        assert result["n_evaluations"] >= 1

    def test_history_recorded_per_eval(self) -> None:
        opt = Optimizer(_make_experiment())
        result = opt.optimize(algorithm=nlopt.LD_LBFGS, max_eval=10)
        n = result["n_evaluations"]
        for name in opt.param_names + ["loss", "iteration"]:
            assert name in result["history"]
            assert len(result["history"][name]) == n

    def test_gradient_free_algorithm_runs(self) -> None:
        opt = Optimizer(_make_experiment())
        result = opt.optimize(algorithm=nlopt.LN_NELDERMEAD, max_eval=15)
        assert result["n_evaluations"] >= 1
        assert "final_loss" in result

    def test_result_reports_nlopt_code_metadata(self) -> None:
        opt = Optimizer(_make_experiment())
        result = opt.optimize(algorithm=nlopt.LD_LBFGS, max_eval=50)
        for key in ("nlopt_result", "nlopt_result_name",
                    "reason", "converged", "status"):
            assert key in result
        assert isinstance(result["nlopt_result"], int)
        assert isinstance(result["nlopt_result_name"], str)
        assert isinstance(result["converged"], bool)

    def test_maxeval_reports_not_converged_but_success(self) -> None:
        """max_eval=1 forces NLopt to hit MAXEVAL_REACHED, which is a
        successful run (trustworthy final_loss) but not a converged one.
        """
        opt = Optimizer(_make_experiment())
        result = opt.optimize(algorithm=nlopt.LD_LBFGS, max_eval=1)
        assert result["status"] == "success"
        assert result["converged"] is False
        assert result["nlopt_result_name"] == "MAXEVAL_REACHED"

    def test_roundoff_limited_is_treated_as_success(self) -> None:
        """A very tight xtol_rel typically drives NLopt into
        ROUNDOFF_LIMITED (raised as an exception by the Python
        binding).  Our wrapper should still report it as a successful,
        converged run rather than a failure.

        This test does not *require* NLopt to hit ROUNDOFF_LIMITED
        (that depends on the algorithm's inner state); if a normal
        convergence code is returned instead the test is a no-op.
        """
        opt = Optimizer(_make_experiment())
        result = opt.optimize(
            algorithm=nlopt.LD_LBFGS, max_eval=500, xtol_rel=1e-15,
        )
        if result["nlopt_result_name"] == "ROUNDOFF_LIMITED":
            assert result["status"] == "success"
            assert result["converged"] is True
            assert "round-off" in result["reason"]

    def test_ftol_rel_short_circuits_near_optimum(self) -> None:
        """With a permissive ftol_rel, NLopt exits via FTOL_REACHED
        instead of grinding into a line-search failure at the optimum.
        """
        opt = Optimizer(_make_experiment())
        result = opt.optimize(
            algorithm=nlopt.LD_LBFGS,
            max_eval=500,
            xtol_rel=1e-15,
            ftol_rel=1e-8,
        )
        # Either FTOL_REACHED fires cleanly, or one of the other benign
        # convergence codes.  In every case the run must be a success.
        assert result["status"] == "success"
        assert result["converged"] is True


class TestLogSoftmaxParametrization:
    """Tests for the ``log_softmax`` family of parametrisations.

    The default strategy is :attr:`Parametrization.LOG_SOFTMAX_ALR`
    (ALR gauge fix); the plain :attr:`Parametrization.LOG_SOFTMAX`
    remains available and is exercised explicitly here.
    """

    def test_default_strategy_is_log_softmax_alr(self) -> None:
        opt = Optimizer(_make_experiment())
        assert opt._param_strategy.name == "log_softmax_alr"

    def test_explicit_log_softmax(self) -> None:
        opt = Optimizer(_make_experiment(),
                        parametrization=Parametrization.LOG_SOFTMAX)
        assert opt._param_strategy.name == "log_softmax"

    def test_round_trip_recovers_normalised_x0(self) -> None:
        opt = Optimizer(_make_experiment(init_T=9500.0))
        x0_int = np.asarray(opt._initial_internal)
        x_phys = opt._param_strategy.physical_from_internal(x0_int)
        # log-physics params recovered exactly
        for name in ("temperature", "electron_density", "length"):
            i = opt.param_names.index(name)
            assert x_phys[i] == pytest.approx(opt._x0[i], rel=1e-12)
        # element block sums to 1
        em = opt._element_mask
        assert x_phys[em].sum() == pytest.approx(1.0, abs=1e-12)
        assert (x_phys[em] >= 0).all()

    def test_internal_bounds_for_log_physics(self) -> None:
        opt = Optimizer(_make_experiment())
        i_T = opt.param_names.index("temperature")
        assert opt._lower_internal[i_T] == pytest.approx(np.log(opt._lower[i_T]))
        assert opt._upper_internal[i_T] == pytest.approx(np.log(opt._upper[i_T]))

    def test_internal_bounds_for_elements_are_logit_window(self) -> None:
        """Under the plain LOG_SOFTMAX strategy every element logit
        lives in ``[-20, 20]`` (the ALR default pins one of them to
        ``[0, 0]`` and is covered separately).
        """
        opt = Optimizer(_make_experiment(),
                        parametrization=Parametrization.LOG_SOFTMAX)
        for i, is_elem in enumerate(opt._element_mask):
            if is_elem:
                assert opt._lower_internal[i] == pytest.approx(-20.0)
                assert opt._upper_internal[i] == pytest.approx(20.0)

    def test_chain_rule_matches_finite_difference(self) -> None:
        """Chain rule applied to a *constructed* physical gradient matches
        the directional derivative of ``g_phys . physical_from_internal``
        in internal coordinates (i.e. the strategy's Jacobian is correct).

        We bypass ``get_gradients`` here because that gradient is
        scaled differently from the loss surface we'd FD against; the unit
        under test is the strategy, not enzyme.
        """
        opt = Optimizer(_make_experiment(true_T=11000.0, init_T=9500.0))
        strat = opt._param_strategy
        x_int = np.asarray(opt._initial_internal, dtype=float)

        # Construct a deterministic linear functional of x_phys: F(x_phys) = c . x_phys.
        # Then dF/dx_phys = c (a constant), and dF/dx_int = chain_rule(c, x_phys).
        rng = np.random.default_rng(0)
        c = rng.standard_normal(opt.n_params)
        c_dict = {n: float(c[i]) for i, n in enumerate(opt.param_names)}

        x_phys = strat.physical_from_internal(x_int)
        g_int = strat.chain_rule(c_dict, x_phys)

        # Central FD of F along several random unit directions
        eps = 1e-6
        for _ in range(3):
            v = rng.standard_normal(opt.n_params)
            v /= np.linalg.norm(v)
            xp = strat.physical_from_internal(x_int + eps * v)
            xm = strat.physical_from_internal(x_int - eps * v)
            fd = float(np.dot(c, (xp - xm))) / (2 * eps)
            ana = float(np.dot(g_int, v))
            assert ana == pytest.approx(fd, rel=1e-4, abs=1e-9)

    def test_optimize_runs_with_log_softmax(self) -> None:
        from libskit._libskit import loss

        exp = _make_experiment(true_T=11000.0, init_T=9500.0)
        L0 = float(loss(exp))
        opt = Optimizer(exp, parametrization=Parametrization.LOG_SOFTMAX)
        result = opt.optimize(algorithm=nlopt.LD_LBFGS, max_eval=20, verbose=False)
        assert result["final_loss"] <= L0 + 1e-12
