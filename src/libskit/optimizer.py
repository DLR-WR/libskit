#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""NLopt-based optimizer for plasma model fitting.

Provides a model-agnostic :class:`Optimizer` that works with any
:class:`PlasmaModel` subclass (``LTEModel``, ``LTELayers``,
``LTETimeIntegrated``, …) and any NLopt algorithm.

The module also provides :func:`convergence_analysis` for running
multi-start optimizations with Latin Hypercube Sampling.

Example
-------
>>> import libskit
>>> import nlopt
>>> optimizer = libskit.Optimizer(experiment)
>>> result = optimizer.optimize(algorithm=nlopt.LD_LBFGS, max_eval=500)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import nlopt

from ._libskit import (
    ChemCam,
    Experiment,
    NormalizationType,
    get_gradients,
    loss,
    optimal_scale_factor,
)
from ._parametrization import (
    LinearRawParametrization,
    LogSoftmaxParametrization,
    LogSoftmaxALRParametrization,
    Parametrization,
    ParametrizationStrategy,
)

# ---------------------------------------------------------------------------
# Default bound heuristics – keyed by substring of the parameter name.
# Order matters: first match wins, so "delta_*" entries (non-negative
# early-to-late differences) must precede the generic "temperature" /
# "electron_density" / "length" entries they would otherwise also match.
# ---------------------------------------------------------------------------
#
# The lower bounds are a small positive floor rather than exactly 0.0:
# LOG_SOFTMAX parametrization (the default) takes log(lower_physical), so an
# exact 0.0 bound would raise (log-physics params must be strictly positive).
# The floor is negligible compared to typical parameter magnitudes and still
# effectively enforces delta >= 0.
_DEFAULT_BOUNDS: list[tuple[str, float, float]] = [
    ("delta_temperature", 1.0, 45_000.0),
    ("delta_electron_density", 1e18, 1e24),
    ("delta_length", 1e-9, 1e-2),
    ("temperature", 5_000.0, 50_000.0),
    ("electron_density", 1e22, 1e24),
    ("length", 1e-5, 1e-2),
    ("tau", 1e-9, 1e-3),
    ("integration_time", 1e-9, 1e-3),
]

# Element ratios fall through to this default
_ELEMENT_BOUNDS = (1e-8, 1.0)


def _infer_bounds(name: str) -> tuple[float, float]:
    """Return sensible (lower, upper) bounds for *name*."""
    for substr, lo, hi in _DEFAULT_BOUNDS:
        if substr in name:
            return (lo, hi)
    return _ELEMENT_BOUNDS


def _is_element_param(name: str) -> bool:
    """Heuristic: element ratio parameters are short names without '_'."""
    return "_" not in name and name not in (
        "temperature",
        "electron_density",
        "length",
    )


# Gradient-based NLopt algorithms (need grad filled)
_GRADIENT_ALGORITHMS = {
    nlopt.LD_LBFGS,
    nlopt.LD_SLSQP,
    nlopt.LD_MMA,
    nlopt.LD_CCSAQ,
    nlopt.LD_TNEWTON,
    nlopt.LD_TNEWTON_PRECOND,
    nlopt.LD_TNEWTON_PRECOND_RESTART,
    nlopt.LD_TNEWTON_RESTART,
    nlopt.LD_VAR1,
    nlopt.LD_VAR2,
}


# NLopt return-code metadata --------------------------------------------
# Maps result code -> (short name, human-readable reason, is_converged).
# * "success" (in the top-level dict below) means NLopt did not raise a
#   fatal error and the reported final_loss is trustworthy.
# * "converged" means one of NLopt's convergence stopping criteria fired.
#   MAXEVAL / MAXTIME are treated as "success but not converged": the run
#   ran out of budget rather than genuinely stalling.
# * ROUNDOFF_LIMITED is treated as "success and converged": it means
#   further progress is prevented by floating-point round-off, i.e. we
#   are at machine precision, not that the run has failed.  Compare
#   NLopt docs: <https://nlopt.readthedocs.io/en/latest/NLopt_Reference/>.
_NLOPT_RESULT_INFO: Dict[int, tuple] = {
    nlopt.SUCCESS: (
        "SUCCESS", "generic success", True,
    ),
    nlopt.STOPVAL_REACHED: (
        "STOPVAL_REACHED", "objective reached stopval", True,
    ),
    nlopt.FTOL_REACHED: (
        "FTOL_REACHED", "objective tolerance ftol reached", True,
    ),
    nlopt.XTOL_REACHED: (
        "XTOL_REACHED", "parameter tolerance xtol reached", True,
    ),
    nlopt.MAXEVAL_REACHED: (
        "MAXEVAL_REACHED",
        "evaluation budget exhausted before a convergence criterion",
        False,
    ),
    nlopt.MAXTIME_REACHED: (
        "MAXTIME_REACHED",
        "time budget exhausted before a convergence criterion",
        False,
    ),
    nlopt.FAILURE: (
        "FAILURE", "generic NLopt failure", False,
    ),
    nlopt.INVALID_ARGS: (
        "INVALID_ARGS", "invalid arguments passed to NLopt", False,
    ),
    nlopt.OUT_OF_MEMORY: (
        "OUT_OF_MEMORY", "NLopt ran out of memory", False,
    ),
    nlopt.ROUNDOFF_LIMITED: (
        "ROUNDOFF_LIMITED",
        "further progress prevented by floating-point round-off "
        "(treated as converged at machine precision)",
        True,
    ),
    nlopt.FORCED_STOP: (
        "FORCED_STOP", "optimisation was forcibly stopped", False,
    ),
}


def _interpret_nlopt_result(code: int) -> tuple:
    """Return (name, reason, converged) for an NLopt result code."""
    if code in _NLOPT_RESULT_INFO:
        return _NLOPT_RESULT_INFO[code]
    return (f"UNKNOWN({code})", f"unknown NLopt code {code}", False)


# Result codes that we still treat as a *successful* (trustworthy) run,
# even though NLopt reports them via a raised exception.
_NLOPT_BENIGN_EXCEPTION_CODES = {nlopt.ROUNDOFF_LIMITED}


class Optimizer:
    """NLopt-based optimizer for any :class:`PlasmaModel`.

    The optimizer discovers parameters via `model.get_parameter_names()`
    and works with **any** plasma model (``LTEModel``, ``LTELayers``,
    ``LTETimeIntegrated``, …) without model-specific code.

    Parameters
    ----------
    experiment : Experiment
        The libskit experiment (contains model, instrument, spectrum,
        loss type).
    bounds : dict, optional
        Per-parameter bound overrides ``{name: (lower, upper)}``.
        Parameters not listed use auto-detected defaults.
    scaling : dict, optional
        Per-parameter scaling overrides ``{name: scale_factor}``.
        By default each parameter is scaled by ``1 / abs(initial_value)``
        so that the scaled value starts at ~1.
    normalize_elements : bool, optional
        If ``True`` (default), element-ratio parameters are renormalized
        to sum to 1 before each loss evaluation.  Only consulted when
        ``parametrization=Parametrization.LINEAR_RAW``; the
        ``LOG_SOFTMAX`` strategy enforces the simplex constraint exactly
        via softmax.
    parametrization : Parametrization, optional
        Coordinate system the optimizer presents to NLopt.  One of:

        * :attr:`Parametrization.LOG_SOFTMAX` (default): ``log T``,
          ``log n_e``, ``log L`` (and other strictly-positive
          multiplicative physics params) plus softmax-encoded element
          ratios.  Recommended for L-BFGS and other gradient-based
          solvers.
        * :attr:`Parametrization.LINEAR_RAW`: linear physics with
          ``1/x0`` scaling and raw element ratios.

        User-supplied ``bounds=`` are always interpreted in **physical**
        units regardless of the parametrization.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        experiment: Experiment,
        *,
        bounds: Optional[Dict[str, tuple[float, float]]] = None,
        scaling: Optional[Dict[str, float]] = None,
        normalize_elements: bool = True,
        parametrization: Parametrization = Parametrization.LOG_SOFTMAX_ALR,
    ) -> None:
        self.experiment = experiment
        self.model = experiment.get_plasma_model()
        self.normalize_elements = normalize_elements

        # Resolve the parametrization strategy from the enum value.
        if parametrization is Parametrization.LOG_SOFTMAX:
            self._param_strategy: ParametrizationStrategy = (
                LogSoftmaxParametrization()
            )
        elif parametrization is Parametrization.LOG_SOFTMAX_ALR:
            self._param_strategy = LogSoftmaxALRParametrization()
        else:  # LINEAR_RAW
            self._param_strategy = LinearRawParametrization(
                scaling=scaling,
                normalize_elements=normalize_elements,
            )

        # Discover parameters from the model
        self.param_names: list[str] = list(self.model.get_parameter_names())

        # --- OptimizedScaleNorm: add scale_factor as extra parameter ---
        self._has_scale_factor = (
            experiment.normalization_type
            == NormalizationType.OptimizedScaleNorm
        )
        if self._has_scale_factor:
            # Compute initial scale_factor from an analytic estimate
            instrument = experiment.get_instrument()
            wavelengths = experiment.true_spectrum.wavelengths
            y_sim = np.real(
                instrument.calculate_spectrum(wavelengths, self.model)
            )
            y_exp = np.asarray(experiment.true_spectrum.irradiance)
            c_init = float(optimal_scale_factor(y_sim, y_exp))
            if abs(c_init) < 1e-30:
                c_init = 1.0
            experiment.scale_factor = c_init
            self.param_names.append("scale_factor")

        # --- ChemCam per-CCD scale factors (VIO and VISNIR, UV is ref) ---
        self._instrument = experiment.get_instrument()
        self._has_chemcam_scales = isinstance(self._instrument, ChemCam)
        if self._has_chemcam_scales:
            self.param_names.extend(["scale_vio", "scale_visnir"])

        self.n_params = len(self.param_names)

        # Identify which parameters are element ratios
        self._element_mask = np.array(
            [_is_element_param(n) for n in self.param_names]
        )

        # Read initial values
        _extra_params = {"scale_factor", "scale_vio", "scale_visnir"}
        self._x0 = np.array(
            [float(self.model.get_parameter(n)) for n in self.param_names
             if n not in _extra_params]
        )
        if self._has_scale_factor:
            self._x0 = np.append(self._x0, experiment.scale_factor)
        if self._has_chemcam_scales:
            self._x0 = np.append(
                self._x0,
                [float(self._instrument.scale_vio),
                 float(self._instrument.scale_visnir)],
            )

        # --- Bounds (always physical, user-facing) --------------------
        user_bounds = bounds or {}
        self._lower = np.empty(self.n_params)
        self._upper = np.empty(self.n_params)
        for i, name in enumerate(self.param_names):
            if name in user_bounds:
                lo, hi = user_bounds[name]
            elif name == "scale_factor":
                # 2 orders of magnitude each way from analytic estimate
                sf = abs(self._x0[i])
                lo, hi = sf * 0.01, sf * 100.0
            elif name in ("scale_vio", "scale_visnir"):
                lo, hi = 0.01, 100.0
            else:
                lo, hi = _infer_bounds(name)
            self._lower[i] = lo
            self._upper[i] = hi

        # --- Bind the parametrization strategy ------------------------
        self._param_strategy.bind(
            param_names=self.param_names,
            initial_physical=self._x0,
            lower_physical=self._lower,
            upper_physical=self._upper,
            element_mask=self._element_mask,
        )

        # Internal-coord vectors that NLopt actually sees
        self._initial_internal = self._param_strategy.initial_internal
        self._lower_internal = self._param_strategy.lower_internal
        self._upper_internal = self._param_strategy.upper_internal

        # Backward-compat aliases.  Under ``linear_raw`` these reproduce
        # the historical ``scaling`` / ``_lower_scaled`` / ``_upper_scaled``
        # arrays exactly.  Under ``log_softmax`` they remain available so
        # downstream notebooks (e.g. convexity_landscape_probe) can still
        # consult per-parameter scale factors / scaled bounds, but the
        # numbers reflect the active strategy.
        if isinstance(self._param_strategy, LinearRawParametrization):
            self.scaling = self._param_strategy.scaling
        else:
            # No single "scaling" array makes sense for log/softmax.
            # Provide a vector of ones so legacy multiplications are no-ops.
            self.scaling = np.ones(self.n_params)
        self._lower_scaled = list(self._lower_internal)
        self._upper_scaled = list(self._upper_internal)

        # History – filled during optimize()
        self.history: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(
        self,
        algorithm: int = nlopt.LD_LBFGS,
        max_eval: int = 500,
        xtol_rel: float = 1e-4,
        ftol_rel: float = 0.0,
        xtol_abs: float = 0.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run the NLopt optimisation.

        Parameters
        ----------
        algorithm : int, optional
            NLopt algorithm constant (default ``nlopt.LD_LBFGS``).
        max_eval : int, optional
            Maximum function evaluations (default 500).
        xtol_rel : float, optional
            Relative tolerance on the parameters (default 1e-4).
        ftol_rel : float, optional
            Relative tolerance on the objective value (default 0.0,
            i.e. disabled).  A small positive value (e.g. ``1e-10``)
            gives NLopt an early-exit criterion when the loss has
            effectively plateaued, which avoids generic ``FAILURE``
            codes triggered by line-search stalls near the optimum.
        xtol_abs : float, optional
            Absolute tolerance on the internal parameters (default
            0.0, disabled).  Guards against unbounded ``xtol_rel``
            demands when a coordinate happens to be near zero.
        verbose : bool, optional
            Print iteration progress (default ``False``).

        Returns
        -------
        dict
            Keys:

            * ``final_loss`` — best objective value seen.
            * ``n_evaluations`` — number of objective evaluations.
            * ``history`` — dict of per-parameter and loss/iteration lists.
            * ``algorithm`` — human-readable algorithm name.
            * ``nlopt_result`` — raw integer return code from NLopt.
            * ``nlopt_result_name`` — short symbolic name (e.g.
              ``"XTOL_REACHED"``, ``"ROUNDOFF_LIMITED"``).
            * ``reason`` — one-line human-readable explanation.
            * ``converged`` — ``True`` if NLopt reported a genuine
              convergence criterion (including ``ROUNDOFF_LIMITED``),
              ``False`` if the run merely hit its evaluation/time
              budget or failed.
            * ``status`` — ``"success"`` or ``"failed"``.  A run is a
              success if the reported ``final_loss`` is trustworthy
              (all positive NLopt codes plus ``ROUNDOFF_LIMITED``);
              ``"failed"`` otherwise.
            * ``error`` — exception message, only present on failure.
        """
        import time

        # Reset history
        self.history = {name: [] for name in self.param_names}
        self.history["loss"] = []
        self.history["iteration"] = []

        is_gradient = algorithm in _GRADIENT_ALGORITHMS

        opt = nlopt.opt(algorithm, self.n_params)
        algo_name = opt.get_algorithm_name()

        # Pick objective
        if is_gradient:
            opt.set_min_objective(
                lambda x, grad: self._objective(x, grad, verbose=verbose)
            )
        else:
            opt.set_min_objective(
                lambda x, grad: self._objective_no_grad(x, verbose=verbose)
            )

        opt.set_lower_bounds(list(self._lower_internal))
        opt.set_upper_bounds(list(self._upper_internal))
        opt.set_maxeval(max_eval)
        opt.set_xtol_rel(xtol_rel)
        if ftol_rel > 0.0:
            opt.set_ftol_rel(ftol_rel)
        if xtol_abs > 0.0:
            opt.set_xtol_abs(xtol_abs)

        # Initial internal vector (already in the strategy's coord space)
        x0_scaled = list(self._initial_internal)

        if verbose:
            print(f"Optimizer: {algo_name}")
            print(f"Parameters ({self.n_params}): {self.param_names}")
            print(f"Max evaluations: {max_eval}")

        t0 = time.time()
        exc_message: Optional[str] = None
        try:
            opt.optimize(x0_scaled)
        except nlopt.RoundoffLimited as exc:
            # NLopt raises this when floating-point round-off prevents
            # further progress.  In practice this means we are at
            # machine precision — treat as a successful convergence.
            exc_message = str(exc) or "roundoff-limited"
        except Exception as exc:  # noqa: BLE001
            exc_message = str(exc) or type(exc).__name__

        elapsed = time.time() - t0
        code = opt.last_optimize_result()
        code_name, reason, converged = _interpret_nlopt_result(code)

        # A run is a "success" if we either finished cleanly (code > 0)
        # or hit one of the benign exception codes (currently only
        # ROUNDOFF_LIMITED).
        is_success = code > 0 or code in _NLOPT_BENIGN_EXCEPTION_CODES

        n = len(self.history["loss"])

        # Post-mortem: a generic FAILURE that occurs after a monotone,
        # plateaued descent is almost always a line-search stall at
        # the optimum (relative loss change < ~1e-10 over the last few
        # evaluations).  Report it as effectively-converged so the
        # notebook / caller can treat it like ROUNDOFF_LIMITED.
        plateau_converged = False
        if code == nlopt.FAILURE and n >= 5:
            tail = np.asarray(self.history["loss"][-5:], dtype=float)
            if np.all(np.isfinite(tail)):
                best = float(np.min(tail))
                spread = float(tail.max() - tail.min())
                denom = max(abs(best), 1e-300)
                if spread / denom < 1e-10:
                    is_success = True
                    converged = True
                    plateau_converged = True
                    code_name = "FAILURE_PLATEAU"
                    reason = (
                        "NLopt reported FAILURE, but the last five "
                        "loss values are flat to <1e-10 relative — "
                        "treating as converged (likely a line-search "
                        "stall at the optimum)."
                    )

        if is_success and not plateau_converged:
            final_loss = float(opt.last_optimum_value())
        elif is_success and plateau_converged:
            # last_optimum_value() may be stale after FAILURE; use the
            # best value we actually observed.
            observed = [v for v in self.history["loss"] if np.isfinite(v)]
            final_loss = float(min(observed)) if observed else float("inf")
        else:
            final_loss = self.history["loss"][-1] if n else float("inf")

        if verbose:
            tag = "Converged" if converged else (
                "Finished" if is_success else "Stopped"
            )
            print(
                f"  {tag} after {elapsed:.1f}s ({n} evals): "
                f"{code_name} — {reason}. final loss = {final_loss:.4e}"
            )

        result: Dict[str, Any] = {
            "final_loss": final_loss,
            "n_evaluations": n,
            "history": self.history.copy(),
            "algorithm": algo_name,
            "nlopt_result": int(code),
            "nlopt_result_name": code_name,
            "reason": reason,
            "converged": converged,
            "status": "success" if is_success else "failed",
        }
        if exc_message is not None:
            result["error"] = exc_message
        return result

    # ------------------------------------------------------------------
    # Internal objective functions
    # ------------------------------------------------------------------
    def _apply_params(self, x_internal: np.ndarray) -> np.ndarray:
        """Map internal coords to physical and push into the model."""
        x = self._param_strategy.physical_from_internal(
            np.asarray(x_internal, dtype=float)
        )

        for i, name in enumerate(self.param_names):
            if name == "scale_factor":
                self.experiment.scale_factor = float(x[i])
            elif name == "scale_vio":
                self._instrument.scale_vio = float(x[i])
            elif name == "scale_visnir":
                self._instrument.scale_visnir = float(x[i])
            else:
                self.model.set_parameter(name, float(x[i]))

        return x

    def _record(self, x: np.ndarray, loss_val: float) -> None:
        for i, name in enumerate(self.param_names):
            self.history[name].append(float(x[i]))
        self.history["loss"].append(float(loss_val))
        self.history["iteration"].append(len(self.history["loss"]))

    def _objective(
        self, x_scaled, grad, *, verbose: bool = False
    ) -> float:
        """Objective for gradient-based algorithms."""
        x = self._apply_params(np.asarray(x_scaled))

        current_loss = float(loss(self.experiment))

        if grad.size > 0:
            g_phys_dict = get_gradients(self.experiment)
            grad[:] = self._param_strategy.chain_rule(g_phys_dict, x)

        self._record(x, current_loss)

        if verbose:
            n = len(self.history["loss"])
            if n <= 3 or n % 10 == 1:
                self._print_iter(n, x, current_loss)

        return current_loss

    def _objective_no_grad(
        self, x_scaled, *, verbose: bool = False
    ) -> float:
        """Objective for gradient-free algorithms."""
        x = self._apply_params(np.asarray(x_scaled))
        current_loss = float(loss(self.experiment))
        self._record(x, current_loss)

        if verbose:
            n = len(self.history["loss"])
            if n <= 3 or n % 50 == 1:
                self._print_iter(n, x, current_loss)

        return current_loss

    def _print_iter(self, n: int, x: np.ndarray, loss_val: float) -> None:
        parts = [f"Iter {n:4d}: Loss={loss_val:.4e}"]
        for i, name in enumerate(self.param_names):
            if self._element_mask[i]:
                parts.append(f"{name}={x[i]:.4f}")
            elif "density" in name:
                parts.append(f"{name}={x[i]:.2e}")
            elif "temperature" in name:
                parts.append(f"{name}={x[i]:.0f}")
            else:
                parts.append(f"{name}={x[i]:.3e}")
        print("  ".join(parts))


# ======================================================================
# Multi-start convergence analysis
# ======================================================================


def _run_single(args: tuple) -> Dict[str, Any]:
    """Worker for parallel convergence analysis (top-level for pickle)."""
    (
        run_index,
        sample,
        param_names,
        element_mask,
        loss_type,
        instrument_cls_name,
        instrument_args,
        spectrum_wl,
        spectrum_ir,
        algorithm,
        max_eval,
        xtol_rel,
        opt_bounds,
        opt_scaling,
        normalize_elements,
        parametrization,
    ) = args

    from ._libskit import (
        Experiment,
        GenericSpectrometer,
        LTEModel,
        Spectrum,
    )

    # Reconstruct instrument
    instrument = GenericSpectrometer(*instrument_args)

    # Reconstruct spectrum
    spectrum = Spectrum(spectrum_wl, spectrum_ir)

    # Build a fresh model from the sampled parameters
    element_ratio = {}
    phys = {}
    for i, name in enumerate(param_names):
        if element_mask[i]:
            element_ratio[name] = float(sample[i])
        else:
            phys[name] = float(sample[i])

    # Normalise element ratios to sum to 1
    esum = sum(element_ratio.values())
    if esum > 1e-12:
        element_ratio = {k: v / esum for k, v in element_ratio.items()}

    model = LTEModel(
        element_ratio,
        phys.get("temperature", 10_000.0),
        phys.get("electron_density", 1e23),
        phys.get("length", 1e-4),
    )

    experiment = Experiment(loss_type, model, instrument, spectrum)

    optimizer = Optimizer(
        experiment,
        bounds=opt_bounds,
        scaling=opt_scaling,
        normalize_elements=normalize_elements,
        parametrization=parametrization,
    )
    result = optimizer.optimize(
        algorithm=algorithm,
        max_eval=max_eval,
        xtol_rel=xtol_rel,
        verbose=False,
    )

    # Attach per-run metadata
    result["run_index"] = run_index
    result["initial_params"] = {
        name: float(sample[i]) for i, name in enumerate(param_names)
    }
    # Read final params from the experiment's internal model (the clone
    # that the optimizer actually modified), NOT from the local `model`
    # variable which is the unmodified original.
    final_model = experiment.get_plasma_model()
    result["final_params"] = {
        name: float(final_model.get_parameter(name)) for name in param_names
    }

    return result


def convergence_analysis(
    experiment: Experiment,
    *,
    num_samples: int = 64,
    algorithm: int = nlopt.LD_LBFGS,
    max_eval: int = 500,
    xtol_rel: float = 1e-4,
    bounds: Optional[Dict[str, tuple[float, float]]] = None,
    scaling: Optional[Dict[str, float]] = None,
    normalize_elements: bool = True,
    num_processes: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
    parametrization: Parametrization = Parametrization.LOG_SOFTMAX,
) -> Dict[str, Any]:
    """Run multi-start optimization with Latin Hypercube Sampling.

    Generates *num_samples* starting points spread across the parameter
    space using Latin Hypercube Sampling, runs an :class:`Optimizer`
    from each, and collects statistics on convergence.

    Parameters
    ----------
    experiment : Experiment
        Template experiment (provides loss type, instrument, spectrum,
        and model type / element list).
    num_samples : int, optional
        Number of starting points (default 64).
    algorithm : int, optional
        NLopt algorithm constant (default ``nlopt.LD_LBFGS``).
    max_eval : int, optional
        Maximum evaluations per run (default 500).
    xtol_rel : float, optional
        Relative parameter tolerance (default 1e-4).
    bounds : dict, optional
        Per-parameter bound overrides (passed to each :class:`Optimizer`).
    scaling : dict, optional
        Per-parameter scaling overrides.
    normalize_elements : bool, optional
        Normalise element ratios (default ``True``).
    num_processes : int or None, optional
        Parallel workers.  ``None`` = ``min(cpu_count, num_samples)``.
        Use ``1`` for serial execution.
    seed : int, optional
        Random seed for LHS (default 42).
    verbose : bool, optional
        Print progress summary (default ``True``).

    Returns
    -------
    dict
        ``runs`` – list of per-run result dicts (same as
        :meth:`Optimizer.optimize` plus ``initial_params``,
        ``final_params``, ``run_index``).

        ``summary`` – aggregate statistics dict with keys:
        ``n_total``, ``n_success``, ``convergence_rate``,
        ``best_loss``, ``median_loss``, ``mean_loss``,
        ``final_params_mean``, ``final_params_std``.
    """
    import multiprocessing as mp
    import time

    from scipy.stats import qmc

    # --- extract template info from the experiment --------------------
    template_model = experiment.get_plasma_model()
    param_names = list(template_model.get_parameter_names())
    n_params = len(param_names)
    element_mask = [_is_element_param(n) for n in param_names]

    # Resolve bounds
    user_bounds = bounds or {}
    lower = np.empty(n_params)
    upper = np.empty(n_params)
    for i, name in enumerate(param_names):
        if name in user_bounds:
            lower[i], upper[i] = user_bounds[name]
        else:
            lower[i], upper[i] = _infer_bounds(name)

    # --- LHS ----------------------------------------------------------
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=num_samples)
    samples = qmc.scale(unit_samples, lower, upper)

    # --- serialisable experiment parts --------------------------------
    loss_type = experiment.loss_type
    spectrum = experiment.true_spectrum
    spectrum_wl = np.array(spectrum.wavelengths)
    spectrum_ir = np.array(spectrum.irradiance)

    # Instrument — currently only GenericSpectrometer is supported.
    instr = experiment.get_instrument()
    instrument_args = (instr.sigma, instr.gamma)
    instrument_cls_name = "GenericSpectrometer"

    # --- build argument tuples ----------------------------------------
    worker_args = []
    for idx in range(num_samples):
        worker_args.append((
            idx,
            samples[idx],
            param_names,
            element_mask,
            loss_type,
            instrument_cls_name,
            instrument_args,
            spectrum_wl,
            spectrum_ir,
            algorithm,
            max_eval,
            xtol_rel,
            user_bounds or None,
            scaling,
            normalize_elements,
            parametrization,
        ))

    # --- run ----------------------------------------------------------
    if num_processes is None:
        num_processes = min(mp.cpu_count(), num_samples)

    if verbose:
        algo_name = nlopt.opt(algorithm, 1).get_algorithm_name()
        print(
            f"Convergence analysis: {num_samples} runs × "
            f"{max_eval} max evals, {algo_name}, "
            f"{num_processes} processes"
        )

    t0 = time.time()
    if num_processes <= 1:
        runs = [_run_single(a) for a in worker_args]
    else:
        with mp.Pool(processes=num_processes) as pool:
            runs = pool.map(_run_single, worker_args)
    elapsed = time.time() - t0

    # --- aggregate summary --------------------------------------------
    successful = [r for r in runs if r["status"] == "success"]
    n_success = len(successful)
    losses = [r["final_loss"] for r in successful] if successful else []

    summary: Dict[str, Any] = {
        "n_total": num_samples,
        "n_success": n_success,
        "convergence_rate": n_success / num_samples if num_samples else 0.0,
        "best_loss": float(np.min(losses)) if losses else float("inf"),
        "median_loss": float(np.median(losses)) if losses else float("inf"),
        "mean_loss": float(np.mean(losses)) if losses else float("inf"),
        "elapsed_seconds": elapsed,
    }

    if successful:
        means = {}
        stds = {}
        for name in param_names:
            vals = [r["final_params"][name] for r in successful]
            means[name] = float(np.mean(vals))
            stds[name] = float(np.std(vals))
        summary["final_params_mean"] = means
        summary["final_params_std"] = stds

    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
        print(
            f"  Converged: {n_success}/{num_samples} "
            f"({summary['convergence_rate']:.0%})"
        )
        if losses:
            print(
                f"  Loss — best: {summary['best_loss']:.2e}, "
                f"median: {summary['median_loss']:.2e}, "
                f"mean: {summary['mean_loss']:.2e}"
            )

    return {"runs": runs, "summary": summary}
