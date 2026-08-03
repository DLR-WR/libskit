#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Gradient Descent Optimizer for LIBS Spectrum Fitting

Provides an adaptive gradient descent optimizer that uses libskit's
automatic differentiation backend to fit plasma model parameters
(temperature, electron density, length, element ratios) to an
experimental spectrum.
"""
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

from ._libskit import Experiment, get_gradients, loss, LTEModel

# Parameter scaling factors for gradient descent updates.
# These reflect the characteristic magnitudes of each fitted parameter so that
# raw gradients (in physical units) translate into well-conditioned step sizes
# at typical learning rates (~1e-4). They are NOT arbitrary tuning knobs.
_TEMPERATURE_SCALE_SQ = (10000.0) ** 2      # K^2     (T ~ 10 kK)
_ELECTRON_DENSITY_SCALE_SQ = (1e24) ** 2    # m^-6    (n_e ~ 1e24 m^-3)
_LENGTH_SCALE_SQ = (1e-3) ** 2              # m^2     (L ~ 1 mm)

# Floors to keep parameters in physically meaningful ranges
_MIN_LENGTH = 1e-7        # m
_MIN_ELEMENT_RATIO = 1e-8


def _compute_quality_metrics(predicted: np.ndarray,
                             observed: np.ndarray,
                             weights: np.ndarray) -> Dict[str, float]:
    """RMSE, weighted chi-squared, and R^2 for a predicted spectrum."""
    residuals = predicted - observed
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    chi_squared = float(np.sum(weights * residuals ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "chi_squared": chi_squared, "r_squared": r_squared}


def _init_history(model: LTEModel, elements: List[str]) -> Dict[str, List[Any]]:
    """Create an empty per-iteration history dictionary."""
    history: Dict[str, List[Any]] = {
        "loss": [],
        "temperature": [],
        "electron_density": [],
        "length": [],
        "learning_rate": [],
        "rmse": [],
        "chi_squared": [],
        "r_squared": [],
    }
    for element in elements:
        history[element] = []
    return history


def _record_state(history: Dict[str, List[Any]], model: LTEModel,
                  elements: List[str], current_lr: float,
                  loss_value: float, metrics: Dict[str, float]) -> None:
    """Append one row of optimisation state to the history dict."""
    history["loss"].append(loss_value)
    history["temperature"].append(model.temperature)
    history["electron_density"].append(model.electron_density)
    history["length"].append(model.length)
    history["learning_rate"].append(current_lr)
    history["rmse"].append(metrics["rmse"])
    history["chi_squared"].append(metrics["chi_squared"])
    history["r_squared"].append(metrics["r_squared"])
    for element in elements:
        history[element].append(model.element_ratio[element])


def _snapshot_parameters(model: LTEModel, elements: List[str]) -> Dict[str, float]:
    """Capture current parameter values for later change-rate calculations."""
    snap = {
        "temperature": model.temperature,
        "electron_density": model.electron_density,
        "length": model.length,
    }
    for element in elements:
        snap[element] = model.element_ratio[element]
    return snap


def _apply_gradient_step(model: LTEModel, gradients: Dict[str, float],
                         current_lr: float, elements: List[str]) -> None:
    """Apply one scaled-gradient update to the plasma model in-place.

    Updates temperature, electron density, length, and element ratios; clips
    each to its physical floor; and renormalises element ratios to sum to 1.
    """
    if "temperature" in gradients:
        model.temperature -= current_lr * gradients["temperature"] * _TEMPERATURE_SCALE_SQ
        model.temperature = max(0.0, model.temperature)

    if "electron_density" in gradients:
        model.electron_density -= current_lr * gradients["electron_density"] * _ELECTRON_DENSITY_SCALE_SQ
        model.electron_density = max(0.0, model.electron_density)

    if "length" in gradients:
        model.length -= current_lr * gradients["length"] * _LENGTH_SCALE_SQ
        model.length = max(_MIN_LENGTH, model.length)

    for element in elements:
        new_value = max(model.element_ratio[element] - current_lr * gradients[element],
                        _MIN_ELEMENT_RATIO)
        model.set_element_ratio(element, new_value)

    # Renormalise element ratios to sum to 1
    element_sum = sum(model.element_ratio[e] for e in elements)
    if element_sum > 0:
        for element in elements:
            model.set_element_ratio(element, model.element_ratio[element] / element_sum)


def _relative_changes(model: LTEModel, old_params: Dict[str, float],
                      elements: List[str]) -> Tuple[Dict[str, float], float]:
    """Return per-parameter relative changes and their maximum."""
    changes: Dict[str, float] = {
        "temperature": abs(model.temperature - old_params["temperature"]) / old_params["temperature"],
        "electron_density": abs(model.electron_density - old_params["electron_density"]) / old_params["electron_density"],
        "length": abs(model.length - old_params["length"]) / old_params["length"],
    }
    for element in elements:
        changes[element] = abs(model.element_ratio[element] - old_params[element]) / old_params[element]
    max_change = max(changes.values())
    return changes, max_change


def custom_gradient_descent(
    experiment: Experiment,
    learning_rate: float,
    max_iterations: int = 1000,
    parameter_change_threshold: float = 0.1,
    parameter_change_patience: int = 5,
    verbose: bool = False,
    lr_decay_factor: float = 0.5,
    lr_decay_patience: int = 10
) -> Dict[str, Any]:
    """
    Gradient descent optimizer with adaptive learning rate and convergence detection.

    Performs gradient descent optimization to fit plasma model parameters to
    an experimental spectrum. Uses automatic differentiation to compute gradients
    and includes adaptive learning rate decay with configurable convergence criteria.

    Parameters
    ----------
    experiment : Experiment
        LIBSkit Experiment object containing the plasma model, instrument,
        target spectrum, and loss function type.
    learning_rate : float
        Initial learning rate for gradient descent. Typical values range
        from 1e-5 to 1e-3 depending on the problem.
    max_iterations : int, optional
        Maximum number of iterations (default: 1000). Acts as a safety limit.
    parameter_change_threshold : float, optional
        Convergence threshold for relative parameter changes (default: 0.1).
        Convergence is detected when all parameter changes are below this
        threshold for ``parameter_change_patience`` consecutive iterations.
    parameter_change_patience : int, optional
        Number of consecutive iterations the convergence criterion must be
        met before declaring convergence (default: 5).
    verbose : bool, optional
        If True, print progress information during optimization (default: False).
    lr_decay_factor : float, optional
        Factor by which to multiply learning rate when no improvement is
        detected (default: 0.5).
    lr_decay_patience : int, optional
        Number of iterations without loss improvement before applying
        learning rate decay (default: 10).

    Returns
    -------
    dict
        Optimization results containing:

        - **model** : LTEModel
            Final optimized plasma model.
        - **history** : dict
            Optimization history with keys:

            - 'loss': List of loss values per iteration
            - 'temperature': Temperature values per iteration
            - 'electron_density': Electron density values per iteration
            - 'length': Length values per iteration
            - 'learning_rate': Learning rate per iteration
            - 'rmse': Root mean square error per iteration
            - 'chi_squared': Chi-squared statistic per iteration
            - 'r_squared': R-squared coefficient per iteration
            - Element symbols: Element ratio values per iteration

        - **converged** : bool
            Whether the optimization converged.
        - **iterations** : int
            Total number of iterations performed.
        - **final_loss** : float
            Final loss value.
        - **convergence_criterion** : str
            Description of the convergence criterion used.

    Notes
    -----
    The optimizer applies parameter scaling internally:

    - Temperature gradients are scaled by (10000)²
    - Electron density gradients are scaled by (1e24)²
    - Length gradients are scaled by (1e-3)²

    Element ratios are normalized to sum to 1.0 after each iteration.

    Example
    -------
    >>> results = custom_gradient_descent(
    ...     experiment,
    ...     learning_rate=1e-4,
    ...     max_iterations=500,
    ...     verbose=True
    ... )
    >>> if results['converged']:
    ...     print(f"Converged in {results['iterations']} iterations")
    ...     print(f"Final temperature: {results['model'].temperature:.0f} K")
    """

    if verbose:
        print("=" * 80)
        print(f"Running gradient descent for up to {max_iterations} iterations...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Learning rate decay factor: {lr_decay_factor}")
        print(f"  Learning rate decay patience: {lr_decay_patience}")
        print(f"  Parameter change threshold: {parameter_change_threshold}")
        print(f"  Parameter change patience: {parameter_change_patience}\n")

    current_model = experiment.get_plasma_model()
    elements = list(current_model.element_ratio.keys())
    history = _init_history(current_model, elements)

    # Learning rate decay tracking
    current_lr = learning_rate
    best_loss = float("inf")
    no_improvement_count = 0

    # Cache wavelengths/instrument/observed once
    wavelengths = experiment.true_spectrum.wavelengths
    instrument = experiment.get_instrument()
    experimental_data = experiment.true_spectrum.irradiance
    weights = 1.0 / (experimental_data + 1e-10)  # Poisson-style weights

    # Initial state
    initial_loss = loss(experiment)
    initial_spectrum = instrument.calculate_spectrum(wavelengths, current_model)
    initial_metrics = _compute_quality_metrics(initial_spectrum, experimental_data, weights)
    _record_state(history, current_model, elements, current_lr, initial_loss, initial_metrics)

    if verbose:
        print(f"Initial state:")
        print(f"  Loss: {initial_loss:.2e}")
        print(f"  Convergence criterion: relative parameter changes < {parameter_change_threshold} for {parameter_change_patience} consecutive iterations")
        print(f"  Starting optimization...\n")

    converged = False
    convergence_counter = 0

    for iteration in range(max_iterations):
        gradients = get_gradients(experiment)
        old_params = _snapshot_parameters(current_model, elements)

        _apply_gradient_step(current_model, gradients, current_lr, elements)

        new_loss = loss(experiment)

        # Learning-rate decay
        if new_loss < best_loss:
            best_loss = new_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= lr_decay_patience:
            old_lr = current_lr
            current_lr *= lr_decay_factor
            no_improvement_count = 0
            if verbose:
                print(f"\n  ⚡ Learning rate decayed: {old_lr:.2e} → {current_lr:.2e} (no improvement for {lr_decay_patience} iterations)")

        changes, max_param_change = _relative_changes(current_model, old_params, elements)

        new_spectrum = instrument.calculate_spectrum(wavelengths, current_model)
        metrics = _compute_quality_metrics(new_spectrum, experimental_data, weights)
        _record_state(history, current_model, elements, current_lr, new_loss, metrics)

        # Convergence: max relative change < threshold for `patience` iterations
        if max_param_change < parameter_change_threshold:
            convergence_counter += 1
            if convergence_counter >= parameter_change_patience:
                converged = True
                if verbose:
                    print(f"\n✓ CONVERGED after {iteration + 1} iterations!")
                    print(f"  All parameter changes < {parameter_change_threshold:.2e} for {parameter_change_patience} consecutive iterations")
                    print(f"  Final max parameter change: {max_param_change:.3f}")
                break
        else:
            convergence_counter = 0

        if verbose:
            element_strs = "".join(
                f"{e}={current_model.element_ratio[e]:.5f} ({changes[e]:.2e}) |  "
                for e in elements
            )
            print(
                f"Iter {iteration + 1:3d}: Loss={new_loss:.3e} || "
                f"Temp={current_model.temperature:.0f}K ({changes['temperature']:.2e}) | "
                f"Ne={current_model.electron_density:.2e} ({changes['electron_density']:.2e}) | "
                f"L={current_model.length:.3e} ({changes['length']:.2e}) || "
                f"{element_strs}"
            )

    if not converged and verbose:
        print(f"\n⚠ Did not converge after {max_iterations} iterations")

    final_model = experiment.get_plasma_model()
    return {
        "model": final_model,
        "history": history,
        "converged": converged,
        "iterations": len(history["loss"]) - 1,
        "final_loss": history["loss"][-1],
        "convergence_criterion": f"max parameter change < {parameter_change_threshold}%",
    }
