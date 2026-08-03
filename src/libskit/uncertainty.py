# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT

"""
Deterministic propagation of Einstein-coefficient (``A_ki``) uncertainties
into the fitted plasma parameters via the implicit-function theorem.

This module is the Python user-facing wrapper around the C++ AD primitives
``libskit.get_jacobian`` (Jacobian J_theta of the simulated spectrum
w.r.t. all plasma parameters) and ``libskit.spectrum_vjp_aki`` (the
reverse-mode vector-Jacobian product J_A^T · v of the spectrum w.r.t.
the line-by-line A_ki values).

The math: a fitted parameter vector ``theta_hat`` is the implicit
minimiser of L(theta; A) = ||y_sim(theta; A) - y_ref||^2_W. Implicit
function theorem -> d theta_hat / d A = -H^{-1} U^T where

    H = J_theta^T W J_theta                  (Hessian of the loss at theta_hat)
    U_{l, alpha} = sum_i w_i J_theta_{i, alpha} J_A_{i, l}
               = (spectrum_vjp_aki(adjoint = w * J_theta[:, alpha]))_l

Catalogue uncertainty in A is `sigma_A_l = epsilon_l * A_l` where epsilon_l
is the NIST relative-accuracy code. The induced parameter covariance is

    Sigma_theta^(A) = H^{-1}  (U^T diag(sigma_A^2) U)  H^{-1}.

When parameters live under a closure constraint (e.g. mass fractions
sum to one in `LTEModel`), the raw H is singular by one rank.  We drop
the dependent column from J_theta before inverting and recover the
dependent row/column of Sigma analytically at the end.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import _libskit as _ext

__all__ = [
    "aki_covariance",
    "stark_covariance",
    "parameter_shift_vectors",
    "line_importance_ranking",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enumerate_aki_lines(aki_values: dict[str, list[float]],
                        aki_accuracies: dict[str, list[float]],
                        elements: list[str]) -> list[tuple[str, int, float, float]]:
    """Walk ions in the same order that ``build_const_data`` populates
    ``aki_active`` and return a flat list ``[(ion, line_idx_in_ion, A, sigma_A)]``.

    The iteration order matches the C++ side exactly:
    outer loop over ``elements`` in the order given (alphabetical, since
    ``LTEModel.element_ratio`` is a ``std::map``); inner loop over
    ionisation roman numerals "I" .. "X"; innermost loop over lines in
    that ion in catalogue order.
    """
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    out: list[tuple[str, int, float, float]] = []
    for el in elements:
        for il in range(len(roman)):
            ion = f"{el} {roman[il]}"
            if ion not in aki_values:
                continue
            A_list = aki_values[ion]
            acc_list = aki_accuracies.get(ion, [0.0] * len(A_list))
            assert len(A_list) == len(acc_list), \
                f"Aki values/accuracies length mismatch for {ion}"
            for li, (A, eps) in enumerate(zip(A_list, acc_list)):
                out.append((ion, li, float(A), float(eps)))
    return out


def _element_columns(param_names: list[str],
                     elements: list[str]) -> list[int]:
    """Indices into the C++ parameter vector that correspond to the
    element mass fractions, in the order C++ iterates them."""
    # Names come back as plain element symbols for LTE (no prefix); for
    # layered models they look like "layer_0_<element>" etc. We restrict
    # the closure constraint logic to flat LTE for now (PLAN.md Phase 3).
    cols: list[int] = []
    for el in elements:
        if el in param_names:
            cols.append(param_names.index(el))
        else:
            return []  # not the simple LTE layout, skip closure expansion
    return cols


def _drop_columns(M: np.ndarray, drop: list[int]) -> np.ndarray:
    keep = [i for i in range(M.shape[1]) if i not in drop]
    return M[:, keep]


def _build_jacobian_theta(experiment: Any) -> np.ndarray:
    """Forward sensitivity J_theta of the simulated spectrum w.r.t. all
    plasma parameters at the experiment's current state.

    Uses libskit's Enzyme-built Jacobian (one column per parameter,
    no finite differences) as required by the project plan.
    """
    J = _ext.get_jacobian(experiment)
    return np.asarray(J)


def _theta_hat_vector(experiment: Any, param_names: list[str]) -> np.ndarray:
    """Return the current parameter vector ``theta_hat`` in the same
    order as ``param_names`` (i.e. matching the columns of
    ``get_jacobian``).

    Only the parameters used by the simple LTE model are supported here
    -- other model types fall back to unit scaling.
    """
    model = experiment.get_plasma_model()
    out = np.zeros(len(param_names))
    er = dict(model.element_ratio) if hasattr(model, "element_ratio") else {}
    for i, name in enumerate(param_names):
        if name == "temperature":
            out[i] = float(model.temperature)
        elif name == "electron_density":
            out[i] = float(model.electron_density)
        elif name == "length":
            out[i] = float(model.length)
        elif name in er:
            out[i] = float(er[name])
        elif name == "scale_factor":
            out[i] = float(getattr(experiment, "scale_factor", 1.0)) or 1.0
        elif name in ("scale_vio", "scale_visnir"):
            instr = getattr(experiment, "instrument_ptr", None)
            out[i] = float(getattr(instr, name, 1.0)) if instr is not None else 1.0
        else:
            out[i] = 1.0
    return out


def _extend_param_names(experiment: Any,
                        param_names: list[str],
                        n_full: int) -> list[str]:
    """Append any trailing scale-type parameters that ``get_jacobian`` exposes
    but ``model.get_parameter_names()`` does not (``scale_factor`` for
    ``OptimizedScaleNorm``; ``scale_vio``/``scale_visnir`` for ChemCam).

    Order must mirror C++ ``build_param_names`` in ``enzyme_gradients.cpp``.
    """
    if len(param_names) >= n_full:
        return list(param_names)
    extras: list[str] = []
    try:
        norm_type = experiment.normalization_type
        if int(norm_type) == int(_ext.NormalizationType.OptimizedScaleNorm):
            extras.append("scale_factor")
    except AttributeError:
        pass
    # ChemCam appends two more scales after the (optional) scale_factor.
    remaining = n_full - len(param_names) - len(extras)
    if remaining >= 2:
        extras.extend(["scale_vio", "scale_visnir"])
    if len(param_names) + len(extras) != n_full:
        raise RuntimeError(
            f"Parameter-name mismatch: get_jacobian has {n_full} columns but "
            f"only {len(param_names) + len(extras)} names could be resolved "
            f"({param_names + extras}). Extend _extend_param_names()."
        )
    return list(param_names) + extras


# Columns of ``get_jacobian`` that only affect the loss normalisation (not the
# spectrum itself).  Their J_theta columns are all zeros, so including them in
# H = J^T W J makes the Hessian singular.  They add no parameter-space
# uncertainty either (∂ y_sim / ∂ scale = 0), so we drop them everywhere.
_LOSS_ONLY_SCALE_NAMES = ("scale_factor", "scale_vio", "scale_visnir")


def _strip_loss_scale_columns(J_theta: np.ndarray,
                              param_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop columns corresponding to normalisation-only scale parameters.

    These have ∂y_sim/∂scale ≡ 0 (the scale enters only through the loss),
    so their inclusion produces a singular Hessian and adds nothing to the
    parameter covariance.
    """
    keep = [i for i, n in enumerate(param_names) if n not in _LOSS_ONLY_SCALE_NAMES]
    if len(keep) == len(param_names):
        return J_theta, list(param_names)
    return J_theta[:, keep], [param_names[i] for i in keep]


def _apply_normalization_projection(experiment: Any,
                                    w: np.ndarray,
                                    J_red_s: np.ndarray) -> np.ndarray:
    """Fold the loss normalisation into the (weighted, scaled) Jacobian.

    OptimizedScaleNorm and AnalyticNorm both profile out a global
    amplitude, which is equivalent to a rank-1 W-orthogonal projection
    of J_theta away from the direction ``z = y_sim``.  After this
    correction, ``J_red_s_eff.T @ (w * J_red_s_eff)`` reproduces the
    Gauss-Newton Hessian of the *normalised* loss, and the same
    ``J_red_s_eff`` used as the VJP adjoint gives the correctly-projected
    U-matrix.  Result: the analytic covariance now matches the parameter
    covariance sampled by MC fits under the same normalisation.

    MaxNorm is currently passed through unchanged (the argmax-pixel
    projection has a different, non-symmetric structure).
    """
    norm_t = getattr(experiment, "normalization_type", None)
    if norm_t is None:
        return J_red_s
    try:
        NT = _ext.NormalizationType
        projected_types = {int(NT.OptimizedScaleNorm), int(NT.AnalyticNorm)}
    except AttributeError:
        return J_red_s
    if int(norm_t) not in projected_types:
        return J_red_s

    instr = experiment.get_instrument()
    model = experiment.get_plasma_model()
    wl = np.asarray(experiment.true_spectrum.wavelengths, dtype=float)
    y_sim = np.real(np.asarray(instr.calculate_spectrum(wl, model), dtype=float))
    if y_sim.shape[0] != J_red_s.shape[0]:
        raise RuntimeError(
            f"Normalisation projector: y_sim length {y_sim.shape[0]} "
            f"does not match Jacobian rows {J_red_s.shape[0]}."
        )
    wz = w * y_sim
    denom = float(wz @ y_sim)
    if denom <= 0.0:
        warnings.warn(
            "Normalisation projector: <z, W z> is non-positive; "
            "skipping projection.",
            RuntimeWarning,
            stacklevel=2,
        )
        return J_red_s
    proj_c = (wz @ J_red_s) / denom              # (p_red,)
    return J_red_s - np.outer(y_sim, proj_c)


def _weights_vector(experiment: Any,
                    n_out: int,
                    weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(n_out)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_out,):
        raise ValueError(
            f"weights must have shape ({n_out},), got {w.shape}"
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return w


def _closure_expand(Sigma_red: np.ndarray,
                    elem_cols_in_full: list[int],
                    dependent_full_col: int,
                    n_full: int,
                    kept_cols: list[int]) -> np.ndarray:
    """Re-insert the dependent-element row/column into a reduced
    covariance using ``x_dep = 1 - sum(x_others)``.

    ``Sigma_red``     - covariance over the parameters indexed by ``kept_cols``.
    ``elem_cols_in_full`` - indices of all element columns in the full param vector.
    ``dependent_full_col``- which element column was dropped.
    ``n_full``        - total number of parameters in the full vector.
    ``kept_cols``     - mapping from reduced-index to full-index.

    For a linear closure ``x_dep = 1 - sum_{e in others} x_e``,
       Var(x_dep) = a^T Sigma_red a,
       Cov(x_dep, theta_k) = -(Sigma_red a)_k        for k != dep
    with ``a_j = 1`` iff ``kept_cols[j]`` is a non-dependent element column.
    """
    n_red = Sigma_red.shape[0]
    a = np.zeros(n_red)
    other_elem_cols = [c for c in elem_cols_in_full if c != dependent_full_col]
    for j, full_col in enumerate(kept_cols):
        if full_col in other_elem_cols:
            a[j] = 1.0

    var_dep = float(a @ Sigma_red @ a)
    cov_dep_red = -Sigma_red @ a  # length n_red

    # Place reduced block into full-sized matrix
    Sigma_full = np.zeros((n_full, n_full))
    for i_red, i_full in enumerate(kept_cols):
        for j_red, j_full in enumerate(kept_cols):
            Sigma_full[i_full, j_full] = Sigma_red[i_red, j_red]
        # Cross-terms with dependent column
        Sigma_full[i_full, dependent_full_col] = cov_dep_red[i_red]
        Sigma_full[dependent_full_col, i_full] = cov_dep_red[i_red]
    Sigma_full[dependent_full_col, dependent_full_col] = var_dep
    # Force exact symmetry against FP drift
    Sigma_full = 0.5 * (Sigma_full + Sigma_full.T)
    return Sigma_full


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def aki_covariance(
    experiment: Any,
    weights: np.ndarray | None = None,
    *,
    include_noise: bool = True,
    sigma_n: np.ndarray | float | None = None,
    dependent_element: str | None = None,
) -> dict[str, Any]:
    """Return the implicit-function-theorem covariance of the fitted
    plasma parameters induced by NIST catalogue uncertainties in A_ki.

    Parameters
    ----------
    experiment : libskit.Experiment
        A fitted experiment.  Its current plasma_model values are
        treated as ``theta_hat``.
    weights : array-like, optional
        Per-wavelength weights ``w_i`` (the diagonal of W).
        Defaults to unit weights (unweighted L2).
    include_noise : bool, default True
        Whether to also compute Sigma_noise = H^{-1} J_theta^T W diag(sigma_n^2) W J_theta H^{-1}.
    sigma_n : float or array-like, optional
        Per-wavelength detector-noise standard deviations.  If a scalar
        is given, it is broadcast.  Required when ``include_noise=True``.
    dependent_element : str, optional
        Which element to treat as the closure-dependent one.  Defaults
        to the last element in the model's ``element_ratio`` map (i.e.
        the alphabetically-last symbol).  Pass an explicit name to
        force a particular choice.

    Returns
    -------
    dict with keys
        ``Sigma_aki``   - (p, p) parameter covariance from Aki catalogue uncertainty
        ``Sigma_noise`` - (p, p) from detector noise (zero if not requested)
        ``Sigma_total`` - sum of the two
        ``correlation`` - (p, p) Pearson correlation derived from Sigma_total
        ``parameter_names`` - list[str], length p
        ``line_importance`` - list of tuples, see :func:`line_importance_ranking`
        ``M``           - (p, L) matrix M = -H^{-1} U^T (for downstream Bayesian fusion)
    """
    # --- 1. Forward Jacobian J_theta via Enzyme ----------------------
    J_theta = _build_jacobian_theta(experiment)
    n_out, n_full = J_theta.shape
    w = _weights_vector(experiment, n_out, weights)

    # Parameter names and element layout
    model = experiment.get_plasma_model()
    param_names = _extend_param_names(experiment, list(model.get_parameter_names()), n_full)
    # Drop scale_factor / scale_vio / scale_visnir: their J columns are all
    # zeros and would make H singular.
    J_theta, param_names = _strip_loss_scale_columns(J_theta, param_names)
    n_full = J_theta.shape[1]
    elements = list(model.element_ratio.keys())
    elem_cols = _element_columns(param_names, elements)

    # --- 2. Decide which column (if any) to drop for the closure ----
    # We build a (n_full × p_red) basis matrix P_basis whose columns span the
    # simplex tangent T_Δ = {v : Σ_e v_e = 0}.  For every kept element column
    # P_basis sets +1 at the kept-element row AND -1 at the dependent-element
    # row, so that varying the reduced coordinate u_j is the closure-preserving
    # combined perturbation  δx_{e_j} = +1, δx_{e*} = -1.  T/ne/L rows are
    # passed through unchanged.  Then J_red = J_theta @ P_basis is the
    # Jacobian on the constrained manifold.
    #
    # This makes both Σ_full = P_basis Σ_red P_basis.T  and  M_full = P_basis
    # M_red  invariant under the choice of dependent_element (up to roundoff).
    do_closure_expand = bool(elem_cols)
    if do_closure_expand:
        if dependent_element is None:
            dependent_element = elements[-1]
        if dependent_element not in param_names:
            raise ValueError(
                f"dependent_element {dependent_element!r} not in parameter "
                f"names {param_names}"
            )
        dep_col = param_names.index(dependent_element)
        kept_cols = [i for i in range(n_full) if i != dep_col]
        other_elem_cols = [c for c in elem_cols if c != dep_col]
        P_basis = np.zeros((n_full, len(kept_cols)))
        for j_red, j_full in enumerate(kept_cols):
            P_basis[j_full, j_red] = 1.0
            if j_full in other_elem_cols:
                P_basis[dep_col, j_red] = -1.0
        J_red = J_theta @ P_basis
    else:
        dep_col = None
        kept_cols = list(range(n_full))
        P_basis = np.eye(n_full)
        J_red = J_theta

    # --- 3. Scale columns for conditioning ---------------------------
    # Compute scale = max(|theta_hat|, eps) per parameter; rescale J so
    # cond(H_scaled) becomes O(1) rather than O(1e25).
    theta_hat = _theta_hat_vector(experiment, param_names)
    scale_full = np.maximum(np.abs(theta_hat), 1e-30)
    scale_red = scale_full[kept_cols]
    J_red_s = J_red * scale_red[None, :]  # dy/du where u = theta/scale

    # Fold the loss normalisation (OptimizedScaleNorm / AnalyticNorm) into
    # J_red_s as a rank-1 W-orthogonal projection out of z = y_sim.  This
    # makes the analytic covariance consistent with what MC fits sample
    # under the same normalisation type.
    J_red_s = _apply_normalization_projection(experiment, w, J_red_s)

    # --- 4. Hessian of the loss in reduced, scaled parameter space ---
    JtW_s = J_red_s.T * w[None, :]
    H_s = JtW_s @ J_red_s
    cond_H = np.linalg.cond(H_s)
    if cond_H > 1e10:
        warnings.warn(
            f"Scaled Hessian H_s = J_s^T W J_s is poorly conditioned "
            f"(cond={cond_H:.2e}). Aki-covariance results may be "
            "numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )
    H_s_inv = np.linalg.inv(H_s)

    # --- 5. Per-parameter VJPs into Aki space ------------------------
    # For each scaled parameter alpha, adjoint = w * scale_alpha * J[:, alpha].
    # The factor scale_alpha is folded into the adjoint so U_s = J_A^T (scaled adjoint).
    n_aki = _ext.get_aki_gradient(experiment).size
    p_red = J_red.shape[1]
    U_s = np.zeros((n_aki, p_red))
    for alpha in range(p_red):
        adjoint = w * J_red_s[:, alpha]
        vjp = np.asarray(_ext.spectrum_vjp_aki(experiment, adjoint))
        if vjp.size != n_aki:
            raise RuntimeError(
                f"spectrum_vjp_aki returned size {vjp.size}, expected {n_aki}"
            )
        U_s[:, alpha] = vjp

    # --- 6. Catalogue covariance diag(sigma_A^2) --------------------
    aki_values = _ext.get_aki_values()
    aki_accuracies = _ext.get_aki_accuracies()
    line_table = _enumerate_aki_lines(aki_values, aki_accuracies, elements)
    if len(line_table) != n_aki:
        raise RuntimeError(
            f"Aki layout mismatch: enumerated {len(line_table)} lines, "
            f"C++ expects {n_aki}"
        )
    A_vec = np.array([row[2] for row in line_table])
    eps_vec = np.array([row[3] for row in line_table])
    sigma_A_vec = eps_vec * A_vec  # absolute sigma per line

    # M_inner = U_s^T diag(sigma_A^2) U_s (in scaled coords)
    UT_w = U_s.T * (sigma_A_vec ** 2)[None, :]  # (p_red, n_aki) - rows scaled
    M_inner_s = UT_w @ U_s  # (p_red, p_red)

    # --- 7. Sandwich in scaled coords, then unscale -----------------
    Sigma_aki_red_s = H_s_inv @ M_inner_s @ H_s_inv
    # Convert back to physical: Sigma_theta = diag(scale) Sigma_u diag(scale)
    Sigma_aki_red = Sigma_aki_red_s * np.outer(scale_red, scale_red)
    Sigma_aki_red = 0.5 * (Sigma_aki_red + Sigma_aki_red.T)

    # --- 8. Optional noise covariance -------------------------------
    Sigma_noise_red = np.zeros_like(Sigma_aki_red)
    if include_noise:
        if sigma_n is None:
            raise ValueError("include_noise=True requires sigma_n")
        sn = np.asarray(sigma_n, dtype=float)
        if sn.ndim == 0:
            sn = np.full(n_out, float(sn))
        if sn.shape != (n_out,):
            raise ValueError(
                f"sigma_n must be scalar or shape ({n_out},), got {sn.shape}"
            )
        # In scaled coords:
        #   Sigma_n_u = H_s^{-1} (J_s^T W diag(sn^2) W J_s) H_s^{-1}
        WJ_s = J_red_s * w[:, None]
        inner_n = WJ_s.T @ (WJ_s * (sn ** 2)[:, None])
        Sigma_noise_red_s = H_s_inv @ inner_n @ H_s_inv
        Sigma_noise_red = Sigma_noise_red_s * np.outer(scale_red, scale_red)
        Sigma_noise_red = 0.5 * (Sigma_noise_red + Sigma_noise_red.T)

    # --- 8. Closure expansion (Ca-style row/column) ----------------
    if do_closure_expand:
        Sigma_aki = _closure_expand(
            Sigma_aki_red, elem_cols, dep_col, n_full, kept_cols)
        Sigma_noise = _closure_expand(
            Sigma_noise_red, elem_cols, dep_col, n_full, kept_cols)
    else:
        Sigma_aki = np.zeros((n_full, n_full))
        Sigma_noise = np.zeros((n_full, n_full))
        for i_red, i_full in enumerate(kept_cols):
            for j_red, j_full in enumerate(kept_cols):
                Sigma_aki[i_full, j_full] = Sigma_aki_red[i_red, j_red]
                Sigma_noise[i_full, j_full] = Sigma_noise_red[i_red, j_red]

    Sigma_total = Sigma_aki + Sigma_noise

    # --- 9. Correlation matrix -------------------------------------
    diag_t = np.maximum(np.diag(Sigma_total), 0)
    std_t = np.sqrt(diag_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(
            (std_t[:, None] > 0) & (std_t[None, :] > 0),
            Sigma_total / (std_t[:, None] * std_t[None, :]),
            0.0,
        )

    # --- 10. Line importance ranking -------------------------------
    # M_s = -H_s^{-1} U_s^T  (scaled coords).  Physical shift in theta_alpha
    # from a unit perturbation in A_l is  M_physical[alpha,l] = scale_alpha * M_s[alpha,l].
    M_scaled = -H_s_inv @ U_s.T  # (p_red, n_aki)
    M_reduced = M_scaled * scale_red[:, None]
    contrib_per_alpha = (sigma_A_vec[None, :] * M_reduced) ** 2  # (p_red, n_aki)
    contrib_total = contrib_per_alpha.sum(axis=0)  # (n_aki,)
    importance = sorted(
        (
            (line_table[l][0], line_table[l][1], line_table[l][2],
             line_table[l][3], float(contrib_total[l]))
            for l in range(n_aki)
        ),
        key=lambda r: -r[4],
    )

    # Closure-expanded parameter-shift matrix M_full of shape (n_full, n_aki).
    # Uses the same closure basis P_basis assembled in step 2:
    #     M_full = P_basis @ M_reduced
    # The dropped element's row is reconstructed via x_dep = 1 - Σ x_others.
    M_full = P_basis @ M_reduced

    return {
        "Sigma_aki": Sigma_aki,
        "Sigma_noise": Sigma_noise,
        "Sigma_total": Sigma_total,
        "correlation": corr,
        "parameter_names": param_names,
        "line_importance": importance,
        "M": M_reduced,
        "M_full": M_full,
    }


def stark_covariance(
    experiment: Any,
    sigma_stark: float | dict[str, float] = 0.05,
    weights: np.ndarray | None = None,
    *,
    include_noise: bool = True,
    sigma_n: np.ndarray | float | None = None,
    dependent_element: str | None = None,
) -> dict[str, Any]:
    """IFT covariance of the fitted parameters induced by uncertainty in
    the per-line Stark FWHM multipliers ``m_l`` (baseline 1.0).

    Since perturbations act multiplicatively on width, a fractional
    Stark-width uncertainty ``sigma_stark`` (e.g. ``0.05`` for 5%,
    ``0.25`` for 25%) translates directly into ``sigma_m_l = sigma_stark``
    per line (or per ion if a dict is passed).

    Parameters
    ----------
    experiment : libskit.Experiment
        A fitted experiment.
    sigma_stark : float or dict[str, float], default 0.05
        Fractional Stark-width uncertainty.  Either a single scalar
        broadcast to every line, or a mapping ``{ion: sigma}`` giving
        an ion-specific value (each ion's lines share that sigma;
        missing ions fall back to the mean of provided values, or 0.05
        if the mapping is empty).
    weights, include_noise, sigma_n, dependent_element : see :func:`aki_covariance`.

    Returns
    -------
    dict with the same keys as :func:`aki_covariance` but with
    ``Sigma_stark`` in place of ``Sigma_aki``.
    """
    # --- 1. Forward Jacobian J_theta (shared with aki path) --------
    J_theta = _build_jacobian_theta(experiment)
    n_out, n_full = J_theta.shape
    w = _weights_vector(experiment, n_out, weights)

    model = experiment.get_plasma_model()
    param_names = _extend_param_names(experiment, list(model.get_parameter_names()), n_full)
    J_theta, param_names = _strip_loss_scale_columns(J_theta, param_names)
    n_full = J_theta.shape[1]
    elements = list(model.element_ratio.keys())
    elem_cols = _element_columns(param_names, elements)

    # --- 2. Closure basis (same as aki_covariance) ------------------
    do_closure_expand = bool(elem_cols)
    if do_closure_expand:
        if dependent_element is None:
            dependent_element = elements[-1]
        if dependent_element not in param_names:
            raise ValueError(
                f"dependent_element {dependent_element!r} not in parameter "
                f"names {param_names}"
            )
        dep_col = param_names.index(dependent_element)
        kept_cols = [i for i in range(n_full) if i != dep_col]
        other_elem_cols = [c for c in elem_cols if c != dep_col]
        P_basis = np.zeros((n_full, len(kept_cols)))
        for j_red, j_full in enumerate(kept_cols):
            P_basis[j_full, j_red] = 1.0
            if j_full in other_elem_cols:
                P_basis[dep_col, j_red] = -1.0
        J_red = J_theta @ P_basis
    else:
        dep_col = None
        kept_cols = list(range(n_full))
        P_basis = np.eye(n_full)
        J_red = J_theta

    # --- 3. Scaling for conditioning -------------------------------
    theta_hat = _theta_hat_vector(experiment, param_names)
    scale_full = np.maximum(np.abs(theta_hat), 1e-30)
    scale_red = scale_full[kept_cols]
    J_red_s = J_red * scale_red[None, :]

    # Fold the loss normalisation into J_red_s (see aki_covariance).
    J_red_s = _apply_normalization_projection(experiment, w, J_red_s)

    JtW_s = J_red_s.T * w[None, :]
    H_s = JtW_s @ J_red_s
    cond_H = np.linalg.cond(H_s)
    if cond_H > 1e10:
        warnings.warn(
            f"Scaled Hessian is poorly conditioned (cond={cond_H:.2e}). "
            "Stark-covariance results may be numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )
    H_s_inv = np.linalg.inv(H_s)

    # --- 4. Per-parameter VJPs into stark-multiplier space ---------
    n_aki = _ext.get_stark_gradient(experiment).size
    p_red = J_red.shape[1]
    U_s = np.zeros((n_aki, p_red))
    for alpha in range(p_red):
        adjoint = w * J_red_s[:, alpha]
        vjp = np.asarray(_ext.spectrum_vjp_stark(experiment, adjoint))
        if vjp.size != n_aki:
            raise RuntimeError(
                f"spectrum_vjp_stark returned size {vjp.size}, expected {n_aki}"
            )
        U_s[:, alpha] = vjp

    # --- 5. Build sigma_m per line, broadcasting the user input ----
    aki_values = _ext.get_aki_values()  # reuse for ion ordering + line counts
    aki_accuracies = _ext.get_aki_accuracies()
    line_table_aki = _enumerate_aki_lines(aki_values, aki_accuracies, elements)
    if len(line_table_aki) != n_aki:
        raise RuntimeError(
            f"Line layout mismatch: enumerated {len(line_table_aki)}, "
            f"C++ expects {n_aki}"
        )
    if isinstance(sigma_stark, dict):
        if not sigma_stark:
            raise ValueError("sigma_stark dict is empty")
        default_sig = float(np.mean(list(sigma_stark.values())))
        sigma_m_vec = np.array([
            float(sigma_stark.get(row[0], default_sig)) for row in line_table_aki
        ])
    else:
        sigma_m_vec = np.full(n_aki, float(sigma_stark))
    if np.any(sigma_m_vec < 0):
        raise ValueError("sigma_stark must be non-negative")

    # line_table for the return payload: swap the Aki-code column for
    # sigma_m so downstream code sees a Stark-appropriate table.
    line_table = [
        (row[0], row[1], row[2], float(sigma_m_vec[l]))
        for l, row in enumerate(line_table_aki)
    ]

    # --- 6. Sandwich ------------------------------------------------
    UT_w = U_s.T * (sigma_m_vec ** 2)[None, :]
    M_inner_s = UT_w @ U_s
    Sigma_stark_red_s = H_s_inv @ M_inner_s @ H_s_inv
    Sigma_stark_red = Sigma_stark_red_s * np.outer(scale_red, scale_red)
    Sigma_stark_red = 0.5 * (Sigma_stark_red + Sigma_stark_red.T)

    # --- 7. Optional noise covariance ------------------------------
    Sigma_noise_red = np.zeros_like(Sigma_stark_red)
    if include_noise:
        if sigma_n is None:
            raise ValueError("include_noise=True requires sigma_n")
        sn = np.asarray(sigma_n, dtype=float)
        if sn.ndim == 0:
            sn = np.full(n_out, float(sn))
        if sn.shape != (n_out,):
            raise ValueError(
                f"sigma_n must be scalar or shape ({n_out},), got {sn.shape}"
            )
        WJ_s = J_red_s * w[:, None]
        inner_n = WJ_s.T @ (WJ_s * (sn ** 2)[:, None])
        Sigma_noise_red_s = H_s_inv @ inner_n @ H_s_inv
        Sigma_noise_red = Sigma_noise_red_s * np.outer(scale_red, scale_red)
        Sigma_noise_red = 0.5 * (Sigma_noise_red + Sigma_noise_red.T)

    # --- 8. Closure expansion --------------------------------------
    if do_closure_expand:
        Sigma_stark = _closure_expand(
            Sigma_stark_red, elem_cols, dep_col, n_full, kept_cols)
        Sigma_noise = _closure_expand(
            Sigma_noise_red, elem_cols, dep_col, n_full, kept_cols)
    else:
        Sigma_stark = np.zeros((n_full, n_full))
        Sigma_noise = np.zeros((n_full, n_full))
        for i_red, i_full in enumerate(kept_cols):
            for j_red, j_full in enumerate(kept_cols):
                Sigma_stark[i_full, j_full] = Sigma_stark_red[i_red, j_red]
                Sigma_noise[i_full, j_full] = Sigma_noise_red[i_red, j_red]

    Sigma_total = Sigma_stark + Sigma_noise

    # --- 9. Correlation --------------------------------------------
    diag_t = np.maximum(np.diag(Sigma_total), 0)
    std_t = np.sqrt(diag_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(
            (std_t[:, None] > 0) & (std_t[None, :] > 0),
            Sigma_total / (std_t[:, None] * std_t[None, :]),
            0.0,
        )

    # --- 10. Line importance ranking -------------------------------
    M_scaled = -H_s_inv @ U_s.T
    M_reduced = M_scaled * scale_red[:, None]
    contrib_per_alpha = (sigma_m_vec[None, :] * M_reduced) ** 2
    contrib_total = contrib_per_alpha.sum(axis=0)
    importance = sorted(
        (
            (line_table[l][0], line_table[l][1], line_table[l][2],
             line_table[l][3], float(contrib_total[l]))
            for l in range(n_aki)
        ),
        key=lambda r: -r[4],
    )

    M_full = P_basis @ M_reduced

    return {
        "Sigma_stark": Sigma_stark,
        "Sigma_noise": Sigma_noise,
        "Sigma_total": Sigma_total,
        "correlation": corr,
        "parameter_names": param_names,
        "line_importance": importance,
        "M": M_reduced,
        "M_full": M_full,
    }


def parameter_shift_vectors(experiment: Any,
                            weights: np.ndarray | None = None,
                            *,
                            dependent_element: str | None = None) -> dict[str, Any]:
    """Return the per-line parameter shift matrix ``M = -H^{-1} U^T``.

    Element ``M[alpha, l]`` is the first-order shift in parameter
    ``alpha`` induced by a unit perturbation in ``A_l``.  Useful for
    downstream Bayesian fusion of external Aki measurements.

    Returns dict with keys ``M`` (p_red x L), ``parameter_names``,
    ``line_table`` (list of ``(ion, line_idx, A, sigma_A)``),
    ``H_inv`` (p_red x p_red).
    """
    res = aki_covariance(
        experiment,
        weights=weights,
        include_noise=False,
        sigma_n=0.0,  # ignored since include_noise=False
        dependent_element=dependent_element,
    )
    # _closure_expand is the only place that re-expands; M itself stays
    # in reduced coords. Use the raw outputs we already computed.
    return {
        "M": res["M"],
        "parameter_names": res["parameter_names"],
        "line_importance": res["line_importance"],
    }


def line_importance_ranking(experiment: Any,
                            weights: np.ndarray | None = None,
                            *,
                            top_n: int | None = None,
                            dependent_element: str | None = None
                            ) -> list[tuple[str, int, float, float, float]]:
    """Return per-line contribution to ``tr(Sigma_theta^(A))``.

    Each row is ``(ion, line_idx, A, sigma_A, contribution)`` and rows
    are sorted by descending contribution.  ``top_n`` truncates the
    output (default: return all entries).
    """
    res = aki_covariance(
        experiment,
        weights=weights,
        include_noise=False,
        sigma_n=0.0,
        dependent_element=dependent_element,
    )
    ranking = res["line_importance"]
    if top_n is not None:
        ranking = ranking[:top_n]
    return ranking
