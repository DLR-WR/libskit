#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Coordinate-transform strategies for :class:`libskit.optimizer.Optimizer`.

The optimizer evaluates ``loss(experiment)`` and ``get_gradients``
in **physical** parameter coordinates (kelvin, m^-3, m, mass fractions in
``[0, 1]``).  A :class:`ParametrizationStrategy` defines an invertible
map from those physical coordinates to the *internal* coordinates that
NLopt sees.

Two strategies ship with libskit, selected via the
:class:`Parametrization` enum:

* :attr:`Parametrization.LOG_SOFTMAX` (default since libskit 2.1):
  ``log T``, ``log n_e``, ``log L`` (and other strictly-positive
  multiplicative physics parameters), with element ratios produced by a
  softmax over unconstrained logits.  Improves L-BFGS conditioning
  dramatically on plasma fits (see notebook
  ``gradients/convergence_gradient_check.ipynb`` in
  ``libskit_experiments``).
* :attr:`Parametrization.LINEAR_RAW`: linear physics with ``1/x0``
  scaling, raw element ratios renormalised to sum to one before each
  loss evaluation.  Kept as a less-preferred fallback for users who
  need to reproduce earlier results.

User-facing bounds (the ``bounds=`` kwarg of :class:`Optimizer`) are
always supplied in physical units.  The strategy converts them to its
internal coordinate space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.special import softmax


# ---------------------------------------------------------------------------
# Public selection enum
# ---------------------------------------------------------------------------

class Parametrization(Enum):
    """Choice of coordinate transform for :class:`libskit.Optimizer`."""

    LOG_SOFTMAX = "log_softmax"
    LOG_SOFTMAX_ALR = "log_softmax_alr"
    LINEAR_RAW = "linear_raw"


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

# Strictly-positive, multiplicatively-acting physics parameters that benefit
# from a log reparametrisation.  Matched as substrings of the parameter name
# so layered models (e.g. ``layer_temperature_0``) inherit the same treatment.
_LOG_PHYSICS_SUBSTRINGS: tuple[str, ...] = (
    "temperature",
    "electron_density",
    "length",
    "tau",
    "integration_time",
)

# Pass-through scalar multipliers — kept linear by both strategies because
# they are not physical plasma quantities.
_LINEAR_PASSTHROUGH: frozenset[str] = frozenset(
    {"scale_factor", "scale_vio", "scale_visnir"}
)


def is_log_physics(name: str) -> bool:
    """Return True if *name* is a positive multiplicative physics parameter."""
    if name in _LINEAR_PASSTHROUGH:
        return False
    return any(s in name for s in _LOG_PHYSICS_SUBSTRINGS)


def is_passthrough(name: str) -> bool:
    """Return True if *name* is a linear scalar multiplier (scale_factor, …)."""
    return name in _LINEAR_PASSTHROUGH


# ---------------------------------------------------------------------------
# Softmax helpers (Jacobian application & inverse — not in scipy)
# ---------------------------------------------------------------------------

def softmax_jac_apply(c: np.ndarray, g_c: np.ndarray) -> np.ndarray:
    """Apply the softmax Jacobian to a downstream gradient.

    If ``c = softmax(u)`` and ``f = h(c)``, then
    ``df/du_i = c_i (g_i - <c, g>)`` where ``g = dh/dc``.
    """
    c = np.asarray(c, dtype=float)
    g_c = np.asarray(g_c, dtype=float)
    return c * (g_c - float(np.dot(c, g_c)))


def inverse_softmax(c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Recover centred logits ``u`` from a probability vector ``c``."""
    c = np.clip(np.asarray(c, dtype=float), eps, None)
    c = c / c.sum()
    u = np.log(c)
    return u - u.mean()


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class ParametrizationStrategy(ABC):
    """Abstract coordinate transform between physical and internal coords.

    A strategy is constructed in two phases: first ``__init__`` records
    user-facing options, then :meth:`bind` is called by the
    :class:`Optimizer` once physical parameter names, initial values, and
    physical bounds are known.
    """

    name: str = "abstract"

    def __init__(self) -> None:
        self._bound = False

    # ---- binding -----------------------------------------------------
    def bind(
        self,
        *,
        param_names: Sequence[str],
        initial_physical: np.ndarray,
        lower_physical: np.ndarray,
        upper_physical: np.ndarray,
        element_mask: np.ndarray,
    ) -> None:
        self.param_names: list[str] = list(param_names)
        self.initial_physical: np.ndarray = np.asarray(
            initial_physical, dtype=float
        ).copy()
        self.lower_physical: np.ndarray = np.asarray(
            lower_physical, dtype=float
        ).copy()
        self.upper_physical: np.ndarray = np.asarray(
            upper_physical, dtype=float
        ).copy()
        self.element_mask: np.ndarray = np.asarray(element_mask, dtype=bool).copy()
        self.n_params: int = len(self.param_names)
        self._bound = True
        self._post_bind()

    def _post_bind(self) -> None:
        """Hook for subclasses to precompute internal state once bound."""

    def _check_bound(self) -> None:
        if not self._bound:
            raise RuntimeError(
                f"{type(self).__name__} must be bound to an Optimizer "
                "before use (call bind(...))."
            )

    # ---- properties subclasses must provide -------------------------
    @property
    @abstractmethod
    def initial_internal(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def lower_internal(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def upper_internal(self) -> np.ndarray: ...

    # ---- transforms --------------------------------------------------
    @abstractmethod
    def physical_from_internal(self, x_int: np.ndarray) -> np.ndarray:
        """Return *normalised* physical parameters from internal coords."""

    @abstractmethod
    def chain_rule(
        self, g_phys_dict: Dict[str, float], x_phys: np.ndarray
    ) -> np.ndarray:
        """Map a physical gradient dict onto an internal-coord gradient."""

    # ---- diagnostics -------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        bound = "bound" if self._bound else "unbound"
        n = getattr(self, "n_params", "?")
        return f"<{type(self).__name__} {bound} n={n}>"


# ---------------------------------------------------------------------------
# LinearRawParametrization (legacy linear scaling)
# ---------------------------------------------------------------------------

class LinearRawParametrization(ParametrizationStrategy):
    """Linear physics with auto-scaling, raw element ratios.

    * Physics parameters and pass-through scalars are scaled by
      ``1 / |x0|`` so that the initial scaled value is ``~1``.  Element
      ratios stay unscaled in ``[0, 1]``.
    * Element ratios are optionally renormalised to sum to one before
      each loss evaluation (controlled by ``normalize_elements``).
    """

    name = "linear_raw"

    def __init__(
        self,
        *,
        scaling: Optional[Dict[str, float]] = None,
        normalize_elements: bool = True,
    ) -> None:
        super().__init__()
        self._user_scaling: Dict[str, float] = dict(scaling or {})
        self.normalize_elements = bool(normalize_elements)

    def _post_bind(self) -> None:
        self.scaling = np.empty(self.n_params)
        for i, name in enumerate(self.param_names):
            if name in self._user_scaling:
                self.scaling[i] = float(self._user_scaling[name])
            elif self.element_mask[i]:
                self.scaling[i] = 1.0
            else:
                val = abs(float(self.initial_physical[i]))
                self.scaling[i] = 1.0 / val if val > 1e-30 else 1.0

    @property
    def initial_internal(self) -> np.ndarray:
        self._check_bound()
        return self.initial_physical * self.scaling

    @property
    def lower_internal(self) -> np.ndarray:
        self._check_bound()
        return self.lower_physical * self.scaling

    @property
    def upper_internal(self) -> np.ndarray:
        self._check_bound()
        return self.upper_physical * self.scaling

    def physical_from_internal(self, x_int: np.ndarray) -> np.ndarray:
        self._check_bound()
        x = np.asarray(x_int, dtype=float) / self.scaling
        if self.normalize_elements and self.element_mask.any():
            esum = x[self.element_mask].sum()
            if esum > 1e-12:
                x[self.element_mask] = x[self.element_mask] / esum
            else:
                n_elem = int(self.element_mask.sum())
                x[self.element_mask] = 1.0 / n_elem
        return x

    def chain_rule(
        self, g_phys_dict: Dict[str, float], x_phys: np.ndarray
    ) -> np.ndarray:
        self._check_bound()
        g = np.empty(self.n_params)
        for i, name in enumerate(self.param_names):
            g[i] = float(g_phys_dict.get(name, 0.0)) / self.scaling[i]
        return g


# ---------------------------------------------------------------------------
# LogSoftmaxParametrization (default)
# ---------------------------------------------------------------------------

class LogSoftmaxParametrization(ParametrizationStrategy):
    """Log-physical + softmax-element parametrisation.

    * Strictly-positive multiplicative physics parameters
      (``temperature``, ``electron_density``, ``length``, ``tau``,
      ``integration_time`` and substring matches thereof) are
      reparametrised as ``y = log x``.  Bounds are mapped accordingly.
    * Element ratios are produced by a softmax over unconstrained
      logits ``u``.  Logit bounds default to ``[-20, 20]``.  The softmax
      output automatically satisfies the simplex constraint
      ``sum(c) = 1`` and ``c >= 0``, so no explicit renormalisation is
      needed.
    * Linear pass-through scalars (``scale_factor``, ``scale_vio``,
      ``scale_visnir``) keep the legacy ``1/|x0|`` scaling.

    The chain rule from physical to internal coordinates is

    * ``dL/d(log p) = p · dL/dp`` for log-physics,
    * ``dL/du = c * (g_c - <c, g_c>)`` for softmax (Jacobi block on the
      element subspace, with ``g_c[i] = dL/dc_i``),
    * ``dL/dx_int = dL/dx_phys / scaling`` for linear pass-throughs.
    """

    name = "log_softmax"

    def __init__(
        self,
        *,
        logit_lower: float = -20.0,
        logit_upper: float = 20.0,
        element_floor: float = 1e-8,
    ) -> None:
        super().__init__()
        if logit_upper <= logit_lower:
            raise ValueError("logit_upper must be > logit_lower")
        self.logit_lower = float(logit_lower)
        self.logit_upper = float(logit_upper)
        self.element_floor = float(element_floor)

    def _post_bind(self) -> None:
        # Per-parameter "kind": 'log', 'element', 'linear'
        self._kind: list[str] = []
        for i, name in enumerate(self.param_names):
            if self.element_mask[i]:
                self._kind.append("element")
            elif is_log_physics(name):
                self._kind.append("log")
            else:
                # scale_factor / scale_vio / scale_visnir / unknown — keep linear
                self._kind.append("linear")

        # Linear-passthrough scaling: 1 / |x0| like legacy
        self._linear_scaling = np.ones(self.n_params)
        for i, kind in enumerate(self._kind):
            if kind == "linear":
                val = abs(float(self.initial_physical[i]))
                self._linear_scaling[i] = 1.0 / val if val > 1e-30 else 1.0

        # Build initial internal vector and bounds
        initial = np.empty(self.n_params)
        lo = np.empty(self.n_params)
        hi = np.empty(self.n_params)

        for i, kind in enumerate(self._kind):
            if kind == "log":
                if self.initial_physical[i] <= 0 or self.lower_physical[i] <= 0:
                    raise ValueError(
                        f"LogSoftmaxParametrization requires strictly positive "
                        f"values and bounds for log-physics parameter "
                        f"{self.param_names[i]!r}; got x0="
                        f"{self.initial_physical[i]}, lower="
                        f"{self.lower_physical[i]}."
                    )
                initial[i] = np.log(float(self.initial_physical[i]))
                lo[i] = np.log(float(self.lower_physical[i]))
                hi[i] = np.log(float(self.upper_physical[i]))
            elif kind == "linear":
                s = self._linear_scaling[i]
                initial[i] = float(self.initial_physical[i]) * s
                lo[i] = float(self.lower_physical[i]) * s
                hi[i] = float(self.upper_physical[i]) * s
            else:  # element
                lo[i] = self.logit_lower
                hi[i] = self.logit_upper
                # initial[i] set below from the joint logit normalisation

        # Element block: invert softmax of the (renormalised) initial
        # element vector to obtain centred logits.  This ensures that
        # softmax(initial_internal[elements]) == c0 / c0.sum().
        em = self.element_mask
        if em.any():
            c0 = self.initial_physical[em].astype(float).copy()
            s = c0.sum()
            if s > 1e-12:
                c0 = c0 / s
            else:
                c0[:] = 1.0 / len(c0)
            u0 = inverse_softmax(c0, eps=self.element_floor)
            # Clip to logit bounds; recentre to keep softmax output ~unchanged
            u0 = np.clip(u0, self.logit_lower, self.logit_upper)
            u0 = u0 - u0.mean()
            initial[em] = u0

        self._initial_internal = initial
        self._lower_internal = lo
        self._upper_internal = hi

    # ---- properties --------------------------------------------------
    @property
    def initial_internal(self) -> np.ndarray:
        self._check_bound()
        return self._initial_internal.copy()

    @property
    def lower_internal(self) -> np.ndarray:
        self._check_bound()
        return self._lower_internal.copy()

    @property
    def upper_internal(self) -> np.ndarray:
        self._check_bound()
        return self._upper_internal.copy()

    # ---- transforms --------------------------------------------------
    def physical_from_internal(self, x_int: np.ndarray) -> np.ndarray:
        self._check_bound()
        x_int = np.asarray(x_int, dtype=float)
        x_phys = np.empty(self.n_params)
        em = self.element_mask
        for i, kind in enumerate(self._kind):
            if kind == "log":
                x_phys[i] = float(np.exp(x_int[i]))
            elif kind == "linear":
                x_phys[i] = float(x_int[i]) / self._linear_scaling[i]
            # elements handled below
        if em.any():
            x_phys[em] = softmax(x_int[em])
        return x_phys

    def chain_rule(
        self, g_phys_dict: Dict[str, float], x_phys: np.ndarray
    ) -> np.ndarray:
        self._check_bound()
        g = np.empty(self.n_params)
        em = self.element_mask
        for i, kind in enumerate(self._kind):
            if kind == "log":
                # dL/d(log p) = p * dL/dp
                g[i] = float(g_phys_dict.get(self.param_names[i], 0.0)) * float(
                    x_phys[i]
                )
            elif kind == "linear":
                g[i] = float(g_phys_dict.get(self.param_names[i], 0.0)) / float(
                    self._linear_scaling[i]
                )
            # elements: filled by softmax-Jacobian block below
        if em.any():
            elem_names = [self.param_names[i] for i in np.where(em)[0]]
            g_c = np.array(
                [float(g_phys_dict.get(n, 0.0)) for n in elem_names]
            )
            c = np.asarray(x_phys[em], dtype=float)
            g[em] = softmax_jac_apply(c, g_c)
        return g


# ---------------------------------------------------------------------------
# LogSoftmaxALRParametrization (softmax with one logit fixed to zero)
# ---------------------------------------------------------------------------

class LogSoftmaxALRParametrization(LogSoftmaxParametrization):
    """Log-physics + softmax-element with the ALR gauge fixed.

    Identical to :class:`LogSoftmaxParametrization` except that one
    element logit (by default the last element in ``param_names``) is
    held at zero.  This removes the additive gauge freedom
    ``softmax(u) == softmax(u + c*1)`` that makes the plain softmax
    parametrisation non-injective, giving the element block ``K - 1``
    independent compositional coordinates (the additive log-ratio, ALR,
    convention).

    Concretely, if ``k*`` is the reference index, we impose
    ``u[k*] == 0`` by:

    * shifting the initial element logits by ``-u[k*]`` (softmax
      invariant, so the initial concentrations are unchanged),
    * setting the internal lower and upper bounds at ``k*`` to
      ``(0, 0)`` so NLopt keeps the variable pinned, and
    * zeroing the internal gradient at ``k*`` so round-off cannot push
      the iterates along the gauge direction.

    The forward map, the softmax Jacobian, and hence
    :meth:`physical_from_internal` and the element block of
    :meth:`chain_rule` are inherited unchanged — the mapping from the
    remaining ``K-1`` free logits to concentrations is exactly the ALR
    transform.
    """

    name = "log_softmax_alr"

    def __init__(
        self,
        *,
        logit_lower: float = -20.0,
        logit_upper: float = 20.0,
        element_floor: float = 1e-8,
        reference_element: Optional[str] = None,
    ) -> None:
        super().__init__(
            logit_lower=logit_lower,
            logit_upper=logit_upper,
            element_floor=element_floor,
        )
        self._reference_element_name = reference_element
        self._alr_ref_idx: Optional[int] = None

    def _post_bind(self) -> None:
        super()._post_bind()

        em = self.element_mask
        if not em.any():
            self._alr_ref_idx = None
            return

        elem_indices = np.where(em)[0]
        if self._reference_element_name is not None:
            if self._reference_element_name not in self.param_names:
                raise ValueError(
                    f"reference_element {self._reference_element_name!r} "
                    "not found in parameter names."
                )
            ref_idx = self.param_names.index(self._reference_element_name)
            if not em[ref_idx]:
                raise ValueError(
                    f"reference_element {self._reference_element_name!r} "
                    "is not an element ratio parameter."
                )
        else:
            # Default: last element in param_names order.
            ref_idx = int(elem_indices[-1])
        self._alr_ref_idx = ref_idx

        # Softmax is invariant under a global additive shift; shift so
        # that u[ref] = 0 without changing the initial concentrations.
        u = self._initial_internal[em].copy()
        # Find the position of ref inside the element block.
        ref_pos = int(np.where(elem_indices == ref_idx)[0][0])
        u -= u[ref_pos]
        # Clip the free logits to the logit window (ref stays at 0).
        u = np.clip(u, self.logit_lower, self.logit_upper)
        u[ref_pos] = 0.0
        self._initial_internal[em] = u

        # Pin the reference logit: equal bounds ⇒ NLopt treats it as
        # fixed at the initial value (which is now 0).
        self._lower_internal[ref_idx] = 0.0
        self._upper_internal[ref_idx] = 0.0

    def chain_rule(
        self, g_phys_dict: Dict[str, float], x_phys: np.ndarray
    ) -> np.ndarray:
        g = super().chain_rule(g_phys_dict, x_phys)
        if self._alr_ref_idx is not None:
            # Zero the gradient along the pinned logit so numerical
            # noise cannot drive the pinned coordinate off its bound.
            g[self._alr_ref_idx] = 0.0
        return g
