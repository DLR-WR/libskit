#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Boltzmann and Saha-Boltzmann Plot Utilities for CF-LIBS Analysis

This module contains calculation functions for:
1. Single-element and multi-element Boltzmann fits
2. Saha-Boltzmann fits with fixed electron density
3. CF-LIBS concentration calculations (Gornushkin 2022)
4. Stark broadening electron density estimation
5. Physical constants and partition functions

Functions are separated from plotting for clean notebook organization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import constants as const
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import linregress
from typing import Dict, Optional, List, Tuple
from molmass import Formula
from ._libskit import calculate_partition_function, calculate_saha_function

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
kB_eV = const.k / const.eV  # Boltzmann constant in eV/K
kB_J = const.k              # Boltzmann constant in J/K
h = const.h                 # Planck constant (J·s)
me = const.m_e              # Electron mass (kg)
c_light = const.c           # Speed of light (m/s)
e_charge = const.e          # Elementary charge (C)
epsilon_0 = const.epsilon_0 # Vacuum permittivity (F/m)
amu = const.u               # Atomic mass unit (kg)


def get_atomic_mass(element: str) -> float:
    """
    Get atomic mass in amu using the molmass package.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Ca', 'Fe')

    Returns
    -------
    float
        Atomic mass in amu
    """
    return Formula(element).mass


def int_to_roman(num: int) -> str:
    """
    Convert an integer to Roman numeral string.

    Parameters
    ----------
    num : int
        Integer to convert (1-3999)

    Returns
    -------
    str
        Roman numeral representation
    """
    if num < 1 or num > 3999:
        raise ValueError(f"Roman numeral conversion only supports 1-3999, got {num}")

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']

    result = ''
    for i, v in enumerate(val):
        while num >= v:
            result += syms[i]
            num -= v
    return result


# =============================================================================
# PARTITION FUNCTIONS AND SAHA EQUATION
# =============================================================================

def partition_function(element: str, sp_num: int, T: float) -> float:
    """
    Calculate partition function U(T) using libskit's C++ implementation.

    Uses the full energy level summation from NIST levels database:
    U(T) = Σ g_i exp(-E_i/kT)

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Ca', 'Fe')
    sp_num : int
        Ionization state: 1 = neutral, 2 = singly ionized, 3 = doubly ionized, etc.
    T : float
        Temperature in K

    Returns
    -------
    float
        Partition function value

    Examples
    --------
    >>> partition_function('Fe', 1, 10000)  # Neutral iron at 10 000 K
    45.243...
    >>> partition_function('Ca', 2, 8000)   # Singly ionized calcium
    2.649...
    """
    # Construct ion string (e.g., 'Ca I', 'Fe II', 'Ar IV')
    ion_str = f"{element} {int_to_roman(sp_num)}"
    return calculate_partition_function(ion_str, T)


# Alias for backward compatibility
partition_function_approx = partition_function


def saha_factor(T: float, ne: float) -> float:
    """
    Calculate Saha factor: 2(2π me kB T)^(3/2) / (ne h³).

    This is the ratio that appears in the Saha equation when converting
    between neutral and ionic species populations.

    Parameters
    ----------
    T : float
        Temperature in K
    ne : float
        Electron density in m^-3

    Returns
    -------
    float
        Saha factor (dimensionless when ne in m^-3)

    Examples
    --------
    >>> saha_factor(10000, 1e23)
    1.564...
    """
    return 2 * (2 * np.pi * me * kB_J * T)**(3/2) / (ne * h**3)


def saha_function(element: str, T: float, ne: float,
                  ionization_energies: Dict[str, float]) -> float:
    """
    Calculate Saha function S(T)/ne for use in concentration formulas.

    S(T) = (2 U+ / U0) * (2πmekT/h²)^(3/2) * exp(-E_ion/kT)

    Parameters
    ----------
    element : str
        Element symbol
    T : float
        Temperature in K
    ne : float
        Electron density in m^-3
    ionization_energies : dict
        {element: E_ion (eV)}

    Returns
    -------
    float
        S(T)/ne ratio used in Eqs. 3.18 and 3.19 of Gornushkin 2022

    Examples
    --------
    >>> ionization_energies = {'Ca': 6.11, 'Fe': 7.90}
    >>> saha_function('Ca', 10000, 1e23, ionization_energies)
    0.001...
    """
    U0 = partition_function_approx(element, 1, T)
    Up = partition_function_approx(element, 2, T)
    E_ion = ionization_energies.get(element, 6.0)  # eV

    # Saha function (in m^-3 units)
    saha = 2 * (Up / U0) * (2 * np.pi * me * kB_J * T / h**2)**(3/2) * \
           np.exp(-E_ion / (kB_eV * T))

    return saha / ne


# =============================================================================
# BOLTZMANN PLOT FITTING
# =============================================================================

def single_element_boltzmann_fit(lines_df: pd.DataFrame,
                                  sigma_threshold: float = 2.5,
                                  max_clip_iter: int = 3) -> Optional[Dict]:
    """
    Perform a Boltzmann plot fit for a single element/ionization state.

    Fits the Boltzmann equation:
        ln(I*λ / (g_k * A_ki)) = -E_k / (k_B * T) + C

    Uses iterative sigma-clipping to reject outliers.

    Parameters
    ----------
    lines_df : pd.DataFrame
        DataFrame with columns: 'ritz_wl_vac(nm)', 'Aki(s^-1)', 'Ek(eV)',
        'g_k', and 'intensity'. Should be pre-filtered to desired ionization state.
    sigma_threshold : float
        Outlier rejection threshold in standard deviations.  Points with
        residuals larger than ``sigma_threshold × σ_resid`` are removed in
        each iteration. Set to 0 or ``np.inf`` to disable clipping.
        Default 2.5.
    max_clip_iter : int
        Maximum number of sigma-clipping iterations (default 3).  Clipping
        stops early if no points are removed.

    Returns
    -------
    dict or None
        Dictionary containing fit results, or None if fit fails.
        Keys include 'T', 'T_err', 'r2', 'n_points', 'x', 'y',
        'slope', 'intercept', 'n_original', 'n_clipped', 'mask',
        'residuals', 'wavelengths'.

    Examples
    --------
    >>> import pandas as pd
    >>> lines = pd.DataFrame({
    ...     'ritz_wl_vac(nm)': [358.12, 373.49, 404.58],
    ...     'Aki(s^-1)': [1.02e8, 9.0e7, 8.6e7],
    ...     'Ek(eV)': [4.47, 4.31, 4.55],
    ...     'g_k': [9, 7, 11],
    ...     'intensity': [1200, 980, 1100]
    ... })
    >>> result = single_element_boltzmann_fit(lines)
    >>> if result:
    ...     print(f"T = {result['T']:.0f} K, R² = {result['r2']:.3f}")
    """
    df = lines_df.copy()

    if len(df) < 3:
        return None

    wl = df['ritz_wl_vac(nm)'].values
    Aki = df['Aki(s^-1)'].values
    Ek = df['Ek(eV)'].values
    gk = df['g_k'].values
    I = df['intensity'].values

    # Compute Boltzmann coordinates
    y = np.log(I * wl / (gk * Aki))
    x = Ek

    # Filter out NaN/inf from log computation
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, wl = x[valid], y[valid], wl[valid]

    if len(x) < 3:
        return None

    n_original = len(x)
    mask = np.ones(len(x), dtype=bool)

    # Iterative sigma-clipping
    do_clip = sigma_threshold > 0 and np.isfinite(sigma_threshold)
    for _ in range(max_clip_iter if do_clip else 0):
        if np.sum(mask) < 3:
            break
        res = linregress(x[mask], y[mask])
        residuals = y[mask] - (res.slope * x[mask] + res.intercept)
        sigma = np.std(residuals)
        if sigma < 1e-15:
            break  # perfect fit, no clipping needed
        # Compute full residuals for all (unclipped) points to update mask
        full_resid = y - (res.slope * x + res.intercept)
        new_mask = np.abs(full_resid) <= sigma_threshold * sigma
        if np.array_equal(new_mask, mask):
            break  # converged
        mask = new_mask

    # Final fit on kept points
    if np.sum(mask) < 3:
        return None

    result = linregress(x[mask], y[mask])

    # Slope must be negative for valid temperature
    if result.slope >= 0:
        return None

    T = -1 / (result.slope * kB_eV)
    T_err = abs(T) * (result.stderr / abs(result.slope)) if result.stderr else 0

    # Residual diagnostics
    residuals = y - (result.slope * x + result.intercept)

    return {
        'T': T,
        'T_err': T_err,
        'r2': result.rvalue**2,
        'n_points': int(np.sum(mask)),
        'x': x[mask],
        'y': y[mask],
        'slope': result.slope,
        'intercept': result.intercept,
        'n_original': n_original,
        'n_clipped': n_original - int(np.sum(mask)),
        'mask': mask,
        'residuals': residuals,
        'wavelengths': wl,
    }


def multi_element_boltzmann_fit(element_data: Dict[str, pd.DataFrame],
                                 T_min: float = 3000.0,
                                 T_max: float = 50000.0,
                                 sigma_clip: float = 0.0,
                                 max_clip_iter: int = 3) -> Optional[Dict]:
    """
    Constrained Boltzmann fit: shared slope (temperature), individual intercepts.

    This fits all elements simultaneously with the constraint that they share
    the same temperature (slope), but each element has its own intercept.
    Optionally performs iterative sigma-clipping to reject outliers.

    Parameters
    ----------
    element_data : dict
        {element: DataFrame with isolated lines including 'intensity' column}
        Each DataFrame must have columns: 'ritz_wl_vac(nm)', 'Aki(s^-1)',
        'Ek(eV)', 'g_k', 'intensity'. Should be pre-filtered to desired ionization state.
    T_min : float
        Minimum allowed temperature (K)
    T_max : float
        Maximum allowed temperature (K)
    sigma_clip : float
        If > 0, reject points with residuals exceeding
        ``sigma_clip × σ_resid`` in each iteration. Default 0 (disabled).
    max_clip_iter : int
        Maximum number of sigma-clipping iterations (default 3).

    Returns
    -------
    dict or None
        Dictionary containing fit results, or None if fit fails.
        Includes residual diagnostics: 'n_original', 'n_clipped',
        'residuals', 'mask', 'wavelengths', 'corr_wl_resid'.

    Examples
    --------
    >>> element_data = {'Fe': fe_lines_df, 'Ca': ca_lines_df}
    >>> result = multi_element_boltzmann_fit(element_data, sigma_clip=2.5)
    >>> if result:
    ...     print(f"Shared T = {result['T']:.0f} K")
    """
    all_x, all_y, all_wl, all_el = [], [], [], []
    el_list = []

    for el, df in element_data.items():
        if len(df) < 3 or 'intensity' not in df.columns:
            continue

        wl = df['ritz_wl_vac(nm)'].values
        Aki = df['Aki(s^-1)'].values
        Ek = df['Ek(eV)'].values
        gk = df['g_k'].values
        I = df['intensity'].values

        y = np.log(I * wl / (gk * Aki))
        x = Ek

        # Filter out NaN/inf from log computation
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 3:
            continue

        all_x.append(x[valid])
        all_y.append(y[valid])
        all_wl.append(wl[valid])
        all_el.append(np.full(np.sum(valid), len(el_list)))
        el_list.append(el)

    if len(el_list) < 1:
        return None

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    wl_all = np.concatenate(all_wl)
    el_idx = np.concatenate(all_el).astype(int)
    n_elements = len(el_list)
    n_original = len(x_all)
    mask = np.ones(n_original, dtype=bool)

    def _fit(x, y, idx, init_slope=None):
        if init_slope is None:
            init_slope = -1 / (kB_eV * 10000)
        init_intercepts = [np.mean(y) - init_slope * np.mean(x)] * n_elements
        x0 = [init_slope] + init_intercepts
        slope_bounds = (-1 / (kB_eV * T_min), -1 / (kB_eV * T_max))

        def objective(params):
            slope = params[0]
            intercepts = params[1:]
            y_pred = slope * x + intercepts[idx]
            return np.sum((y - y_pred)**2)

        res = minimize(objective, x0, method='L-BFGS-B',
                       bounds=[slope_bounds] + [(-50, 50)] * n_elements)
        return res.x

    # Initial fit
    params = _fit(x_all, y_all, el_idx)

    # Iterative sigma-clipping
    do_clip = sigma_clip > 0 and np.isfinite(sigma_clip)
    for _ in range(max_clip_iter if do_clip else 0):
        slope_cur = params[0]
        intercepts_cur = params[1:]
        y_pred = slope_cur * x_all + intercepts_cur[el_idx]
        resid = y_all - y_pred
        sigma = np.std(resid[mask])
        if sigma < 1e-15:
            break
        new_mask = np.abs(resid) <= sigma_clip * sigma
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        if np.sum(mask) < 3:
            break
        params = _fit(x_all[mask], y_all[mask], el_idx[mask], init_slope=slope_cur)

    slope_fit = params[0]
    intercepts_fit = params[1:]
    T_fit = -1 / (slope_fit * kB_eV)

    # Calculate R² and residuals
    y_pred = slope_fit * x_all + intercepts_fit[el_idx]
    residuals = y_all - y_pred
    y_kept = y_all[mask]
    y_pred_kept = y_pred[mask]
    ss_res = np.sum((y_kept - y_pred_kept)**2)
    ss_tot = np.sum((y_kept - np.mean(y_kept))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Wavelength–residual correlation
    if np.sum(mask) > 3 and np.std(wl_all[mask]) > 1e-10:
        from scipy.stats import pearsonr
        corr_wl_resid, _ = pearsonr(wl_all[mask], residuals[mask])
    else:
        corr_wl_resid = np.nan

    return {
        'T': T_fit,
        'slope': slope_fit,
        'r2': r2,
        'elements': el_list,
        'intercepts': {el: intercepts_fit[i] for i, el in enumerate(el_list)},
        'x_data': {el: all_x[i] for i, el in enumerate(el_list)},
        'y_data': {el: all_y[i] for i, el in enumerate(el_list)},
        'n_points': int(np.sum(mask)),
        # Residual diagnostics
        'n_original': n_original,
        'n_clipped': n_original - int(np.sum(mask)),
        'residuals': residuals,
        'mask': mask,
        'wavelengths': wl_all,
        'corr_wl_resid': corr_wl_resid,
    }


# =============================================================================
# SAHA-BOLTZMANN PLOT FITTING
# =============================================================================

def saha_boltzmann_fit_single_element(lines_I: pd.DataFrame,
                                       lines_II: pd.DataFrame,
                                       E_ion: float,
                                       ne_fixed: float,
                                       sigma_clip: float = 0.0,
                                       max_clip_iter: int = 3) -> Optional[Dict]:
    """
    Saha-Boltzmann fit for a single element with FIXED electron density.

    Combines neutral (I) and ionic (II) lines onto a unified energy scale
    and optimizes only temperature.  Optionally performs iterative
    sigma-clipping to reject outlier lines.

    Parameters
    ----------
    lines_I : DataFrame
        Neutral lines with 'intensity' column
    lines_II : DataFrame
        Ionic lines with 'intensity' column
    E_ion : float
        Ionization energy in eV
    ne_fixed : float
        Fixed electron density in m^-3 (from Stark broadening)
    sigma_clip : float
        If > 0, reject points with residuals exceeding
        ``sigma_clip × σ_resid`` in each iteration. Default 0 (disabled).
    max_clip_iter : int
        Maximum number of sigma-clipping iterations (default 3).

    Returns
    -------
    dict or None
        Fit results including T, slope, intercept, and data for plotting.
        When sigma clipping is enabled, also includes 'mask', 'n_original',
        'n_clipped'.
    """
    def get_boltzmann_coords(df):
        wl = df['ritz_wl_vac(nm)'].values
        Aki = df['Aki(s^-1)'].values
        Ek = df['Ek(eV)'].values
        gk = df['g_k'].values
        I = df['intensity'].values
        y = np.log(I * wl / (gk * Aki))
        valid = np.isfinite(y) & (I > 0)
        return Ek[valid], y[valid]

    x_I, y_I = get_boltzmann_coords(lines_I)
    x_II_base, y_II = get_boltzmann_coords(lines_II)

    if len(x_I) < 2 or len(x_II_base) < 2:
        return None

    # Ionic x-coordinate includes ionization energy
    x_II = x_II_base + E_ion

    n_I, n_II = len(x_I), len(x_II)
    n_total = n_I + n_II
    mask = np.ones(n_total, dtype=bool)

    def _fit_T(x_all, y_all_base, ionic_mask):
        """Fit temperature on the given subset."""
        def objective(T):
            if T <= 0:
                return 1e20
            saha_corr = np.log(saha_factor(T, ne_fixed))
            y_corr = y_all_base.copy()
            y_corr[ionic_mask] -= saha_corr
            slope = -1 / (kB_eV * T)
            intercept = np.mean(y_corr - slope * x_all)
            y_pred = slope * x_all + intercept
            return np.sum((y_corr - y_pred)**2)
        return minimize_scalar(objective, bounds=(3000, 50000), method='bounded')

    # All data arrays (full, unclipped)
    x_full = np.concatenate([x_I, x_II])
    y_full = np.concatenate([y_I, y_II])
    ionic_full = np.concatenate([np.zeros(n_I, dtype=bool),
                                  np.ones(n_II, dtype=bool)])

    # Initial fit on all points
    result = _fit_T(x_full, y_full, ionic_full)
    T_fit = result.x

    # Iterative sigma-clipping
    do_clip = sigma_clip > 0 and np.isfinite(sigma_clip)
    for _ in range(max_clip_iter if do_clip else 0):
        saha_corr = np.log(saha_factor(T_fit, ne_fixed))
        y_corr = y_full.copy()
        y_corr[ionic_full] -= saha_corr
        slope_cur = -1 / (kB_eV * T_fit)
        intercept_cur = np.mean(y_corr[mask] - slope_cur * x_full[mask])
        resid = y_corr - (slope_cur * x_full + intercept_cur)
        sigma = np.std(resid[mask])
        if sigma < 1e-15:
            break
        new_mask = np.abs(resid) <= sigma_clip * sigma
        # Ensure at least 2 Fe I and 2 Fe II remain
        if np.sum(new_mask[:n_I]) < 2 or np.sum(new_mask[n_I:]) < 2:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        # Re-fit on kept points
        result = _fit_T(x_full[mask], y_full[mask], ionic_full[mask])
        T_fit = result.x

    # Final parameters at optimal T
    saha_corr = np.log(saha_factor(T_fit, ne_fixed))
    y_corr = y_full.copy()
    y_corr[ionic_full] -= saha_corr
    slope = -1 / (kB_eV * T_fit)
    intercept = np.mean(y_corr[mask] - slope * x_full[mask])

    return {
        'T': T_fit, 'ne': ne_fixed, 'E_ion': E_ion,
        'x_I': x_I, 'y_I': y_I, 'x_II': x_II, 'y_II': y_II,
        'slope': slope, 'intercept': intercept,
        'mask': mask, 'n_original': n_total,
        'n_clipped': n_total - int(np.sum(mask)),
    }


def multi_element_saha_boltzmann_fit(element_data: Dict[str, pd.DataFrame],
                                      ionization_energies: Dict[str, float],
                                      ne_fixed: float,
                                      sigma_clip: float = 0.0,
                                      max_clip_iter: int = 3) -> Optional[Dict]:
    """
    Constrained Saha-Boltzmann fit: shared temperature, one intercept per element.

    Uses FIXED ne from Stark broadening. Each element gets a single intercept
    (neutral and ionic lines of same element share the intercept after Saha correction).

    Optionally performs iterative sigma-clipping to reject outlier lines
    that degrade the fit (e.g. self-absorbed or blended lines).

    Parameters
    ----------
    element_data : dict
        {element: DataFrame with both neutral and ionic lines}
    ionization_energies : dict
        {element: E_ion (eV)}
    ne_fixed : float
        Fixed electron density in m^-3
    sigma_clip : float
        If > 0, reject points with residuals exceeding
        ``sigma_clip × σ_resid`` in each iteration. Default 0 (disabled).
    max_clip_iter : int
        Maximum number of sigma-clipping iterations (default 3). Only used
        when *sigma_clip* > 0.

    Returns
    -------
    dict or None
        Fit results including T, R², intercepts per element, residual
        diagnostics, and data for plotting.  New keys compared to the
        non-clipping version:

        - ``n_original``: total points before clipping
        - ``n_clipped``: number of points removed
        - ``residuals``: residual array (all original points)
        - ``mask``: boolean mask of kept points
        - ``wavelengths``: wavelength array matching residuals
        - ``corr_wl_resid``: Pearson correlation between wavelength and
          residual (flags systematic calibration errors)
    """
    all_data = []
    el_list = []

    for el, df in element_data.items():
        if el not in ionization_energies:
            continue
        if len(df) == 0 or 'sp_num' not in df.columns:
            continue
        E_ion = ionization_energies[el]

        # Require both neutral and ionic lines
        n_I = (df['sp_num'] == 1).sum()
        n_II = (df['sp_num'] == 2).sum()
        if n_I < 2 or n_II < 2:
            continue

        if el not in el_list:
            el_list.append(el)
        el_idx = el_list.index(el)

        for sp_num in [1, 2]:
            lines_sp = df[df['sp_num'] == sp_num]
            if len(lines_sp) < 2:
                continue

            wl = lines_sp['ritz_wl_vac(nm)'].values
            Aki = lines_sp['Aki(s^-1)'].values
            Ek = lines_sp['Ek(eV)'].values
            gk = lines_sp['g_k'].values
            I = lines_sp['intensity'].values

            y = np.log(I * wl / (gk * Aki))
            valid = np.isfinite(y) & (I > 0)

            if np.sum(valid) < 2:
                continue

            # For ionic: x = Ek + E_ion (unified energy scale)
            x = Ek[valid] + E_ion if sp_num == 2 else Ek[valid]

            all_data.append({
                'x': x, 'y': y[valid], 'wl': wl[valid],
                'el': el, 'sp_num': sp_num,
                'E_ion': E_ion, 'el_idx': el_idx
            })

    if len(all_data) < 2 or len(el_list) < 1:
        return None

    # Concatenate all data
    x_all = np.concatenate([d['x'] for d in all_data])
    y_all_base = np.concatenate([d['y'] for d in all_data])
    wl_all = np.concatenate([d['wl'] for d in all_data])
    el_idx_all = np.concatenate([np.full(len(d['x']), d['el_idx']) for d in all_data]).astype(int)
    is_ionic = np.concatenate([np.full(len(d['x']), d['sp_num'] == 2) for d in all_data])

    n_elements = len(el_list)
    n_original = len(x_all)
    mask = np.ones(n_original, dtype=bool)

    def _fit(x, y_base, ionic_mask, idx, T_init=10000.0):
        """Run the constrained optimization on the given subset."""
        n_el = len(el_list)

        def objective(params):
            T = params[0]
            intercepts = params[1:]
            if T <= 0:
                return 1e20
            saha_corr = np.log(saha_factor(T, ne_fixed))
            y_corr = y_base.copy()
            y_corr[ionic_mask] -= saha_corr
            slope = -1 / (kB_eV * T)
            y_pred = slope * x + intercepts[idx]
            return np.sum((y_corr - y_pred)**2)

        init_intercepts = [np.mean(y_base) + np.mean(x) / (kB_eV * T_init)] * n_el
        x0 = [T_init] + init_intercepts

        res = minimize(objective, x0, method='L-BFGS-B',
                       bounds=[(3000, 50000)] + [(-50, 50)] * n_el)
        return res.x

    # Initial fit
    params = _fit(x_all, y_all_base, is_ionic, el_idx_all)

    # Iterative sigma-clipping
    do_clip = sigma_clip > 0 and np.isfinite(sigma_clip)
    for _ in range(max_clip_iter if do_clip else 0):
        T_cur = params[0]
        intercepts_cur = params[1:]
        saha_corr = np.log(saha_factor(T_cur, ne_fixed))
        y_corr = y_all_base.copy()
        y_corr[is_ionic] -= saha_corr
        slope_cur = -1 / (kB_eV * T_cur)
        y_pred = slope_cur * x_all + intercepts_cur[el_idx_all]
        resid = y_corr - y_pred
        sigma = np.std(resid[mask])
        if sigma < 1e-15:
            break
        new_mask = np.abs(resid) <= sigma_clip * sigma
        # Ensure at least 3 points per element remain
        safe = True
        for i_el in range(n_elements):
            if np.sum(new_mask & (el_idx_all == i_el)) < 3:
                safe = False
                break
        if not safe or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        # Re-fit on kept points
        params = _fit(x_all[mask], y_all_base[mask], is_ionic[mask],
                      el_idx_all[mask], T_init=T_cur)

    T_fit = params[0]
    intercepts_fit = params[1:]

    # Calculate R² and residuals on kept points
    saha_corr = np.log(saha_factor(T_fit, ne_fixed))
    y_corr = y_all_base.copy()
    y_corr[is_ionic] -= saha_corr
    slope = -1 / (kB_eV * T_fit)
    y_pred = slope * x_all + intercepts_fit[el_idx_all]
    residuals = y_corr - y_pred

    y_corr_kept = y_corr[mask]
    y_pred_kept = y_pred[mask]
    ss_res = np.sum((y_corr_kept - y_pred_kept)**2)
    ss_tot = np.sum((y_corr_kept - np.mean(y_corr_kept))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Wavelength–residual correlation (flags calibration drift)
    if np.sum(mask) > 3 and np.std(wl_all[mask]) > 1e-10:
        from scipy.stats import pearsonr
        corr_wl_resid, _ = pearsonr(wl_all[mask], residuals[mask])
    else:
        corr_wl_resid = np.nan

    return {
        'T': T_fit, 'ne': ne_fixed, 'r2': r2,
        'slope': slope,
        'elements': el_list,
        'intercepts': {el: intercepts_fit[i] for i, el in enumerate(el_list)},
        'data': all_data,
        'n_points': int(np.sum(mask)),
        # Residual diagnostics
        'n_original': n_original,
        'n_clipped': n_original - int(np.sum(mask)),
        'residuals': residuals,
        'mask': mask,
        'wavelengths': wl_all,
        'corr_wl_resid': corr_wl_resid,
    }


# =============================================================================
# CF-LIBS CONCENTRATION CALCULATIONS (Gornushkin 2022)
# =============================================================================

def calculate_cf_concentrations(intercepts: Dict[str, float],
                                 T: float,
                                 ne: float,
                                 ionization_energies: Dict[str, float]) -> Dict:
    """
    Calculate CF-LIBS concentrations using Gornushkin 2022 Eqs. 3.18-3.19.

    Parameters
    ----------
    intercepts : dict
        {element: d_s} where d_s is the Boltzmann plot intercept
    T : float
        Plasma temperature in K
    ne : float
        Electron density in m^-3
    ionization_energies : dict
        {element: E_ion (eV)}

    Returns
    -------
    dict
        {element: {'c0': neutral_conc, 'cp': ionic_conc, 'c_total': total_conc}}
    """
    # Step 1: Calculate F factor (Eq. 3.19)
    # F = Σ_s U0_s * exp(d_s) * (1 + S_s(T)/ne)
    F = 0.0
    for el, d_s in intercepts.items():
        U0_s = partition_function_approx(el, 1, T)
        S_over_ne = saha_function(el, T, ne, ionization_energies)
        F += U0_s * np.exp(d_s) * (1 + S_over_ne)

    # Step 2: Calculate concentrations (Eq. 3.18)
    results = {}
    for el, d_s in intercepts.items():
        U0_s = partition_function_approx(el, 1, T)
        S_over_ne = saha_function(el, T, ne, ionization_energies)

        # Neutral concentration
        c0_s = (U0_s / F) * np.exp(d_s)

        # Ionic concentration
        cp_s = (U0_s * S_over_ne / F) * np.exp(d_s)

        # Total concentration
        c_s = c0_s + cp_s

        results[el] = {'c0': c0_s, 'cp': cp_s, 'c_total': c_s}

    return results


# =============================================================================
# ELECTRON DENSITY FROM STARK BROADENING
# =============================================================================

def compute_ne_from_stark_results(ne_result: Dict,
                                   iqr_factor: float = 1.5) -> Tuple[float, Dict]:
    """
    Compute robust electron density estimate from Stark broadening results.

    Uses IQR-based outlier rejection for robustness.

    Parameters
    ----------
    ne_result : dict
        Output from estimate_ne_from_lines()
    iqr_factor : float
        Multiplier for IQR outlier rejection (default 1.5)

    Returns
    -------
    ne_median : float
        Median electron density after outlier rejection
    stats : dict
        Statistics including mean, std, n_fits, n_clean
    """
    if not ne_result.get('success', False):
        return np.nan, {'error': ne_result.get('error', 'Unknown')}

    ne_values = np.array([f['ne'] for f in ne_result['fits']])

    # Robust statistics with IQR outlier rejection
    q1, q3 = np.percentile(ne_values, [25, 75])
    iqr = q3 - q1
    mask = (ne_values >= q1 - iqr_factor*iqr) & (ne_values <= q3 + iqr_factor*iqr)
    ne_clean = ne_values[mask]

    ne_median = np.median(ne_clean)

    return ne_median, {
        'n_fits': len(ne_values),
        'n_clean': len(ne_clean),
        'median': ne_median,
        'mean': np.mean(ne_clean),
        'std': np.std(ne_clean)
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def filter_lines_by_sp_num(isolated_lines: Dict[str, pd.DataFrame],
                            sp_num: int) -> Dict[str, pd.DataFrame]:
    """
    Filter isolated lines dictionary to a specific ionization state.

    Parameters
    ----------
    isolated_lines : dict
        {element: DataFrame with 'sp_num' column}
    sp_num : int
        Ionization state to filter (1 = neutral, 2 = ionic)

    Returns
    -------
    dict
        {element: DataFrame filtered to sp_num}
    """
    return {el: df[df['sp_num'] == sp_num]
            for el, df in isolated_lines.items() if len(df) > 0}


def run_individual_boltzmann_fits(isolated_lines: Dict[str, pd.DataFrame],
                                   target_elements: List[str],
                                   sigma_threshold: float = 2.5,
                                   max_clip_iter: int = 3) -> Dict[int, Dict]:
    """
    Run individual Boltzmann fits for each element and ionization state.

    Parameters
    ----------
    isolated_lines : dict
        {element: DataFrame with intensity and sp_num columns}
    target_elements : list
        Elements to fit
    sigma_threshold : float
        Outlier rejection threshold for single_element_boltzmann_fit
        (default 2.5).
    max_clip_iter : int
        Maximum sigma-clipping iterations (default 3).

    Returns
    -------
    dict
        {sp_num: {element: fit_result}}
    """
    results = {1: {}, 2: {}}

    for sp_num in [1, 2]:
        for el in target_elements:
            if el not in isolated_lines or len(isolated_lines[el]) == 0:
                continue

            lines_sp = isolated_lines[el][isolated_lines[el]['sp_num'] == sp_num]
            if len(lines_sp) < 3:
                continue

            result = single_element_boltzmann_fit(
                lines_sp,
                sigma_threshold=sigma_threshold,
                max_clip_iter=max_clip_iter
            )
            if result is not None and result['T'] > 0:
                results[sp_num][el] = result

    return results


def run_constrained_boltzmann_fits(isolated_lines: Dict[str, pd.DataFrame],
                                    sigma_clip: float = 0.0,
                                    max_clip_iter: int = 3) -> Dict[int, Optional[Dict]]:
    """
    Run constrained multi-element Boltzmann fits for each ionization state.

    Parameters
    ----------
    isolated_lines : dict
        {element: DataFrame with intensity and sp_num columns}
    sigma_clip : float
        Outlier rejection threshold for multi_element_boltzmann_fit.
        Set to 0 to disable (default).
    max_clip_iter : int
        Maximum sigma-clipping iterations (default 3).

    Returns
    -------
    dict
        {sp_num: fit_result or None}
    """
    return {
        1: multi_element_boltzmann_fit(
            filter_lines_by_sp_num(isolated_lines, 1),
            sigma_clip=sigma_clip, max_clip_iter=max_clip_iter
        ),
        2: multi_element_boltzmann_fit(
            filter_lines_by_sp_num(isolated_lines, 2),
            sigma_clip=sigma_clip, max_clip_iter=max_clip_iter
        )
    }


def run_individual_saha_boltzmann_fits(isolated_lines: Dict[str, pd.DataFrame],
                                        target_elements: List[str],
                                        ionization_energies: Dict[str, float],
                                        ne_stark: float) -> Dict[str, Dict]:
    """
    Run individual Saha-Boltzmann fits for each element.

    Parameters
    ----------
    isolated_lines : dict
        {element: DataFrame with both I and II lines}
    target_elements : list
        Elements to fit
    ionization_energies : dict
        {element: E_ion (eV)}
    ne_stark : float
        Electron density from Stark broadening

    Returns
    -------
    dict
        {element: fit_result}
    """
    results = {}

    for el in target_elements:
        if el not in ionization_energies or el not in isolated_lines:
            continue

        df = isolated_lines[el]
        el_I = df[df['sp_num'] == 1]
        el_II = df[df['sp_num'] == 2]

        if len(el_I) < 2 or len(el_II) < 2:
            continue

        result = saha_boltzmann_fit_single_element(
            el_I, el_II, ionization_energies[el], ne_stark
        )
        if result:
            results[el] = result

    return results
