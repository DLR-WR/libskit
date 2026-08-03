#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Extended Boltzmann Plot Method for Multiplet Emission Lines

Implementation based on:
    T. Völker and I.B. Gornushkin, "Extension of the Boltzmann plot method
    for multiplet emission lines", Journal of Quantitative Spectroscopy and
    Radiative Transfer 310 (2023) 108741
    https://doi.org/10.1016/j.jqsrt.2023.108741

This module provides functions for determining plasma temperature and electron
density from LIBS spectra using the extended Saha-Boltzmann plot method with
direct integration of multiplet lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import constants as const
from scipy.stats import linregress  # type: ignore[import-untyped]
from scipy.integrate import trapezoid
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Physical constants
h = const.h           # Planck constant (J·s)
c = const.c           # Speed of light (m/s)
kB = const.k          # Boltzmann constant (J/K)
kB_eV = const.k / const.eV  # Boltzmann constant (eV/K)
me = const.m_e        # Electron mass (kg)


def _load_line_database(database_path: str | None = None) -> pd.DataFrame:
    """Load the spectral lines database."""
    if database_path is None:
        try:
            from libskit import Database
            active = Database.active()
            if active is None:
                raise RuntimeError(
                    "No active libskit Database. Construct one via "
                    "libskit.Database.bundled() / .from_path(...) and call "
                    "its activate() method, or pass database_path explicitly."
                )
            database_path = str(active.path)
        except ImportError:
            raise ImportError("Cannot find database path. Install libskit package or provide database_path.")

    lines_file = Path(database_path) / "lines_processed_all.csv"
    if not lines_file.exists():
        raise FileNotFoundError(f"Lines database not found at {lines_file}")

    return pd.read_csv(lines_file)


def _load_ionization_database(database_path: str | None = None) -> pd.DataFrame:
    """Load the ionization energies database."""
    if database_path is None:
        try:
            from libskit import Database
            active = Database.active()
            if active is None:
                raise RuntimeError(
                    "No active libskit Database. Construct one via "
                    "libskit.Database.bundled() / .from_path(...) and call "
                    "its activate() method, or pass database_path explicitly."
                )
            database_path = str(active.path)
        except ImportError:
            raise ImportError("Cannot find database path. Install libskit package or provide database_path.")

    ion_file = Path(database_path) / "ionization_energies_processed.csv"
    if not ion_file.exists():
        raise FileNotFoundError(f"Ionization database not found at {ion_file}")

    return pd.read_csv(ion_file)


def _get_ionization_energy(element: str, ionization_db: pd.DataFrame) -> float:
    """Get first ionization energy for an element from database."""
    sp_name = f"{element} I"
    match = ionization_db[ionization_db['Sp. Name'] == sp_name]
    if len(match) > 0:
        return match.iloc[0]['Ionization Energy (b) (eV)']
    else:
        raise ValueError(f"Ionization energy not found for {element}")


def _get_line_params(lines_db: pd.DataFrame, element: str, sp_num: int,
                     target_wl: float, tolerance: float = 0.15) -> Optional[Dict]:
    """Get line parameters from database."""
    mask = (
        (lines_db['element'] == element) &
        (lines_db['sp_num'] == sp_num) &
        (np.abs(lines_db['ritz_wl_vac(nm)'] - target_wl) < tolerance)
    )
    matches = lines_db[mask]

    if len(matches) == 0:
        return None

    idx = np.argmin(np.abs(matches['ritz_wl_vac(nm)'] - target_wl))
    row = matches.iloc[idx]

    return {
        'wavelength': row['ritz_wl_vac(nm)'],
        'Aki': row['Aki(s^-1)'],
        'Ek': row['Ek(eV)'],
        'gk': row['g_k'],
        'Ei': row['Ei(eV)'],
        'gi': row['g_i'],
        'sp_num': sp_num
    }


def _group_multiplets(wavelengths: List[float], threshold: float = 2.0) -> List[List[float]]:
    """Group nearby lines into multiplets."""
    if len(wavelengths) == 0:
        return []

    sorted_wls = sorted(wavelengths)
    groups = []
    current_group = [sorted_wls[0]]

    for wl in sorted_wls[1:]:
        if wl - current_group[-1] < threshold:
            current_group.append(wl)
        else:
            groups.append(current_group)
            current_group = [wl]
    groups.append(current_group)

    return groups


def _multiplet_correction_term(line_params: List[Dict], T: float, ref_idx: int = 0) -> float:
    """
    Calculate the multiplet correction term from Eq. (5) of the paper.
    """
    if len(line_params) <= 1:
        return 0.0

    ref = line_params[ref_idx]
    ref_term = ref['gk'] * ref['Aki'] / ref['wavelength'] * np.exp(-ref['Ek'] / (kB_eV * T))

    sum_others = 0.0
    for i, p in enumerate(line_params):
        if i != ref_idx:
            sum_others += p['gk'] * p['Aki'] / p['wavelength'] * np.exp(-p['Ek'] / (kB_eV * T))

    correction = np.log(1 + sum_others / ref_term)
    return correction


def _saha_factor(T: float, ne: float) -> float:
    """
    Calculate ln(2*(2*pi*me*kB*T)^1.5 / (n_e * h^3))
    """
    prefactor = 2 * (2 * np.pi * me * kB * T)**1.5 / (ne * h**3)
    return np.log(prefactor)


def _planck_function(wavelength_nm: np.ndarray, T: float) -> np.ndarray:
    """Planck function for blackbody radiation. wavelength in nm, T in K."""
    wl_m = wavelength_nm * 1e-9
    return (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * kB * T)) - 1)


def _self_absorption_correction(wl_nm: np.ndarray, I_obs: np.ndarray, T: float) -> np.ndarray:
    """
    Correct for self-absorption using Eq. (8) from the paper.
    """
    B = _planck_function(wl_nm, T)
    scale = np.max(I_obs) / np.max(B) * 2.0
    I_obs_scaled = I_obs / scale

    I_corrected = np.copy(I_obs)
    valid = I_obs_scaled < B * 0.99

    ratio = I_obs_scaled[valid] / B[valid]
    correction_factor = -np.log(1 - ratio) / ratio
    I_corrected[valid] = I_obs[valid] * correction_factor

    return I_corrected


def _measure_line_width(spectrum_wl: np.ndarray, spectrum_I: np.ndarray,
                        wl_center: float, search_range: float = 1.0) -> Optional[float]:
    """Measure the FWHM of a spectral line. Returns FWHM in nm."""
    if not np.isfinite(wl_center):
        return None

    mask = (spectrum_wl >= wl_center - search_range) & (spectrum_wl <= wl_center + search_range)
    wl_local = spectrum_wl[mask]
    I_local = spectrum_I[mask]

    if len(I_local) < 5:
        return None

    peak_idx = np.argmax(I_local)
    I_max = I_local[peak_idx]
    wl_peak = wl_local[peak_idx]
    half_max = I_max / 2

    left_mask = wl_local < wl_peak
    if np.any(I_local[left_mask] < half_max):
        left_idx = np.where(left_mask & (I_local < half_max))[0][-1]
        wl_left = wl_local[left_idx]
    else:
        wl_left = wl_local[0]

    right_mask = wl_local > wl_peak
    if np.any(I_local[right_mask] < half_max):
        right_idx = np.where(right_mask & (I_local < half_max))[0][0]
        wl_right = wl_local[right_idx]
    else:
        wl_right = wl_local[-1]

    return wl_right - wl_left


def _electron_density_from_stark(fwhm_measured: float, T: float,
                                  fwhm_inst: float, wl_nm: float = 393.37) -> float:
    """
    Calculate electron density from Stark broadening.
    """
    fwhm_stark_sq = fwhm_measured**2 - fwhm_inst**2
    if fwhm_stark_sq <= 0:
        return 1e22

    fwhm_stark = np.sqrt(fwhm_stark_sq)

    w_ref = 0.015
    T_ref = 10000
    w = w_ref * (T / T_ref)**0.1

    ne_cm3 = (fwhm_stark / (2 * w)) * 1e16
    ne = ne_cm3 * 1e6

    return ne


def _detect_lines(lines_db: pd.DataFrame, spectrum_wl: np.ndarray, spectrum_I: np.ndarray,
                  element: str, sp_num: int, min_Aki: float = 1e6,
                  noise_threshold: float = 5.0, noise_level: float = 0.001) -> List[Dict]:
    """
    Detect lines of a given element/ion that are above the noise floor.
    """
    wl_min, wl_max = spectrum_wl.min(), spectrum_wl.max()

    # Filter database
    mask = (
        (lines_db['element'] == element) &
        (lines_db['sp_num'] == sp_num) &
        (lines_db['ritz_wl_vac(nm)'] >= wl_min) &
        (lines_db['ritz_wl_vac(nm)'] <= wl_max) &
        (lines_db['Aki(s^-1)'] >= min_Aki)
    )
    candidates = lines_db[mask]

    detected = []
    noise_abs = noise_level * np.max(spectrum_I)

    for _, row in candidates.iterrows():
        wl = row['ritz_wl_vac(nm)']
        idx = np.argmin(np.abs(spectrum_wl - wl))
        local_max = spectrum_I[max(0, idx-5):min(len(spectrum_I), idx+5)].max()

        if local_max > noise_threshold * noise_abs:
            detected.append({
                'wavelength': row['ritz_wl_vac(nm)'],
                'Aki': row['Aki(s^-1)'],
                'Ek': row['Ek(eV)'],
                'gk': row['g_k'],
                'Ei': row['Ei(eV)'],
                'gi': row['g_i'],
                'sp_num': sp_num
            })

    # Deduplicate
    seen = set()
    unique = []
    for p in detected:
        wl_key = round(p['wavelength'], 2)
        if wl_key not in seen:
            seen.add(wl_key)
            unique.append(p)

    return unique


def _build_saha_boltzmann_data(T: float, ne: float, spectrum_wl: np.ndarray,
                                spectrum_I: np.ndarray, neutral_params: List[Dict],
                                ion_params: List[Dict], chi: float,
                                multiplet_threshold: float = 2.0) -> List[Tuple]:
    """
    Build Saha-Boltzmann plot data using direct integration method.
    """
    data_points = []

    def get_integral(wl_center: float, wl_range: float) -> Optional[float]:
        mask = (spectrum_wl >= wl_center - wl_range) & (spectrum_wl <= wl_center + wl_range)
        if np.sum(mask) < 2:
            return None
        return trapezoid(spectrum_I[mask], spectrum_wl[mask])

    def boltzmann_y_singlet(I_int: float, wl: float, gk: float, Aki: float) -> float:
        return np.log(I_int * wl / (gk * Aki))

    def boltzmann_y_multiplet(I_int: float, line_params: List[Dict], T: float, ref_idx: int = 0) -> float:
        ref = line_params[ref_idx]
        y_base = np.log(I_int * ref['wavelength'] / (ref['gk'] * ref['Aki']))
        correction = _multiplet_correction_term(line_params, T, ref_idx)
        return y_base - correction

    def saha_boltzmann_y_ion(I_int: float, line_params: List[Dict], T: float,
                            ne: float, ref_idx: int = 0) -> float:
        ref = line_params[ref_idx]
        y_base = np.log(I_int * ref['wavelength'] / (ref['gk'] * ref['Aki']))
        saha_corr = _saha_factor(T, ne)
        multiplet_corr = _multiplet_correction_term(line_params, T, ref_idx)
        return y_base - saha_corr - multiplet_corr

    # Process neutral lines
    neutral_groups = _group_multiplets([p['wavelength'] for p in neutral_params], threshold=multiplet_threshold)
    data_points.extend(_collect_group_points(
        neutral_groups, neutral_params, get_integral, T, ne, chi,
        ion_state='neutral',
        boltzmann_y_singlet=boltzmann_y_singlet,
        boltzmann_y_multiplet=boltzmann_y_multiplet,
        saha_boltzmann_y_ion=saha_boltzmann_y_ion,
    ))

    # Process ion lines
    ion_groups = _group_multiplets([p['wavelength'] for p in ion_params], threshold=multiplet_threshold)
    data_points.extend(_collect_group_points(
        ion_groups, ion_params, get_integral, T, ne, chi,
        ion_state='ion',
        boltzmann_y_singlet=boltzmann_y_singlet,
        boltzmann_y_multiplet=boltzmann_y_multiplet,
        saha_boltzmann_y_ion=saha_boltzmann_y_ion,
    ))

    return data_points


def _collect_group_points(groups: List[List[float]], all_params: List[Dict],
                          get_integral, T: float, ne: float, chi: float,
                          ion_state: str,
                          boltzmann_y_singlet, boltzmann_y_multiplet,
                          saha_boltzmann_y_ion) -> List[Tuple]:
    """Build Saha-Boltzmann data points for one ionisation state.

    Iterates over multiplet groups, integrates the local intensity, picks the
    strongest line as reference, and returns ``(E_eff, y, ion_state, n_lines)``
    tuples consumed by ``_build_saha_boltzmann_data``.
    """
    points: List[Tuple] = []
    for group_wls in groups:
        group_params = [
            p for p in all_params
            if any(abs(p['wavelength'] - wl) < 0.1 for wl in group_wls)
        ]
        if not group_params:
            continue

        wl_center = float(np.mean([p['wavelength'] for p in group_params]))
        wl_range = max(1.5, (max(group_wls) - min(group_wls)) / 2 + 1.0)

        I_int = get_integral(wl_center, wl_range)
        if I_int is None or I_int <= 0:
            continue

        ref_idx = int(np.argmax([p['gk'] * p['Aki'] for p in group_params]))
        ref = group_params[ref_idx]

        if ion_state == 'neutral':
            if len(group_params) == 1:
                y = boltzmann_y_singlet(I_int, ref['wavelength'], ref['gk'], ref['Aki'])
            else:
                y = boltzmann_y_multiplet(I_int, group_params, T, ref_idx)
            E_eff = ref['Ek']
        else:  # ion
            y = saha_boltzmann_y_ion(I_int, group_params, T, ne, ref_idx)
            E_eff = ref['Ek'] + chi

        points.append((E_eff, y, ion_state, len(group_params)))
    return points


def _fit_boltzmann_temperature(data_points: List[Tuple]) -> Tuple[float, float, float, float, float]:
    """
    Fit temperature from Boltzmann plot data using linear regression.
    Returns: T, T_err, R2, intercept, slope
    """
    if len(data_points) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x = np.array([d[0] for d in data_points])
    y = np.array([d[1] for d in data_points])

    # Check for valid data
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x = x[valid]
    y = y[valid]

    result = linregress(x, y)

    # Slope must be negative for valid temperature
    if result.slope >= 0:
        return np.nan, np.nan, result.rvalue**2, result.intercept, result.slope

    T = -1 / (kB_eV * result.slope)
    T_err = T * abs(result.stderr / result.slope) if result.stderr else 0

    return T, T_err, result.rvalue**2, result.intercept, result.slope


def estimate_from_bp_gornushkin(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    element: str = "Ca",
    fwhm_inst: float = 0.25,
    ne_known: Optional[float] = None,
    T_init: float = 8000.0,
    ne_init: float = 1e23,
    max_iter: int = 20,
    tol: float = 0.001,
    apply_self_absorption: bool = True,
    min_Aki: float = 1e6,
    noise_level: float = 0.001,
    database_path: Optional[str] = None,
    verbose: bool = False
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Estimate plasma temperature and electron density from a LIBS spectrum
    using the extended Saha-Boltzmann plot method with direct integration.

    Based on: Völker & Gornushkin, JQSRT 310 (2023) 108741

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array in nm
    intensities : np.ndarray
        Intensity array (arbitrary units)
    element : str, optional
        Element to use for the Boltzmann plot (default: "Ca")
    fwhm_inst : float, optional
        Instrumental FWHM in nm (default: 0.25)
    ne_known : float, optional
        If provided, use this known electron density instead of estimating
        from Stark broadening. Use this for "pure T effect" investigation.
    T_init : float, optional
        Initial temperature guess in K (default: 8000)
    ne_init : float, optional
        Initial electron density guess in m^-3 (default: 1e23)
    max_iter : int, optional
        Maximum iterations (default: 20)
    tol : float, optional
        Convergence tolerance for relative change in T (default: 0.001)
    apply_self_absorption : bool, optional
        Whether to apply self-absorption correction (default: True)
    min_Aki : float, optional
        Minimum Einstein A coefficient for line selection (default: 1e6)
    noise_level : float, optional
        Estimated noise level as fraction of max intensity (default: 0.001)
    database_path : str, optional
        Path to database directory. If None, uses default from libskit.
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    T : float
        Estimated temperature in K
    ne : float
        Estimated electron density in m^-3
    info : dict
        Additional information including:
        - 'history': dict with T, ne, R2 at each iteration
        - 'data_points': final Saha-Boltzmann data points
        - 'n_neutral_lines': number of neutral lines used
        - 'n_ion_lines': number of ion lines used
        - 'R2': final R² value
        - 'converged': whether the algorithm converged

    Examples
    --------
    >>> import numpy as np
    >>> import libskit
    >>>
    >>> # Generate a synthetic spectrum (bundled database is auto-activated
    >>> # on import; pass db= explicitly only when overriding it).
    >>> elements = {"Ca": 1.0, "Si": 0.3, "Fe": 0.05}
    >>> model = libskit.LTEModel(elements, 10000.0, 5e23, 0.5e-3)
    >>> spec = libskit.GenericSpectrometer(0.1, 0.0)
    >>> wavelengths = np.arange(200, 950, 0.1)
    >>> intensities = spec.calculate_spectrum(wavelengths, model)
    >>>
    >>> # Estimate T and ne
    >>> T, ne, info = libskit.estimate_from_bp_gornushkin(
    ...     wavelengths, intensities, element="Ca"
    ... )
    >>> print(f"T = {T:.0f} K, ne = {ne:.2e} m^-3")
    """
    # Load databases
    lines_db = _load_line_database(database_path)
    ionization_db = _load_ionization_database(database_path)

    # Get ionization energy
    chi = _get_ionization_energy(element, ionization_db)

    # Detect lines
    neutral_params = _detect_lines(lines_db, wavelengths, intensities,
                                    element, sp_num=1, min_Aki=min_Aki,
                                    noise_level=noise_level)
    ion_params = _detect_lines(lines_db, wavelengths, intensities,
                                element, sp_num=2, min_Aki=min_Aki,
                                noise_level=noise_level)

    if verbose:
        print(f"Detected {len(neutral_params)} {element} I lines and {len(ion_params)} {element} II lines")

    if len(neutral_params) < 2:
        raise ValueError(f"Insufficient {element} I lines detected ({len(neutral_params)}). "
                        f"Need at least 2 for Boltzmann plot.")

    # Initialize
    T = T_init
    T_prev = 0
    ne = ne_known if ne_known is not None else ne_init

    I_spectrum = intensities.copy()

    history = {
        'T': [T],
        'ne': [ne],
        'R2': [],
        'relative_change': [1.0]
    }

    # Find Stark broadening line (use strongest ion line)
    if ion_params:
        stark_line_wl = max(ion_params, key=lambda p: p['Aki'])['wavelength']
    else:
        stark_line_wl = None

    # Iterative algorithm
    i = 0
    converged = False
    while abs(T - T_prev) / T > tol and i < max_iter:
        T_prev = T

        # Step: Get electron density
        if ne_known is not None:
            ne = ne_known
        elif stark_line_wl is not None:
            fwhm = _measure_line_width(wavelengths, I_spectrum, stark_line_wl, search_range=1.5)
            if fwhm is not None and fwhm > fwhm_inst:
                ne = _electron_density_from_stark(fwhm, T, fwhm_inst, stark_line_wl)
            ne = np.clip(ne, 1e21, 1e26)

        # Step: Build Saha-Boltzmann data and fit temperature
        data_points = _build_saha_boltzmann_data(T, ne, wavelengths, I_spectrum,
                                                  neutral_params, ion_params, chi)

        if len(data_points) < 3:
            if verbose:
                print(f"Warning: Only {len(data_points)} data points, insufficient for fit")
            break

        T_new, T_err, R2, intercept, slope = _fit_boltzmann_temperature(data_points)

        # Check for valid temperature
        if np.isnan(T_new) or T_new <= 0:
            if verbose:
                print(f"  Iteration {i}: Invalid temperature (slope={slope:.6f}), skipping update")
            i += 1
            continue

        # Step: Apply self-absorption correction
        if apply_self_absorption:
            I_spectrum = _self_absorption_correction(wavelengths, intensities, T_new)

        rel_change = abs(T_new - T) / T if T > 0 else 1.0
        T = T_new

        history['T'].append(T)
        history['ne'].append(ne)
        history['R2'].append(R2)
        history['relative_change'].append(rel_change)

        i += 1

        if rel_change < tol:
            converged = True
            break

    if verbose:
        if converged:
            print(f"Converged after {i} iterations")
        else:
            print(f"Reached max iterations ({max_iter})")
        print(f"T = {T:.0f} K, ne = {ne:.2e} m^-3, R² = {R2:.4f}")

    # Get final data points
    final_data = _build_saha_boltzmann_data(T, ne, wavelengths, I_spectrum,
                                             neutral_params, ion_params, chi)

    # Count lines
    n_neutral = sum(1 for d in final_data if d[2] == 'neutral')
    n_ion = sum(1 for d in final_data if d[2] == 'ion')

    info = {
        'history': history,
        'data_points': final_data,
        'n_neutral_lines': n_neutral,
        'n_ion_lines': n_ion,
        'R2': history['R2'][-1] if history['R2'] else None,
        'converged': converged,
        'iterations': i,
        'element': element,
        'chi': chi
    }

    return T, ne, info


# =============================================================================
# Public utility functions
# =============================================================================

def get_ionization_energy(element: str, database_path: Optional[str] = None) -> float:
    """
    Get the first ionization energy for an element from the database.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Ca', 'Fe', 'Mg')
    database_path : str, optional
        Path to database directory. If None, uses default from libskit.

    Returns
    -------
    float
        First ionization energy in eV

    Raises
    ------
    ValueError
        If element not found in database
    """
    ionization_db = _load_ionization_database(database_path)
    return _get_ionization_energy(element, ionization_db)


def get_ionization_energies(elements: List[str], database_path: Optional[str] = None) -> Dict[str, float]:
    """
    Get ionization energies for multiple elements from the database.

    Parameters
    ----------
    elements : list of str
        List of element symbols (e.g., ['Ca', 'Fe', 'Mg'])
    database_path : str, optional
        Path to database directory. If None, uses default from libskit.

    Returns
    -------
    dict
        Mapping of element symbol to first ionization energy in eV

    Raises
    ------
    ValueError
        If any element not found in database
    """
    ionization_db = _load_ionization_database(database_path)
    return {el: _get_ionization_energy(el, ionization_db) for el in elements}


def boltzmann_fit(lines_df: pd.DataFrame,
                  intensity_col: str = 'intensity',
                  sigma_threshold: float = 2.5) -> Optional[Dict]:
    """
    Perform a Boltzmann plot fit with iterative outlier rejection.

    Fits the Boltzmann equation (V\u00f6lker & Gornushkin 2023, Eq. 1):
        ln(I*\u03bb / (g_k * A_ki)) = -E_k / (k_B * T) + C

    The slope ``-1/(k_B T)`` recovers the plasma temperature; the
    iterative loop rejects up to three rounds of points whose linear-fit
    residual exceeds ``sigma_threshold`` standard deviations, which
    mitigates the influence of self-absorbed or mis-identified lines
    discussed in V\u00f6lker & Gornushkin (2023), \u00a72.

    Parameters
    ----------
    lines_df : pd.DataFrame
        DataFrame with columns: 'ritz_wl_vac(nm)', 'Aki(s^-1)', 'Ek(eV)',
        'g_k', and the intensity column
    intensity_col : str
        Name of the intensity column (default: 'intensity')
    sigma_threshold : float
        Outlier rejection threshold in standard deviations (default: 2.5)

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'T': fitted temperature in K
        - 'T_err': temperature uncertainty in K
        - 'r2': R² value of the fit
        - 'n_points': number of points used
        - 'x': E_k values used
        - 'y': ln(I*λ / g_k*A_ki) values used
        - 'slope': fitted slope
        - 'intercept': fitted intercept
        Returns None if fit fails
    """
    wl = lines_df['ritz_wl_vac(nm)'].values
    Aki = lines_df['Aki(s^-1)'].values
    Ek = lines_df['Ek(eV)'].values
    gk = lines_df['g_k'].values
    I = lines_df[intensity_col].values

    y = np.log(I * wl / (gk * Aki))
    x = Ek

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 3:
        return None

    # Check for identical x values
    if np.max(x) == np.min(x):
        return None

    # Iterative outlier rejection
    for _ in range(3):
        if len(x) < 3 or np.max(x) == np.min(x):
            break
        result = linregress(x, y)
        residuals = y - (result.slope * x + result.intercept)
        std_res = np.std(residuals)
        mask = np.abs(residuals) < sigma_threshold * std_res
        if mask.all():
            break
        x, y = x[mask], y[mask]

    if len(x) < 3:
        return None

    result = linregress(x, y)
    T = -1 / (result.slope * kB_eV)
    T_err = abs(T) * (result.stderr / abs(result.slope)) if result.stderr else 0

    return {
        'T': T,
        'T_err': T_err,
        'r2': result.rvalue**2,
        'n_points': len(x),
        'x': x,
        'y': y,
        'slope': result.slope,
        'intercept': result.intercept
    }


def saha_factor(T: float, ne: float) -> float:
    """
    Calculate the Saha ionization factor.

    Implements the right-hand side of the Saha equation as used in the
    Saha-Boltzmann plot of V\u00f6lker & Gornushkin (2023), Eq. 5:

        S(T, n_e) = 2 (2\u03c0 m_e k_B T)^{3/2} / (n_e h^3)

    Pre-multiplying the upper-ion intensities by ``S(T, n_e)`` puts both
    ionization stages on the same Boltzmann axis so a single linear fit
    yields the temperature.

    Parameters
    ----------
    T : float
        Temperature in K
    ne : float
        Electron density in m^-3

    Returns
    -------
    float
        Saha factor (dimensionless)
    """
    return 2 * (2 * np.pi * me * kB * T)**(3/2) / (ne * h**3)
