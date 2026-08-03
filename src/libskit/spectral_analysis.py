#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Spectral Analysis Utilities for LIBS/Emission Spectroscopy

Functions for:
1. Peak detection and emission line identification
2. Finding well-separated lines for Boltzmann analysis
3. Measuring integrated line intensities
4. Electron density estimation from Stark broadening

Based on:
- NIST Atomic Spectra Database conventions
- Stark-B database parameterization (Sahal-Bréchot et al.)
- Standard CF-LIBS methodology
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import constants as const
from scipy.signal import find_peaks, savgol_filter
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from collections import Counter
from typing import Optional, List, Dict, Tuple, Union


# =============================================================================
# PHYSICAL CONSTANTS (from scipy.constants)
# =============================================================================
kB_eV = const.k / const.eV  # Boltzmann constant in eV/K


# =============================================================================
# VOIGT PROFILE FITTING
# =============================================================================

def voigt(x: np.ndarray, amplitude: float, center: float,
          sigma: float, gamma: float) -> np.ndarray:
    """
    Voigt profile for spectral line fitting.

    Parameters
    ----------
    x : array
        Wavelength positions
    amplitude : float
        Peak amplitude (area under curve)
    center : float
        Line center wavelength
    sigma : float
        Gaussian standard deviation (Doppler broadening)
    gamma : float
        Lorentzian HWHM (Stark + natural broadening)

    Returns
    -------
    array
        Voigt profile values
    """
    return amplitude * voigt_profile(x - center, sigma, gamma)


def _voigt_offset(x: np.ndarray, amplitude: float, center: float,
                  sigma: float, gamma: float, offset: float) -> np.ndarray:
    """Voigt profile with a constant offset — used internally by curve_fit."""
    return amplitude * voigt_profile(x - center, sigma, gamma) + offset


def fit_voigt_profile(wavelengths: np.ndarray, spectrum: np.ndarray,
                      line_center: float, window: float = 0.3) -> Dict:
    """
    Fit a Voigt profile + constant offset to a spectral line.

    The model is:  amplitude * V(x; center, sigma, gamma) + offset

    Parameters
    ----------
    wavelengths : array
        Wavelength array in nm
    spectrum : array
        Intensity array
    line_center : float
        Expected line center in nm
    window : float
        Half-width of fitting window in nm

    Returns
    -------
    dict with fit results or error information

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.linspace(399, 401, 200)
    >>> spectrum = 100 * np.exp(-0.5 * ((wl - 400.0) / 0.05) ** 2) + 10
    >>> result = fit_voigt_profile(wl, spectrum, 400.0, window=0.5)
    >>> result['success']
    True
    >>> abs(result['center'] - 400.0) < 0.01
    True
    """
    mask = (wavelengths > line_center - window) & (wavelengths < line_center + window)
    if np.sum(mask) < 10:
        return {'success': False, 'error': 'Insufficient data points'}

    wl_local = wavelengths[mask]
    spec_local = spectrum[mask]

    # Initial guesses — use the requested line_center position directly,
    # NOT argmax(window), which can snap to a neighboring peak in crowded spectra
    nearest_idx = np.argmin(np.abs(wl_local - line_center))
    amp_init = spec_local[nearest_idx]
    center_init = line_center  # Start from the detected peak position

    # Estimate initial offset from the edges of the window
    n_edge = max(3, len(spec_local) // 10)
    offset_init = float(np.mean(np.concatenate([
        spec_local[:n_edge], spec_local[-n_edge:]])))
    amp_init = max(amp_init - offset_init, offset_init * 0.1)

    # Estimate initial width from half-maximum points around the target peak
    half_max = amp_init / 2 + offset_init
    above_half = spec_local > half_max
    if np.sum(above_half) > 1:
        wl_above = wl_local[above_half]
        fwhm_est = wl_above[-1] - wl_above[0]
        sigma_init = fwhm_est / 2.355 / 2  # Split between Gauss and Lorentz
        gamma_init = fwhm_est / 4
    else:
        sigma_init = 0.01
        gamma_init = 0.01

    # Center bounds scale with the window — allow optimizer to refine position
    # but prevent it from jumping to a distant neighbor
    center_margin = min(window * 0.5, 0.2)
    amp_upper = max(np.max(spec_local) * 10, 1.0)
    offset_lower = min(-np.max(np.abs(spec_local)), 0)
    offset_upper = np.max(spec_local)

    try:
        popt, pcov = curve_fit(
            _voigt_offset, wl_local, spec_local,
            p0=[amp_init, center_init, sigma_init, gamma_init, offset_init],
            bounds=([0, line_center - center_margin, 1e-5, 1e-5, offset_lower],
                    [amp_upper, line_center + center_margin, 1, 1, offset_upper]),
            maxfev=4000
        )

        amplitude, center, sigma, gamma, offset = popt
        perr = np.sqrt(np.diag(pcov))

        # Calculate FWHM components
        fwhm_gauss = 2.355 * sigma  # 2*sqrt(2*ln(2))*sigma
        fwhm_lorentz = 2 * gamma

        # Residual quality metric
        fit_curve = _voigt_offset(wl_local, *popt)
        residual_rms = np.sqrt(np.mean((spec_local - fit_curve)**2))
        residual_rel = residual_rms / np.max(spec_local)

        return {
            'success': True,
            'amplitude': amplitude,
            'center': center,
            'sigma': sigma,
            'gamma': gamma,
            'offset': offset,
            'sigma_err': perr[2],
            'gamma_err': perr[3],
            'offset_err': perr[4],
            'fwhm_gauss': fwhm_gauss,
            'fwhm_lorentz': fwhm_lorentz,
            'residual_rel': residual_rel,
            'wl_local': wl_local,
            'spec_local': spec_local,
            'fit_curve': fit_curve
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


# =============================================================================
# PEAK DETECTION AND LINE IDENTIFICATION
# =============================================================================

def estimate_baseline_noise(spectrum: np.ndarray) -> float:
    """
    Estimate baseline noise level for emission spectra.

    Uses the Median Absolute Deviation (MAD) of point-to-point differences,
    computed only on non-zero differences to handle clipped emission spectra
    where many adjacent points are both zero.

    This method is robust to:
    - Clipped negative values (common in emission spectra)
    - Slowly varying continuum
    - Presence of peaks

    Parameters
    ----------
    spectrum : array
        Intensity array (non-negative for emission spectra)

    Returns
    -------
    float
        Estimated noise standard deviation

    Examples
    --------
    >>> import numpy as np
    >>> noise = np.random.normal(0, 0.01, 1000)
    >>> sigma = estimate_baseline_noise(noise)
    >>> abs(sigma - 0.01) < 0.005
    True
    """
    # Use point-to-point differences to estimate noise
    diff = np.diff(spectrum)

    # Filter out zero differences (from clipped baseline regions)
    nonzero_diff = diff[np.abs(diff) > 1e-12]

    if len(nonzero_diff) < 10:
        # Fallback if almost all diffs are zero
        return float(np.std(spectrum) / 10)

    # Use MAD of non-zero differences (robust to peak edges)
    # MAD = median(|x - median(x)|)
    median_diff = np.median(nonzero_diff)
    mad = np.median(np.abs(nonzero_diff - median_diff))

    # Convert MAD to std (factor of 1.4826 for Gaussian)
    # and correct for sqrt(2) factor from differencing
    noise_std = float(mad * 1.4826 / np.sqrt(2))

    # Ensure we don't return exactly zero
    if noise_std < 1e-10:
        noise_std = float(np.std(spectrum) / 100)

    return noise_std


def find_spectrum_peaks(wavelengths: np.ndarray, spectrum: np.ndarray,
                        height_fraction: float = 0.01,
                        prominence_fraction: float = 0.005,
                        min_distance_nm: float = 0.05,
                        noise_threshold_sigma: float = 5.0,
                        use_second_derivative: bool = True) -> Dict:
    """
    Find peaks in a spectrum with optional second derivative enhancement.

    The second derivative method is inspired by the ALIAS methodology
    (Noël et al., 2025) and enhances peaks while filtering slow continuum
    variations, providing better peak localization.

    Parameters
    ----------
    wavelengths : array
        Wavelength array in nm
    spectrum : array
        Intensity array
    height_fraction : float
        Minimum peak height as fraction of max intensity
    prominence_fraction : float
        Minimum peak prominence as fraction of max intensity
    min_distance_nm : float
        Minimum distance between peaks in nm
    noise_threshold_sigma : float
        Minimum peak height as multiple of baseline noise standard deviation.
        Peaks below noise_threshold_sigma * noise_std are rejected. Default 5.0.
    use_second_derivative : bool
        If True, use second derivative enhancement for peak detection.
        If False, use direct peak finding on the spectrum. Default True.

    Returns
    -------
    dict with:
        - wavelengths: peak positions in nm
        - intensities: peak intensities (from original spectrum)

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.linspace(300, 400, 1000)
    >>> spectrum = (50 * np.exp(-0.5 * ((wl - 350) / 0.1) ** 2)
    ...           + 30 * np.exp(-0.5 * ((wl - 370) / 0.1) ** 2))
    >>> peaks = find_spectrum_peaks(wl, spectrum)
    >>> len(peaks['wavelengths'])  # Should find 2 peaks
    2
    """
    # Estimate baseline noise
    noise_std = estimate_baseline_noise(spectrum)
    noise_threshold = noise_threshold_sigma * noise_std

    # Convert nm distance to index distance
    dw = np.median(np.diff(wavelengths))
    min_distance_idx = max(1, int(min_distance_nm / dw))

    if use_second_derivative:
        # Second derivative method (ALIAS approach)
        # Use Savitzky-Golay filter for smooth second derivative (reduces noise amplification)
        # Window length should be odd and >= 5 for 2nd derivative
        window_length = max(5, min(51, len(spectrum) // 20))
        if window_length % 2 == 0:
            window_length += 1

        # Compute second derivative with smoothing (deriv=2)
        second_deriv = savgol_filter(spectrum, window_length, polyorder=3, deriv=2, delta=dw)

        # Negative second derivative highlights peaks (peaks have negative curvature)
        # Clip to positive values
        enhanced = np.maximum(-second_deriv, 0)

        # Normalize for threshold comparison
        enhanced_max = np.max(enhanced)
        if enhanced_max > 0:
            enhanced_norm = enhanced / enhanced_max
        else:
            enhanced_norm = enhanced

        # Find peaks in the enhanced signal
        peaks_idx, properties = find_peaks(
            enhanced_norm,
            height=height_fraction,
            prominence=prominence_fraction,
            distance=min_distance_idx
        )
    else:
        # Direct peak finding on normalized spectrum
        spec_norm = spectrum / np.max(spectrum)

        peaks_idx, properties = find_peaks(
            spec_norm,
            height=height_fraction,
            prominence=prominence_fraction,
            distance=min_distance_idx
        )

    # Filter peaks by noise threshold (applied to original spectrum)
    above_noise = spectrum[peaks_idx] > noise_threshold
    peaks_idx = peaks_idx[above_noise]

    return {
        'wavelengths': wavelengths[peaks_idx],
        'intensities': spectrum[peaks_idx]
    }


def fit_all_peaks(wavelengths: np.ndarray, spectrum: np.ndarray,
                  peak_wavelengths: np.ndarray,
                  fit_window: float = 0.3) -> pd.DataFrame:
    """
    Fit Voigt profiles to all detected peaks.

    Parameters
    ----------
    wavelengths : array
        Wavelength array in nm
    spectrum : array
        Intensity array
    peak_wavelengths : array
        Detected peak positions in nm (from find_spectrum_peaks)
    fit_window : float
        Half-width of fitting window in nm

    Returns
    -------
    DataFrame with columns:
        - peak_wl: original peak position
        - center: fitted center wavelength
        - amplitude: fitted amplitude (integrated intensity)
        - sigma: Gaussian width
        - gamma: Lorentzian HWHM
        - fwhm: total FWHM (approximate)
        - success: whether fit succeeded
    """
    results = []

    for peak_wl in peak_wavelengths:
        fit_result = fit_voigt_profile(wavelengths, spectrum, peak_wl, window=fit_window)

        if fit_result['success']:
            # Approximate total FWHM using the Olivero-Longbottom formula
            fg = fit_result['fwhm_gauss']
            fl = fit_result['fwhm_lorentz']
            fwhm = 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)

            results.append({
                'peak_wl': peak_wl,
                'center': fit_result['center'],
                'amplitude': fit_result['amplitude'],
                'sigma': fit_result['sigma'],
                'gamma': fit_result['gamma'],
                'fwhm': fwhm,
                'fwhm_gauss': fg,
                'fwhm_lorentz': fl,
                'residual_rel': fit_result['residual_rel'],
                'success': True
            })
        else:
            results.append({
                'peak_wl': peak_wl,
                'center': np.nan,
                'amplitude': np.nan,
                'sigma': np.nan,
                'gamma': np.nan,
                'fwhm': np.nan,
                'fwhm_gauss': np.nan,
                'fwhm_lorentz': np.nan,
                'residual_rel': np.nan,
                'success': False
            })

    return pd.DataFrame(results)


def merge_duplicate_peaks(fitted_peaks: pd.DataFrame,
                          center_tol: float = 0.05) -> pd.DataFrame:
    """
    Merge fitted peaks whose Voigt centers converge to the same position.

    Broad emission lines with noise on top can produce multiple detected
    peaks that all fit to the same center.  This function groups successful
    fits whose centers are within *center_tol* nm and keeps only the one
    with the lowest relative residual.

    Parameters
    ----------
    fitted_peaks : DataFrame
        Output from ``fit_all_peaks()``.
    center_tol : float
        Maximum distance (nm) between fitted centers to consider them the
        same line.  Default 0.1 nm.

    Returns
    -------
    DataFrame with duplicate rows removed (unsuccessful fits are also
    dropped).
    """
    df = fitted_peaks.copy()
    valid = df[df['success']].sort_values('center').reset_index(drop=True)

    if len(valid) == 0:
        return df.iloc[0:0]  # empty with same columns

    # Greedy clustering: walk sorted centres, group within tolerance
    keep_idx = []
    cluster_start = 0
    for i in range(1, len(valid)):
        if valid.loc[i, 'center'] - valid.loc[cluster_start, 'center'] > center_tol:
            # End of cluster – pick best residual
            cluster = valid.iloc[cluster_start:i]
            best = cluster['residual_rel'].idxmin()
            keep_idx.append(best)
            cluster_start = i
    # last cluster
    cluster = valid.iloc[cluster_start:]
    best = cluster['residual_rel'].idxmin()
    keep_idx.append(best)

    return valid.loc[keep_idx].reset_index(drop=True)


def filter_by_fit_quality(fitted_peaks: pd.DataFrame,
                          max_residual: float = 0.1) -> pd.DataFrame:
    """
    Filter fitted peaks by fit quality.

    Parameters
    ----------
    fitted_peaks : DataFrame
        Output from fit_all_peaks()
    max_residual : float
        Maximum allowed relative residual (default 0.1 = 10%)

    Returns
    -------
    DataFrame with added 'good_fit' column (boolean)
    """
    result = fitted_peaks.copy()
    result['good_fit'] = (
        result['success'] &
        (result['residual_rel'] <= max_residual)
    )
    return result


def find_isolated_peaks(fitted_peaks: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which peaks are isolated based on their FWHM.

    A peak is isolated if no other peak center is within 1 FWHM of its center.

    Parameters
    ----------
    fitted_peaks : DataFrame
        Output from fit_all_peaks()

    Returns
    -------
    DataFrame with added 'isolated' column (boolean)
    """
    result = fitted_peaks.copy()
    result['isolated'] = False

    # Only consider successful fits
    valid = result[result['success']].copy()

    if len(valid) == 0:
        return result

    centers = valid['center'].values
    fwhms = valid['fwhm'].values

    for i, row in valid.iterrows():
        center = row['center']
        fwhm = row['fwhm']

        # Check distance to all other peaks
        distances = np.abs(centers - center)
        # Exclude self (distance = 0)
        other_mask = distances > 0.001

        if np.sum(other_mask) == 0:
            # Only one peak, it's isolated
            result.loc[i, 'isolated'] = True
        else:
            # Check if nearest neighbor is beyond 1 FWHM
            min_distance = np.min(distances[other_mask])
            result.loc[i, 'isolated'] = min_distance > fwhm

    return result


def find_candidate_lines(peak_wavelengths: np.ndarray,
                         intensities: np.ndarray,
                         lines_db: pd.DataFrame,
                         wl_tolerance: float = 0.1,
                         min_aki: float = 1e4) -> List[Dict]:
    """
    Find candidate emission lines for each detected peak.

    This function only associates peaks with possible line candidates,
    without making any attribution decisions. Use attribute_peaks_to_lines()
    to select the best match for each peak.

    Parameters
    ----------
    peak_wavelengths : array
        Detected peak positions in nm
    intensities : array
        Line intensities (should be integrated intensities for Boltzmann analysis)
    lines_db : DataFrame
        Line database with columns: element, sp_num, ritz_wl_vac(nm),
        Aki(s^-1), Ei(eV), Ek(eV), g_i, g_k
    wl_tolerance : float
        Maximum wavelength difference for matching (nm)
    min_aki : float
        Minimum transition probability to consider

    Returns
    -------
    List of dicts, each containing:
        - peak_wl: detected peak wavelength
        - intensity: line intensity (as provided)
        - candidates: DataFrame of all candidate lines within tolerance
    """
    # Pre-filter database
    wl_min, wl_max = peak_wavelengths.min() - 1, peak_wavelengths.max() + 1
    valid_lines = lines_db[
        (lines_db['ritz_wl_vac(nm)'] >= wl_min) &
        (lines_db['ritz_wl_vac(nm)'] <= wl_max) &
        (lines_db['Aki(s^-1)'] >= min_aki)
    ].copy()

    # Vectorized wavelength array for fast searching
    db_wl = valid_lines['ritz_wl_vac(nm)'].values

    results = []

    for peak_wl, intensity in zip(peak_wavelengths, intensities):
        # Find all lines within tolerance
        wl_diff = np.abs(db_wl - peak_wl)
        mask = wl_diff < wl_tolerance

        if not np.any(mask):
            results.append({
                'peak_wl': peak_wl,
                'intensity': intensity,
                'candidates': pd.DataFrame()
            })
            continue

        candidates = valid_lines[mask].copy()
        candidates['wl_diff'] = np.abs(candidates['ritz_wl_vac(nm)'] - peak_wl)

        results.append({
            'peak_wl': peak_wl,
            'intensity': intensity,
            'candidates': candidates
        })

    return results


def attribute_peaks_to_lines(peak_candidates: List[Dict],
                             elements: Optional[List[str]] = None,
                             wl_penalty_sigma: Optional[float] = None) -> List[Dict]:
    """
    Attribute each peak to exactly one emission line.

    Algorithm:
    For each candidate, calculate a score = Aki * exp(-Ei) which combines
    transition probability (higher is better) with lower level energy
    (lower is better, as transitions from lower states are typically stronger).
    The candidate with the highest score is selected.

    An optional wavelength penalty can penalize candidates that are far from
    the peak center: score *= exp(-|Δλ| / sigma). This helps resolve ambiguity
    in crowded spectral regions where multiple elements have strong lines
    within the wavelength tolerance.

    Parameters
    ----------
    peak_candidates : list
        Output from find_candidate_lines()
    elements : list of str, optional
        If provided, only consider lines from these elements during attribution.
        If None, consider all elements in the candidates.
    wl_penalty_sigma : float, optional
        If provided, multiply the score by exp(-|Δλ| / sigma) where Δλ is the
        wavelength difference between the candidate line and the detected peak.
        Typical values are 0.01–0.05 nm. If None, no wavelength penalty is
        applied (backward-compatible default).

    Returns
    -------
    List of dicts, each containing:
        - peak_wl: detected peak wavelength
        - intensity: line intensity (as provided to find_candidate_lines)
        - candidates: list of top candidate dicts (sorted by score)
        - best_match: attributed line (dict) or None
    """
    results = []

    for peak_data in peak_candidates:
        peak_wl = peak_data['peak_wl']
        intensity = peak_data['intensity']
        candidates = peak_data['candidates']

        # Filter by elements if specified
        if elements is not None and len(candidates) > 0:
            candidates = candidates[candidates['element'].isin(elements)]

        if len(candidates) == 0:
            results.append({
                'peak_wl': peak_wl,
                'intensity': intensity,
                'candidates': [],
                'best_match': None
            })
            continue

        # Calculate score: Aki * exp(-Ei)
        candidates = candidates.copy()
        candidates['score'] = candidates['Aki(s^-1)'] * np.exp(-candidates['Ei(eV)'])

        # Apply wavelength proximity penalty if requested
        if wl_penalty_sigma is not None and wl_penalty_sigma > 0:
            candidates['score'] *= np.exp(-candidates['wl_diff'] / wl_penalty_sigma)

        # Sort by score (descending) and select the best
        candidates_sorted = candidates.sort_values('score', ascending=False)
        best = candidates_sorted.iloc[0]

        results.append({
            'peak_wl': peak_wl,
            'intensity': intensity,
            'candidates': candidates_sorted.head(5).to_dict('records'),
            'best_match': best.to_dict()
        })

    return results


def filter_attributed_by_element_count(attributed: List[Dict],
                                       peak_candidates: List[Dict],
                                       min_count: int = 3,
                                       max_iterations: int = 10,
                                       elements: Optional[List[str]] = None,
                                       wl_penalty_sigma: Optional[float] = None) -> List[Dict]:
    """
    Iteratively filter attributed peaks by requiring a minimum number of
    peaks per element.

    Elements with fewer attributed peaks than min_count are removed, and
    attribution is re-run with the reduced element set. This process is
    repeated until convergence (the element set stabilizes).

    This addresses the "one-hit-wonder" problem where rare/unlikely elements
    win attribution for a few peaks by chance due to having strong spectroscopic
    parameters, but lack the multi-line consistency expected from truly present
    elements.

    Parameters
    ----------
    attributed : list
        Output from attribute_peaks_to_lines()
    peak_candidates : list
        Output from find_candidate_lines() (needed for re-attribution)
    min_count : int
        Minimum number of attributed peaks required for an element (default: 3)
    max_iterations : int
        Maximum number of iterations (default: 10)
    elements : list of str, optional
        Initial element restriction (passed to attribute_peaks_to_lines)
    wl_penalty_sigma : float, optional
        Wavelength penalty parameter (passed to attribute_peaks_to_lines)

    Returns
    -------
    List of dicts with the same structure as attribute_peaks_to_lines output,
    but with elements that don't meet the min_count threshold removed.
    """
    current = attributed
    allowed_elements = set(elements) if elements is not None else None

    for _ in range(max_iterations):
        # Count peaks per element
        elem_counts: Dict[str, int] = {}
        for r in current:
            if r['best_match'] is not None:
                el = r['best_match']['element']
                elem_counts[el] = elem_counts.get(el, 0) + 1

        # Filter elements with enough counts
        new_allowed = {el for el, cnt in elem_counts.items() if cnt >= min_count}

        if allowed_elements is not None:
            new_allowed = new_allowed & allowed_elements

        if new_allowed == allowed_elements:
            break  # converged

        allowed_elements = new_allowed

        # Re-attribute with restricted element set
        current = attribute_peaks_to_lines(
            peak_candidates,
            elements=list(allowed_elements) if allowed_elements else None,
            wl_penalty_sigma=wl_penalty_sigma
        )

    return current


def find_peaks_and_identify(wavelengths: np.ndarray, spectrum: np.ndarray,
                            lines_db: pd.DataFrame,
                            height_fraction: float = 0.01,
                            wl_tolerance: float = 0.1,
                            min_aki: float = 1e4,
                            elements: Optional[List[str]] = None) -> Dict:
    """
    Combined peak detection and line identification.

    Uses a two-step process:
    1. find_candidate_lines() - finds all possible line matches for each peak
    2. attribute_peaks_to_lines() - selects the best match by ranking candidates
       using score = Aki * exp(-Ei), which favors high transition probability
       and low excitation energy.

    Parameters
    ----------
    wavelengths : array
        Wavelength array in nm
    spectrum : array
        Intensity array
    lines_db : DataFrame
        Line database
    height_fraction : float
        Minimum peak height as fraction of max
    wl_tolerance : float
        Maximum wavelength tolerance for line matching (nm)
    min_aki : float
        Minimum transition probability
    elements : list of str, optional
        If provided, only consider lines from these elements during attribution.
        If None, consider all elements in the database.

    Returns
    -------
    dict with:
        - peaks: peak detection results
        - candidates: raw candidate associations (from find_candidate_lines)
        - identifications: attributed lines (from attribute_peaks_to_lines)
        - element_counts: Counter of identified elements
        - summary: DataFrame of identified lines
    """
    # Find peaks
    peaks = find_spectrum_peaks(wavelengths, spectrum,
                                height_fraction=height_fraction)

    if len(peaks['wavelengths']) == 0:
        return {
            'peaks': peaks,
            'candidates': [],
            'identifications': [],
            'element_counts': Counter(),
            'summary': pd.DataFrame()
        }

    # Step 1: Find candidate lines for each peak
    candidates = find_candidate_lines(
        peaks['wavelengths'], peaks['intensities'],
        lines_db, wl_tolerance=wl_tolerance, min_aki=min_aki
    )

    # Step 2: Attribute peaks to lines
    identifications = attribute_peaks_to_lines(candidates, elements=elements)

    # Count elements
    element_counts = Counter()
    for ident in identifications:
        if ident['best_match'] is not None:
            element_counts[ident['best_match']['element']] += 1

    # Create summary DataFrame
    rows = []
    for ident in identifications:
        if ident['best_match'] is None:
            continue
        m = ident['best_match']
        rows.append({
            'peak_wl': ident['peak_wl'],
            'intensity': ident['intensity'],
            'element': m['element'],
            'sp_num': m['sp_num'],
            'db_wl': m['ritz_wl_vac(nm)'],
            'wl_diff': m['wl_diff'],
            'Aki': m['Aki(s^-1)'],
            'Ei_eV': m['Ei(eV)'],
            'Ek_eV': m['Ek(eV)'],
            'g_i': m['g_i'],
            'g_k': m['g_k']
        })

    summary = pd.DataFrame(rows)

    return {
        'peaks': peaks,
        'candidates': candidates,
        'identifications': identifications,
        'element_counts': element_counts,
        'summary': summary
    }


# =============================================================================
# ISOLATED LINES FOR BOLTZMANN ANALYSIS
# =============================================================================

def _dedup_fine_structure(df: pd.DataFrame,
                          dedup_tol_nm: float = 0.01) -> pd.DataFrame:
    """
    Deduplicate fine-structure components within a single element/ion group.

    Multiple database rows can exist at nearly the same wavelength (within
    *dedup_tol_nm*) for the same element and ionization state.  These are
    typically fine-structure components or forbidden/allowed variants of
    the same transition. Keeping all of them biases Boltzmann plots because
    the weak forbidden component produces an absurdly low y-value on the
    same x-value as the dominant allowed line.

    Within each cluster the row with the largest ``Aki(s^-1)`` is retained.

    Parameters
    ----------
    df : DataFrame
        Lines for ONE element (may contain both sp_num 1 and 2).
        Must have columns ``ritz_wl_vac(nm)``, ``Aki(s^-1)``, ``sp_num``.
    dedup_tol_nm : float
        Wavelength tolerance in nm for considering two rows as the same
        physical line (default 0.01 nm).

    Returns
    -------
    DataFrame
        Deduplicated copy of *df*.
    """
    if len(df) == 0 or dedup_tol_nm <= 0:
        return df

    keep_idx = []
    for sp in df['sp_num'].unique():
        sub = df[df['sp_num'] == sp].sort_values('ritz_wl_vac(nm)')
        wl = sub['ritz_wl_vac(nm)'].values
        aki = sub['Aki(s^-1)'].values
        idx = sub.index.values

        used = np.zeros(len(wl), dtype=bool)
        for i in range(len(wl)):
            if used[i]:
                continue
            # Collect cluster: all rows within dedup_tol_nm of this one
            cluster = [i]
            for j in range(i + 1, len(wl)):
                if wl[j] - wl[i] > dedup_tol_nm:
                    break
                cluster.append(j)
                used[j] = True
            # Keep the row with highest Aki in the cluster
            best = cluster[int(np.argmax(aki[cluster]))]
            keep_idx.append(idx[best])

    return df.loc[keep_idx]


def find_isolated_lines(lines_db: pd.DataFrame,
                        elements: List[str],
                        wl_range: Tuple[float, float],
                        min_separation: float = 0.5,
                        min_aki: float = 1e5,
                        dedup_tol_nm: float = 0.01) -> Dict[str, pd.DataFrame]:
    """
    Find well-isolated emission lines from the database.

    A line is considered isolated if it is well-separated (by min_separation)
    from lines of OTHER elements in the database.

    Fine-structure components (multiple database rows at nearly the same
    wavelength for the same element/ion) are deduplicated by keeping only
    the row with the highest ``Aki(s^-1)`` within *dedup_tol_nm*.

    This function does NOT require spectrum data - it only analyzes the
    database to find lines that are spectrally isolated. Use
    measure_line_intensities() to get intensities from a spectrum.

    Parameters
    ----------
    lines_db : DataFrame
        Line database with columns: element, sp_num, ritz_wl_vac(nm),
        Aki(s^-1), Ei(eV), Ek(eV), g_i, g_k
    elements : list
        Elements to find isolated lines for
    wl_range : tuple
        (min_wavelength, max_wavelength) in nm
    min_separation : float
        Minimum separation (nm) from lines of OTHER elements
    min_aki : float
        Minimum transition probability to consider
    dedup_tol_nm : float
        Wavelength tolerance (nm) for fine-structure deduplication.
        Lines of the same element/ion within this tolerance are merged,
        keeping only the strongest (highest Aki). Set to 0 to disable.
        Default 0.01 nm.

    Returns
    -------
    dict mapping element names to DataFrames of isolated lines
    """
    wl_min, wl_max = wl_range

    # Get all lines within wavelength range with sufficient Aki
    all_lines = lines_db[
        (lines_db['ritz_wl_vac(nm)'] > wl_min) &
        (lines_db['ritz_wl_vac(nm)'] < wl_max) &
        (lines_db['Aki(s^-1)'] >= min_aki)
    ].copy()

    if len(all_lines) == 0:
        return {el: pd.DataFrame() for el in elements}

    # Get target element lines
    target_lines = all_lines[all_lines['element'].isin(elements)].copy()

    if len(target_lines) == 0:
        return {el: pd.DataFrame() for el in elements}

    # Build arrays for vectorized separation check
    all_wl = all_lines['ritz_wl_vac(nm)'].values
    all_el = all_lines['element'].values

    # Process each target line
    isolated_results = []

    for _, row in target_lines.iterrows():
        wl = row['ritz_wl_vac(nm)']
        element = row['element']

        # Check separation from OTHER elements
        distances = np.abs(all_wl - wl)
        nearby_mask = (distances < min_separation) & (distances > 0.001)  # exclude self
        nearby_elements = all_el[nearby_mask]
        n_other_elements = np.sum(nearby_elements != element)

        if n_other_elements == 0:
            # Line is isolated from other elements
            result_row = row.to_dict()
            # Count same-element lines nearby (multiplet info)
            result_row['n_multiplet'] = np.sum(nearby_elements == element)
            isolated_results.append(result_row)

    # Organize by element
    result = {el: [] for el in elements}
    for row in isolated_results:
        result[row['element']].append(row)

    # Convert to DataFrames, deduplicate fine-structure, and sort by Aki
    for el in elements:
        if len(result[el]) > 0:
            df = pd.DataFrame(result[el])
            if dedup_tol_nm > 0:
                df = _dedup_fine_structure(df, dedup_tol_nm)
            df = df.sort_values('Aki(s^-1)', ascending=False)
            result[el] = df
        else:
            result[el] = pd.DataFrame()

    return result


def measure_line_intensities(wavelengths: np.ndarray,
                             spectrum: np.ndarray,
                             lines_df: pd.DataFrame,
                             fit_window: float = 0.3) -> pd.DataFrame:
    """
    Measure integrated line intensities by fitting Voigt profiles.

    For each line in lines_df, fits a Voigt profile and integrates it
    to get the line intensity (area under the fitted profile).

    Parameters
    ----------
    wavelengths : array
        Wavelength array in nm
    spectrum : array
        Intensity/irradiance array
    lines_df : DataFrame
        DataFrame with line information, must have 'ritz_wl_vac(nm)' column.
        Typically output from find_isolated_lines().
    fit_window : float
        Half-width of fitting window in nm (default 0.3)

    Returns
    -------
    DataFrame
        Copy of input DataFrame with added 'intensity' column
    """
    if len(lines_df) == 0:
        return lines_df.copy()

    result = lines_df.copy()
    intensities = []

    for _, row in result.iterrows():
        wl = row['ritz_wl_vac(nm)']

        # Fit Voigt profile
        fit_result = fit_voigt_profile(wavelengths, spectrum, wl, window=fit_window)

        if fit_result['success']:
            # Intensity is the amplitude parameter (area under Voigt profile)
            intensities.append(fit_result['amplitude'])
        else:
            intensities.append(np.nan)

    result['intensity'] = intensities
    return result


# =============================================================================
# ELECTRON DENSITY FROM STARK BROADENING
# =============================================================================

def calculate_stark_width(temperature: float, electron_density: float,
                          a0: float, a1: float, a2: float,
                          n_e_ref: float) -> float:
    """
    Calculate Stark FWHM using Stark-B parameterization.

    Formula from Stark-B database (Sahal-Bréchot et al.):
        w = 10^(a0 + a1*log10(T) + a2*(log10(T))^2) / 10  [nm, from Angstrom]
        FWHM = w * (n_e / n_e_ref) * 1e-6

    Where:
        - n_e is in m^-3
        - n_e_ref is in cm^-3 (from database)
        - The factor 1e-6 handles the unit conversion

    Parameters
    ----------
    temperature : float
        Plasma temperature in K
    electron_density : float
        Electron density in m^-3
    a0, a1, a2 : float
        Stark-B polynomial coefficients
    n_e_ref : float
        Reference electron density in cm^-3

    Returns
    -------
    float
        Stark FWHM in nm
    """
    log_T = np.log10(temperature)
    w = 10**(a0 + a1 * log_T + a2 * log_T**2) / 10.0  # Angstrom to nm
    fwhm = w * (electron_density / n_e_ref) * 1e-6
    return fwhm


def estimate_electron_density(fwhm_lorentz: float, temperature: float,
                              a0: float, a1: float, a2: float,
                              n_e_ref: float) -> float:
    """
    Estimate electron density from measured Stark (Lorentzian) FWHM.

    Inverts the Stark-B formula:
        n_e = FWHM * n_e_ref * 1e6 / w

    Parameters
    ----------
    fwhm_lorentz : float
        Measured Lorentzian FWHM in nm
    temperature : float
        Plasma temperature in K
    a0, a1, a2 : float
        Stark-B polynomial coefficients
    n_e_ref : float
        Reference electron density in cm^-3

    Returns
    -------
    float
        Electron density in m^-3
    """
    log_T = np.log10(temperature)
    w = 10**(a0 + a1 * log_T + a2 * log_T**2) / 10.0  # Angstrom to nm

    if w <= 0 or fwhm_lorentz <= 0:
        return np.nan

    # n_e in m^-3 = FWHM_nm * n_e_ref_cm3 * 1e6 / w_nm
    n_e = fwhm_lorentz * n_e_ref * 1e6 / w

    return n_e


def estimate_ne_from_line(wavelengths: np.ndarray, spectrum: np.ndarray,
                          line_wl: float, line_data: Dict,
                          temperature: float,
                          instrumental_gamma: float = 0.0) -> Dict:
    """
    Estimate electron density from a single line by Voigt fitting.

    Parameters
    ----------
    wavelengths : array
        Wavelength array
    spectrum : array
        Intensity array
    line_wl : float
        Line wavelength in nm
    line_data : dict
        Line data including Stark parameters. Required keys:
        - 'electron a0', 'electron a1', 'electron a2': Stark-B polynomial coefficients
        - 'N (cm-3)': Reference electron density in cm^-3
    temperature : float
        Plasma temperature in K
    instrumental_gamma : float
        Instrumental Lorentzian HWHM in nm (subtracted from fit)

    Returns
    -------
    dict with fit results and ne estimate
    """
    # Fit Voigt profile
    fit = fit_voigt_profile(wavelengths, spectrum, line_wl)

    if not fit['success']:
        return {'success': False, 'error': fit.get('error', 'Fit failed')}

    # Extract Stark parameters - require them to be present
    a0 = line_data.get('electron a0', line_data.get('a0'))
    a1 = line_data.get('electron a1', line_data.get('a1'))
    a2 = line_data.get('electron a2', line_data.get('a2'))
    n_e_ref = line_data.get('N (cm-3)', line_data.get('n_e_ref'))

    if a0 is None or a1 is None or a2 is None or n_e_ref is None:
        return {'success': False, 'error': 'Missing Stark parameters in line_data'}

    a0 = float(a0)
    a1 = float(a1)
    a2 = float(a2)
    n_e_ref = float(n_e_ref)

    # Correct for instrumental broadening (Lorentzian HWHM adds linearly)
    gamma_stark = fit['gamma'] - instrumental_gamma
    if gamma_stark <= 0:
        return {'success': False, 'error': 'Gamma too small after instrumental correction'}

    fwhm_stark = 2 * gamma_stark

    # Calculate electron density
    n_e = estimate_electron_density(fwhm_stark, temperature, a0, a1, a2, n_e_ref)

    return {
        'success': True,
        'line_wl': line_wl,
        'gamma_fit': fit['gamma'],
        'gamma_stark': gamma_stark,
        'fwhm_stark': fwhm_stark,
        'sigma': fit['sigma'],
        'ne': n_e,
        'w_parameter': 10**(a0 + a1*np.log10(temperature) + a2*np.log10(temperature)**2) / 10,
        'fit': fit
    }


def estimate_ne_from_lines(wavelengths: np.ndarray, spectrum: np.ndarray,
                           lines_df: pd.DataFrame, temperature: float,
                           instrumental_gamma: float = 0.0) -> Dict:
    """
    Estimate electron density from multiple lines by Voigt fitting.

    Returns individual ne estimates for each line. Statistical analysis
    (mean, median, outlier removal) should be done by the caller.

    Parameters
    ----------
    wavelengths : array
        Wavelength array
    spectrum : array
        Intensity array
    lines_df : DataFrame
        Lines with Stark parameters. Required columns:
        'ritz_wl_vac(nm)', 'electron a0', 'electron a1', 'electron a2', 'N (cm-3)'
    temperature : float
        Plasma temperature in K
    instrumental_gamma : float
        Instrumental Lorentzian HWHM in nm

    Returns
    -------
    dict with:
        - 'fits': list of individual fit results, each containing 'line_wl', 'ne',
          'element', 'sp_num', and fit details
        - 'success': True if at least one fit succeeded
        - 'error': error message if no fits succeeded
    """
    # Check for required columns
    required = ['electron a0', 'electron a1', 'electron a2', 'N (cm-3)', 'ritz_wl_vac(nm)']
    if not all(col in lines_df.columns for col in required):
        missing = [col for col in required if col not in lines_df.columns]
        return {
            'success': False,
            'error': f'Missing required columns: {missing}',
            'fits': []
        }

    fits = []

    for _, row in lines_df.iterrows():
        result = estimate_ne_from_line(
            wavelengths, spectrum,
            row['ritz_wl_vac(nm)'], row.to_dict(),
            temperature, instrumental_gamma
        )

        if result['success'] and np.isfinite(result['ne']) and result['ne'] > 0:
            result['element'] = row.get('element', '')
            result['sp_num'] = row.get('sp_num', 0)
            result['db_wl'] = row['ritz_wl_vac(nm)']
            fits.append(result)

    if len(fits) == 0:
        return {
            'success': False,
            'error': 'No successful fits',
            'fits': []
        }

    return {
        'success': True,
        'fits': fits
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_identified_spectrum(wavelengths: np.ndarray, spectrum: np.ndarray,
                             identifications: List[Dict],
                             elements_to_show: Optional[List[str]] = None,
                             wl_range: Optional[Tuple[float, float]] = None,
                             figsize: Tuple[int, int] = (14, 6),
                             ax=None):
    """
    Plot spectrum with annotated emission line identifications.

    Parameters
    ----------
    wavelengths : array
        Wavelength array
    spectrum : array
        Intensity array
    identifications : list
        Output from identify_emission_lines
    elements_to_show : list, optional
        Only annotate these elements (None = all)
    wl_range : tuple, optional
        (wl_min, wl_max) to display
    figsize : tuple
        Figure size
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib figure and axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Normalize spectrum
    spec_norm = spectrum / np.max(spectrum)

    # Apply wavelength range
    if wl_range is not None:
        mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
        wl_plot = wavelengths[mask]
        spec_plot = spec_norm[mask]
    else:
        wl_plot = wavelengths
        spec_plot = spec_norm

    # Plot spectrum
    ax.plot(wl_plot, spec_plot, 'k-', lw=0.5, alpha=0.8)

    # Color map for elements
    unique_elements = set()
    for ident in identifications:
        if ident['best_match'] is not None:
            unique_elements.add(ident['best_match']['element'])

    if elements_to_show is not None:
        unique_elements = unique_elements & set(elements_to_show)

    colors = plt.cm.tab10.colors
    element_colors = {el: colors[i % len(colors)]
                      for i, el in enumerate(sorted(unique_elements))}

    # Annotate peaks
    annotated_positions = []  # Track to avoid overlap

    for ident in identifications:
        if ident['best_match'] is None:
            continue

        match = ident['best_match']
        element = match['element']

        if elements_to_show is not None and element not in elements_to_show:
            continue

        peak_wl = ident['peak_wl']

        # Skip if outside range
        if wl_range is not None and (peak_wl < wl_range[0] or peak_wl > wl_range[1]):
            continue

        # Get intensity at peak
        idx = np.argmin(np.abs(wavelengths - peak_wl))
        intensity = spec_norm[idx]

        color = element_colors.get(element, 'gray')
        sp_str = 'I' if match['sp_num'] == 1 else 'II'

        # Vertical line
        ax.axvline(peak_wl, color=color, alpha=0.3, lw=0.5)

        # Determine annotation position to avoid overlap
        y_offset = 0.02
        for pos_wl, pos_y in annotated_positions:
            if abs(peak_wl - pos_wl) < 3:  # Within 3 nm
                y_offset += 0.05

        y_pos = min(intensity + y_offset, 1.0)
        annotated_positions.append((peak_wl, y_pos))

        # Annotation
        ax.annotate(
            f'{element} {sp_str}',
            xy=(peak_wl, intensity),
            xytext=(peak_wl, y_pos),
            fontsize=7,
            color=color,
            ha='center',
            rotation=90,
            arrowprops=dict(arrowstyle='-', color=color, alpha=0.3, lw=0.5)
        )

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_ylim(-0.02, 1.1)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [plt.Line2D([0], [0], color=c, label=el)
                       for el, c in element_colors.items()]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    return fig, ax
