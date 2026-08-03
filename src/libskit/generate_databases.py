#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Database Generation Utilities

Functions for processing and merging atomic spectroscopy databases:
- Parsing Stark-B broadening parameter files
- Loading and normalising NIST ASD emission line data
- Matching Stark broadening data to emission lines
- Generating the extended lines database used by libskit
"""
from __future__ import annotations

from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Tuple

import pandas as pd


# NIST accuracy codes mapped to numeric uncertainty fractions (upper bounds).
# See https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html#OUTACC
ACCURACY_MAP: dict[str, float] = {
    "AAA": 0.003,
    "AA": 0.01,
    "A+": 0.02,
    "A": 0.03,
    "B+": 0.07,
    "B": 0.10,
    "C+": 0.18,
    "C": 0.25,
    "D+": 0.40,
    "D": 0.50,
    "E": 0.50,
}


def acc_code_to_fraction(code: str | float) -> float:
    """Convert a NIST accuracy code string to a numeric uncertainty fraction.

    Trailing primes (e.g. ``"A'"``) are stripped before lookup.
    Missing or unrecognised codes are mapped to 0.50 (most conservative).
    """
    if not isinstance(code, str) or not code.strip():
        return 0.50
    code = code.strip().rstrip("'")
    return ACCURACY_MAP.get(code, 0.50)


def get_stark_db_from_file(file_path: str, element_string: str) -> Tuple[pd.DataFrame, int]:
    """
    Load Stark broadening parameters from a CSV file.

    Reads and processes Stark-B database files, handling format inconsistencies
    and normalizing notation to match NIST conventions.

    Parameters
    ----------
    file_path : str
        Directory path containing the Stark database CSV files.
    element_string : str
        Ion identifier string (e.g., "Fe_I", "Ca_II").

    Returns
    -------
    tuple
        - pd.DataFrame: DataFrame with processed Stark parameters including:
            - conf_i, term_i, J_i: Lower level quantum numbers
            - conf_k, term_k, J_k: Upper level quantum numbers
            - electron a0, a1, a2: Width polynomial coefficients
            - electron b0, b1, b2: Shift polynomial coefficients
            - N (cm-3): Reference electron density
        - int: Total number of unique Stark records loaded.

    Notes
    -----
    The function performs several normalization steps:
    - Strips whitespace from column names and values
    - Corrects missing dots in configuration strings (e.g., "3d6(5D)4s" -> "3d6.(5D).4s")
    - Replaces 'o' with '*' to match NIST notation for odd parity
    - Handles "/2" notation in parameter values
    - Removes duplicate entries
    """

    # load the stark data from the .csv file
    stark = pd.read_csv(file_path + element_string + ".csv", comment="#", sep=";")

    # strip empty spaces to the left and right from every column name
    stark.columns = stark.columns.str.strip()

    # remove space in front and end of string
    stark["Lower Level"] = stark["Lower Level"].str.strip()
    stark["Upper Level"] = stark["Upper Level"].str.strip()

    # sometimes in the stark data, there are terms where there is a point missing
    # example: 3d6(5D).4s should be 3d6.(5D).4s
    def correct_level_conf_stark(x):
        # find every open bracket
        bracket_positions = [pos for pos, char in enumerate(x) if char == "("]

        # check if the char before is a dot
        for bracket_position in bracket_positions:
            # if not --> insert it
            if x[bracket_position - 1] != ".":
                x = x[:bracket_position] + "." + x[bracket_position:]
        return x

    stark["Lower Level"] = stark["Lower Level"].transform(correct_level_conf_stark)
    stark["Upper Level"] = stark["Upper Level"].transform(correct_level_conf_stark)

    # split 'Level' from stark-b to conf, term and J as in NIST
    # sometimes in stark-b they include the Js, sometimes they don't
    if len(stark["Lower Level"].str.split(" ", expand=True).columns) == 2:
        stark[["conf_i", "term_i"]] = stark["Lower Level"].str.split(" ", expand=True)
        stark["J_i"] = None
    elif len(stark["Lower Level"].str.split(" ", expand=True).columns) == 3:
        stark[["conf_i", "term_i", "J_i"]] = stark["Lower Level"].str.split(
            " ", expand=True
        )

    if len(stark["Upper Level"].str.split(" ", expand=True).columns) == 2:
        stark[["conf_k", "term_k"]] = stark["Upper Level"].str.split(" ", expand=True)
        stark["J_k"] = None
    elif len(stark["Upper Level"].str.split(" ", expand=True).columns) == 3:
        stark[["conf_k", "term_k", "J_k"]] = stark["Upper Level"].str.split(
            " ", expand=True
        )

    # replace the small o with star to conform with NIST
    stark["conf_i"] = stark["conf_i"].str.replace("o", "*")
    stark["conf_k"] = stark["conf_k"].str.replace("o", "*")
    stark["term_i"] = stark["term_i"].str.replace("o", "*")
    stark["term_k"] = stark["term_k"].str.replace("o", "*")

    # strip spaces from the stark parameters
    # this is necessary, because sometimes there is a "/2" at the end of some values
    # causing pandas to interpret the whole column as object
    stark["electron a0"] = stark["electron a0"].astype(str).str.strip()
    stark["electron a1"] = stark["electron a1"].astype(str).str.strip()
    stark["electron a2"] = stark["electron a2"].astype(str).str.strip()
    stark["electron b0"] = stark["electron b0"].astype(str).str.strip()
    stark["electron b1"] = stark["electron b1"].astype(str).str.strip()
    stark["electron b2"] = stark["electron b2"].astype(str).str.strip()

    # for some values the electron b0 entry contains a /2
    # i interpret that as meaning, the value should be divided by two
    stark["electron b0"] = stark["electron b0"].apply(
        lambda x: float(x[:-2]) / 2.0 if str(x).endswith("/2") else x
    )

    # drop duplicates. duplicates are entries that have the same configuration, term and J in upper and lower level
    stark = stark.drop_duplicates(
        subset=[
            "conf_i",
            "term_i",
            "J_i",
            "conf_k",
            "term_k",
            "J_k",
        ]
    )

    stark_total = stark.shape[0]

    return stark, stark_total


def get_line_db_from_file(file_path: str, element_string: str) -> Tuple[pd.DataFrame, Tuple[int, int]]:
    """
    Load emission line data from a NIST-formatted CSV file.

    Reads and processes emission line data, normalizing notation to match
    Stark-B conventions for later matching.

    Parameters
    ----------
    file_path : str
        Directory path containing the line database CSV files.
    element_string : str
        Ion identifier string (e.g., "Fe_I", "Ca_II").

    Returns
    -------
    tuple
        - pd.DataFrame: DataFrame with line data including:
            - element: Chemical element symbol
            - sp_num: Ionization state (1=neutral, 2=singly ionized, etc.)
            - ritz_wl_vac(nm): Vacuum wavelength in nanometers
            - Aki(s^-1): Einstein A coefficient
            - Ei(eV), Ek(eV): Lower and upper energy levels
            - g_i, g_k: Statistical weights
            - conf_i, term_i, conf_k, term_k: Quantum state descriptions
        - tuple[int, int]: (total_lines, lines_after_dropping_na)

    Notes
    -----
    The function:
    - Removes angle brackets from configuration strings
    - Removes spaces from term symbols
    - Drops rows with missing essential values
    """
    line = pd.read_csv(file_path + element_string + ".csv", dtype=str)

    # we have to eliminate the <> signs in the configuration to conform with stark-b
    line["conf_i"] = line["conf_i"].str.replace(r"[<>]", "", regex=True)
    line["conf_k"] = line["conf_k"].str.replace(r"[<>]", "", regex=True)

    # eliminate spaces in the term to fit with stark
    line["term_i"] = line["term_i"].str.replace(r" ", "", regex=True)
    line["term_k"] = line["term_k"].str.replace(r" ", "", regex=True)

    lines_total = line.shape[0]

    # drop lines that miss values in the fields that we need
    line = line.dropna(
        subset=[
            "ritz_wl_vac(nm)",
            "Aki(s^-1)",
            "Ei(eV)",
            "Ek(eV)",
            "g_i",
            "g_k",
            "conf_i",
            "term_i",
            "conf_k",
            "term_k",
        ]
    )
    lines_after_dropping_na = line.shape[0]

    return line, (lines_total, lines_after_dropping_na)


def match_line_and_stark_data(
    file_path_line: str, file_path_stark: str, ion_string: str
) -> Tuple[pd.DataFrame, Tuple[int, int, int, int]]:
    """
    Match emission lines with their Stark broadening parameters.

    Performs a join between NIST emission line data and Stark-B broadening
    parameters based on quantum state descriptions (configuration, term, J).

    Parameters
    ----------
    file_path_line : str
        Directory path containing NIST line database CSV files.
    file_path_stark : str
        Directory path containing Stark-B database CSV files.
    ion_string : str
        Ion identifier string (e.g., "Fe_I", "Ca_II").

    Returns
    -------
    tuple
        - pd.DataFrame: Matched database with columns:
            - element, sp_num: Element and ionization state
            - ritz_wl_vac(nm): Vacuum wavelength
            - Aki(s^-1): Einstein A coefficient
            - Ei(eV), Ek(eV), g_i, g_k: Energy levels and weights
            - N (cm-3): Reference electron density
            - electron a0/a1/a2: Stark width coefficients
            - electron b0/b1/b2: Stark shift coefficients
        - tuple[int, int, int, int]: Statistics
            (total_lines, lines_after_na, stark_lines, matched_lines)

    Notes
    -----
    The matching algorithm tries multiple strategies:
    1. Exact match on configuration and term
    2. Suffix match (for cases where Stark includes core orbitals)
    3. Full match including J quantum numbers for ambiguous cases
    """
    # get the line and stark db for this ion
    line, (num_lines_total, num_lines_after_dropping_na) = get_line_db_from_file(
        file_path_line, ion_string
    )
    stark, num_stark_lines = get_stark_db_from_file(file_path_stark, ion_string)

    # create an empty database
    df = pd.DataFrame(
        columns=[
            "element",
            "sp_num",
            "ritz_wl_vac(nm)",
            "Aki(s^-1)",
            "Ei(eV)",
            "Ek(eV)",
            "g_i",
            "g_k",
            "N (cm-3)",
            "electron a0",
            "electron a1",
            "electron a2",
            "electron b0",
            "electron b1",
            "electron b2",
        ]
    )

    num_matched_lines = 0
    for _, row in line.iterrows():

        print(row)

        # we first try to match the conf and term we find in nist with the ones in stark
        matching_stark = stark[
            (stark["conf_i"] == row["conf_i"])
            & (stark["term_i"] == row["term_i"])
            & (stark["conf_k"] == row["conf_k"])
            & (stark["term_k"] == row["term_k"])
            # & (stark["source"] == 0)
        ]

        print(matching_stark.to_string())

        # sometimes the stark configuration includes terms in front, whereas nist doesn't
        # example: stark 2p6.3s.9s WHEREAS nist 3s.9s
        # therefore we check if the ends of the level configuration string match
        if matching_stark.empty:
            matching_stark = stark[
                (stark["conf_i"].str.endswith(row["conf_i"]))
                & (stark["term_i"] == row["term_i"])
                & (stark["conf_k"].str.endswith(row["conf_k"]))
                & (stark["term_k"] == row["term_k"])
                #     & (stark["source"] == 0)
            ]

        # if there are multiple matches, we have to use the J values also
        if not matching_stark.empty and matching_stark.shape[0] > 1:
            matching_stark = stark[
                (stark["conf_i"] == row["conf_i"])
                & (stark["term_i"] == row["term_i"])
                & (stark["conf_k"] == row["conf_k"])
                & (stark["term_k"] == row["term_k"])
                & (stark["J_i"] == row["J_i"])
                & (stark["J_k"] == row["J_k"])
                #                & (stark["source"] == 0)
            ]

        # print if we found a hit
        if not matching_stark.empty:
            num_matched_lines += 1

            entry = pd.DataFrame.from_dict(
                {
                    "element": row["element"],
                    "sp_num": row["sp_num"],
                    "ritz_wl_vac(nm)": row["ritz_wl_vac(nm)"],
                    "Aki(s^-1)": row["Aki(s^-1)"],
                    "Ei(eV)": row["Ei(eV)"],
                    "Ek(eV)": row["Ek(eV)"],
                    "g_i": row["g_i"],
                    "g_k": row["g_k"],
                    "N (cm-3)": matching_stark["N (cm-3)"],
                    "electron a0": matching_stark["electron a0"],
                    "electron a1": matching_stark["electron a1"],
                    "electron a2": matching_stark["electron a2"],
                    "electron b0": matching_stark["electron b0"],
                    "electron b1": matching_stark["electron b1"],
                    "electron b2": matching_stark["electron b2"],
                }
            )

            df = pd.concat([df, entry], ignore_index=True)

            if matching_stark.shape[0] > 1:
                print("Multiple matches found!")
                print(matching_stark)

    return df, (
        num_lines_total,
        num_lines_after_dropping_na,
        num_stark_lines,
        num_matched_lines,
    )


def create_big_matched_file(stark_path: str, line_path: str, processed_path: str) -> None:
    """
    Create a consolidated database file matching all available ions.

    Iterates through all ion files in the Stark database directory, matches
    them with corresponding NIST line data, and writes a single consolidated
    CSV file containing all matched lines.

    Parameters
    ----------
    stark_path : str
        Directory path containing Stark-B database CSV files.
    line_path : str
        Directory path containing NIST line database CSV files.
    processed_path : str
        Output directory path for the consolidated database file.

    Returns
    -------
    None
        Writes 'lines_processed_all.csv' to the processed_path directory.

    Notes
    -----
    The output file is used directly by the LIBSkit C++ backend for
    spectrum calculations with Stark broadening.

    Example
    -------
    >>> create_big_matched_file(
    ...     "./raw_data/stark/",
    ...     "./raw_data/nist/",
    ...     "./databases/"
    ... )
    """
    # get all the ions for which we have stark data
    ions = [Path(f).stem for f in listdir(stark_path) if isfile(join(stark_path, f))]

    # create an empty database
    all_ions_df = pd.DataFrame(
        columns=[
            "element",
            "sp_num",
            "ritz_wl_vac(nm)",
            "Aki(s^-1)",
            "Ei(eV)",
            "Ek(eV)",
            "g_i",
            "g_k",
            "N (cm-3)",
            "electron a0",
            "electron a1",
            "electron a2",
            "electron b0",
            "electron b1",
            "electron b2",
        ]
    )

    for ion in ions:
        matching_lines, _ = match_line_and_stark_data(line_path, stark_path, ion)
        all_ions_df = pd.concat([all_ions_df, matching_lines], ignore_index=True)

    all_ions_df.to_csv(processed_path + "lines_processed_all.csv", index=False)


def generate_extended_lines_database(
    database_path: str,
    default_stark_fwhm: float = 0.01,
    top_n: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Generate an extended emission lines database that includes lines without
    Stark broadening parameters.

    Merges the existing ``lines_processed_all.csv`` (lines *with* known Stark
    parameters) with additional "complete" lines from
    ``nist_asd_lines_vacuum.csv`` that have all required spectroscopic data
    (wavelength, Aki, energies, statistical weights) but lack Stark parameters.

    Missing Stark parameters are imputed so that:

    * The Stark FWHM equals *default_stark_fwhm* (nm) when the electron
      density matches the reference density (``n_e_ref = 1e17 cm⁻³``).
    * The FWHM scales linearly with electron density (physically motivated).
    * There is no temperature dependence (``a1 = a2 = 0``) and no Stark
      shift (``b0 = b1 = b2 = 0``).

    After merging, lines can optionally be pruned to the *top_n* strongest
    lines per ion.  The ranking criterion is ``Aki * exp(-Ei)`` which
    favours strong transitions from low-lying energy levels.

    Parameters
    ----------
    database_path : str
        Path to the directory containing ``lines_processed_all.csv`` and
        ``nist_asd_lines_vacuum.csv``.
    default_stark_fwhm : float, optional
        Default Stark broadening FWHM in nanometres at the reference electron
        density.  Default is 0.01 nm.
    top_n : int or None, optional
        If given, keep only the *top_n* strongest lines per ion (element +
        ionisation stage) ranked by ``Aki * exp(-Ei)``.  ``None`` (default)
        keeps all lines.
    verbose : int, optional
        Verbosity level.  If > 0, print a warning when duplicate lines are
        detected and removed.  Default is 0 (silent).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 16 columns (the original 15 from
        ``lines_processed_all.csv`` plus ``Acc``, a numeric accuracy
        fraction for the Einstein A coefficient).
    """
    import numpy as np

    # ---- Load the existing (Stark-matched) database ----
    lines_path = join(database_path, "lines_processed_all.csv")
    lines = pd.read_csv(lines_path)

    # ---- Load the full NIST database and filter for complete rows ----
    nist_path = join(database_path, "nist_asd_lines_vacuum.csv")
    nist = pd.read_csv(nist_path)

    required_cols = [
        "ritz_wl_vac(nm)",
        "Aki(s^-1)",
        "Ei(eV)",
        "Ek(eV)",
        "g_i",
        "g_k",
    ]
    nist_complete = nist.dropna(subset=required_cols).copy()

    # ---- Convert NIST accuracy codes to numeric fractions ----
    nist_complete["Acc_frac"] = nist_complete["Acc"].apply(acc_code_to_fraction)

    # ---- Identify lines already present in the processed database ----
    # Use physics-based keys (energy levels, statistical weights, and
    # wavelength) to reliably detect duplicates regardless of small
    # wavelength rounding differences (e.g. air vs vacuum convention).
    # The wavelength is rounded to 1 decimal (0.1 nm) — tight enough to
    # separate distinct multiplet components but loose enough to catch
    # air/vacuum mismatches (~0.1–0.2 nm shift).
    def _physics_key(df: pd.DataFrame) -> pd.Series:
        return (
            df["element"]
            + "_"
            + df["sp_num"].astype(str)
            + "_"
            + df["Ei(eV)"].astype(float).round(3).astype(str)
            + "_"
            + df["Ek(eV)"].astype(float).round(3).astype(str)
            + "_"
            + df["g_i"].astype(int).astype(str)
            + "_"
            + df["g_k"].astype(int).astype(str)
            + "_"
            + df["ritz_wl_vac(nm)"].astype(float).round(1).astype(str)
        )

    lines["_key"] = _physics_key(lines)
    nist_complete["_key"] = _physics_key(nist_complete)

    # ---- Join Acc onto existing Stark-matched lines from NIST ----
    # Build a lookup from physics-key → Acc fraction.  Where multiple
    # NIST rows share the same key, keep the first (typically identical).
    nist_acc_lookup = (
        nist_complete[["_key", "Acc_frac"]]
        .drop_duplicates(subset="_key", keep="first")
        .set_index("_key")["Acc_frac"]
    )
    lines["Acc"] = lines["_key"].map(nist_acc_lookup).fillna(0.50)

    new_mask = ~nist_complete["_key"].isin(lines["_key"])
    new_lines = nist_complete.loc[new_mask].copy()

    # ---- Compute imputed Stark parameters ----
    # The C++ Stark FWHM formula is:
    #   w = 10^(a0 + a1*log10(T) + a2*log10(T)^2) / 10   [nm]
    #   fwhm = w * ne / n_e_ref * 1e-6
    # With a1=a2=0 and ne = n_e_ref*1e6 (same physical density):
    #   fwhm = 10^a0 / 10
    # So a0 = log10(10 * default_stark_fwhm).
    a0_default = float(np.log10(10.0 * default_stark_fwhm))
    n_e_ref = 1e17  # cm^-3, standard reference

    new_df = pd.DataFrame(
        {
            "element": new_lines["element"].values,
            "sp_num": new_lines["sp_num"].astype(int).values,
            "ritz_wl_vac(nm)": new_lines["ritz_wl_vac(nm)"].astype(float).values,
            "Aki(s^-1)": new_lines["Aki(s^-1)"].astype(float).values,
            "Ei(eV)": new_lines["Ei(eV)"].astype(float).values,
            "Ek(eV)": new_lines["Ek(eV)"].astype(float).values,
            "g_i": new_lines["g_i"].astype(float).values,
            "g_k": new_lines["g_k"].astype(float).values,
            "N (cm-3)": n_e_ref,
            "electron a0": a0_default,
            "electron a1": 0.0,
            "electron a2": 0.0,
            "electron b0": 0.0,
            "electron b1": 0.0,
            "electron b2": 0.0,
            "Acc": new_lines["Acc_frac"].values,
        }
    )

    # ---- Optionally keep only the top N *new* lines per ion ----
    # Lines from the standard (Stark-matched) database are always kept;
    # the top_n cutoff only applies to the newly added imputed lines.
    if top_n is not None:
        # Ranking: Aki * exp(-Ei)  (higher = stronger / lower-energy)
        new_df["_rank_score"] = (
            new_df["Aki(s^-1)"].astype(float)
            * np.exp(-new_df["Ei(eV)"].astype(float))
        )
        # Build ion key from element + sp_num
        new_df["_ion"] = (
            new_df["element"].astype(str) + "_" + new_df["sp_num"].astype(str)
        )
        new_df = (
            new_df
            .sort_values("_rank_score", ascending=False)
            .groupby("_ion", sort=False)
            .head(top_n)
            .drop(columns=["_rank_score", "_ion"])
            .reset_index(drop=True)
        )

    extended = pd.concat([lines, new_df], ignore_index=True)

    # ---- Sanity check: no processed-database line should have been ----
    # ---- re-added from NIST. If any are found, the physics-key --------
    # ---- matching likely has a problem (e.g. air/vacuum mismatch). ----
    if verbose > 0 and len(new_df) > 0:
        new_keys = set(_physics_key(new_df))
        lines_keys = set(lines["_key"])
        overlap = lines_keys & new_keys
        if overlap:
            print(
                f"  ⚠ Extended database: {len(overlap)} line(s) from "
                f"lines_processed_all.csv were also added from "
                f"nist_asd_lines_vacuum.csv — possible air/vacuum "
                f"wavelength mismatch in the source database."
            )

    lines = lines.drop(columns=["_key"])

    # Drop internal columns from the extended result
    cols_to_drop = [c for c in ("_key",) if c in extended.columns]
    if cols_to_drop:
        extended = extended.drop(columns=cols_to_drop)

    # Sort by (element, sp_num, wavelength) so the C++ backend's binary
    # search can find every line.  Without this sort the newly appended
    # lines appear after the last original line of each ion and are
    # invisible to the wavelength-range lookup.
    extended = extended.sort_values(
        ["element", "sp_num", "ritz_wl_vac(nm)"]
    ).reset_index(drop=True)

    return extended
