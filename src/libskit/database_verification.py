#!/usr/bin/env python3
#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""
Database verification module for libskit atomic databases.

This module provides functions to verify that the required CSV database files
exist and conform to the expected format for libskit's C++ backend.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constant for valid roman numerals (supports up to 100 for highly ionized atoms)
VALID_ROMANS = {
    'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX',
    'XXXI', 'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', 'XXXVII', 'XXXVIII', 'XXXIX', 'XL',
    'XLI', 'XLII', 'XLIII', 'XLIV', 'XLV', 'XLVI', 'XLVII', 'XLVIII', 'XLIX', 'L',
    "LI", "LII", "LIII", "LIV", "LV", "LVI", "LVII", "LVIII", "LIX", "LX",
    "LXI", "LXII", "LXIII", "LXIV", "LXV", "LXVI", "LXVII", "LXVIII", "LXIX", "LXX",
    "LXXI", "LXXII", "LXXIII", "LXXIV", "LXXV", "LXXVI", "LXXVII", "LXXVIII", "LXXIX", "LXXX",
    "LXXXI", "LXXXII", "LXXXIII", "LXXXIV", "LXXXV", "LXXXVI", "LXXXVII", "LXXXVIII", "LXXXIX", "XC",
    "XCI", "XCII", "XCIII", "XCIV", "XCV", "XCVI", "XCVII", "XCVIII", "XCIX", "C",
    "CI", "CII", "CIII", "CIV", "CV", "CVI", "CVII", "CVIII", "CIX", "CX", "CXI", "CXII", "CXIII", "CXIV", "CXV", "CXVI", "CXVII", "CXVIII", "CXIX", "CXX",
    "CXXI", "CXXII", "CXXIII", "CXXIV", "CXXV", "CXXVI", "CXXVII", "CXXVIII", "CXXIX", "CXXX",
    "CXXXI", "CXXXII", "CXXXIII", "CXXXIV", "CXXXV", "CXXXVI", "CXXXVII", "CXXXVIII", "CXXXIX", "CXL",
    "CXLI", "CXLII", "CXLIII", "CXLIV", "CXLV", "CXLVI", "CXLVII", "CXLVIII", "CXLIX", "CL"
}


class EmissionLineRecord(BaseModel):
    """Schema for emission line database records."""
    element: str = Field(..., description="Chemical element symbol (e.g., 'Na', 'Ca')")
    sp_num: int = Field(..., ge=1, description="Species number (ionization state)")
    ritz_wl_vac_nm: float = Field(..., alias="ritz_wl_vac(nm)", gt=0,
                                  description="Wavelength in vacuum (nm)")
    Aki_s_minus_1: float = Field(..., alias="Aki(s^-1)", ge=0,
                                 description="Einstein A coefficient (s^-1)")
    Ei_eV: float = Field(..., alias="Ei(eV)", ge=0,
                         description="Lower energy level (eV)")
    Ek_eV: float = Field(..., alias="Ek(eV)", ge=0,
                         description="Upper energy level (eV)")
    g_i: float = Field(..., gt=0, description="Statistical weight of lower level")
    g_k: float = Field(..., gt=0, description="Statistical weight of upper level")
    N_cm_minus_3: float = Field(..., alias="N (cm-3)", ge=0,
                                description="Reference electron density (cm^-3)")
    electron_a0: float = Field(..., alias="electron a0", description="Stark broadening parameter a0")
    electron_a1: float = Field(..., alias="electron a1", description="Stark broadening parameter a1")
    electron_a2: float = Field(..., alias="electron a2", description="Stark broadening parameter a2")
    electron_b0: float = Field(..., alias="electron b0", description="Stark broadening parameter b0")
    electron_b1: float = Field(..., alias="electron b1", description="Stark broadening parameter b1")
    electron_b2: float = Field(..., alias="electron b2", description="Stark broadening parameter b2")

    @field_validator('element')
    @classmethod
    def validate_element(cls, v):
        """Validate element symbols are proper chemical elements."""
        # Common elements found in LIBS databases
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        }
        if v not in valid_elements:
            raise ValueError(f"Invalid element symbol: {v}")
        return v

    @field_validator('Ek_eV')
    @classmethod
    def validate_energy_levels(cls, v, info):
        """Ensure upper energy level is higher than lower level."""
        if info.data and 'Ei_eV' in info.data and v <= info.data['Ei_eV']:
            raise ValueError(f"Upper energy level ({v}) must be higher than lower level ({info.data['Ei_eV']})")
        return v


class IonizationEnergyRecord(BaseModel):
    """Schema for ionization energy database records."""
    sp_name: str = Field(..., alias="Sp. Name",
                         description="Species name (e.g., 'H I', 'Ca II')")
    ionization_energy_eV: float = Field(..., alias="Ionization Energy (b) (eV)",
                                        gt=0, description="Ionization energy in eV")

    @field_validator('sp_name')
    @classmethod
    def validate_species_name(cls, v):
        """Validate species name format (Element + Roman numeral)."""
        parts = v.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Species name must be 'Element RomanNumeral', got: {v}")

        element, roman = parts
        # Validate element symbol
        if not element[0].isupper() or not element.isalpha():
            raise ValueError(f"Invalid element in species name: {element}")

        # Validate roman numeral (supports up to 50+ for highly ionized atoms)
        if roman not in VALID_ROMANS:
            raise ValueError(f"Invalid roman numeral in species name: {roman}")

        return v


class EnergyLevelRecord(BaseModel):
    """Schema for energy level database records."""
    g: float = Field(..., ge=0, description="Statistical weight")
    level_eV: float = Field(..., alias="Level (eV)", ge=0,
                           description="Energy level in eV")
    sp_name: str = Field(..., alias="Sp. Name",
                         description="Species name (e.g., 'H I', 'Ca II')")

    @field_validator('sp_name')
    @classmethod
    def validate_species_name(cls, v):
        """Validate species name format (Element + Roman numeral)."""
        parts = v.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Species name must be 'Element RomanNumeral', got: {v}")

        element, roman = parts
        if not element[0].isupper() or not element.isalpha():
            raise ValueError(f"Invalid element in species name: {element}")

        if roman not in VALID_ROMANS:
            raise ValueError(f"Invalid roman numeral in species name: {roman}")

        return v


class DatabaseVerificationResult(BaseModel):
    """Result of database verification."""
    is_valid: bool = Field(..., description="Whether all databases are valid")
    database_path: str = Field(..., description="Path to database directory")
    files_found: Dict[str, bool] = Field(..., description="Which required files were found")
    validation_results: Dict[str, Dict[str, Any]] = Field(..., description="Detailed validation results")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")


def verify_databases(database_directory: str,
                    sample_size: Optional[int] = 1000,
                    strict_mode: bool = True) -> DatabaseVerificationResult:
    """
    Verify that libskit atomic databases exist and conform to expected format.

    Args:
        database_directory: Path to directory containing the database CSV files
        sample_size: Number of records to validate from each file (None for all records)
        strict_mode: If True, fail on warnings; if False, only fail on errors

    Returns:
        DatabaseVerificationResult object with detailed validation information

    Required files:
        - lines_processed_all.csv: Emission line data (15 columns)
        - ionization_energies_processed.csv: Ionization energy data (2 columns)
        - levels_processed.csv: Energy level data (3 columns)

    Examples
    --------
    >>> import libskit
    >>> from libskit.database_verification import verify_databases
    >>> result = verify_databases(libskit.Database.active().path)
    >>> result.is_valid
    True
    >>> result.files_found['lines_processed_all.csv']
    True
    """

    db_path = Path(database_directory)
    result = DatabaseVerificationResult(
        is_valid=True,
        database_path=str(db_path),
        files_found={},
        validation_results={},
        errors=[],
        warnings=[]
    )

    # Required database files and their expected schemas
    required_files = {
        "lines_processed_all.csv": {
            "schema": EmissionLineRecord,
            "expected_columns": [
                "element", "sp_num", "ritz_wl_vac(nm)", "Aki(s^-1)",
                "Ei(eV)", "Ek(eV)", "g_i", "g_k", "N (cm-3)",
                "electron a0", "electron a1", "electron a2",
                "electron b0", "electron b1", "electron b2"
            ],
            "min_records": 10
        },
        "ionization_energies_processed.csv": {
            "schema": IonizationEnergyRecord,
            "expected_columns": ["Sp. Name", "Ionization Energy (b) (eV)"],
            "min_records": 100
        },
        "levels_processed.csv": {
            "schema": EnergyLevelRecord,
            "expected_columns": ["g", "Level (eV)", "Sp. Name"],
            "min_records": 100
        }
    }

    logger.info(f"Verifying databases in: {db_path}")

    # Check if directory exists
    if not db_path.exists():
        result.errors.append(f"Database directory does not exist: {db_path}")
        result.is_valid = False
        return result

    if not db_path.is_dir():
        result.errors.append(f"Database path is not a directory: {db_path}")
        result.is_valid = False
        return result

    # Check each required file
    for filename, file_config in required_files.items():
        file_path = db_path / filename
        result.files_found[filename] = file_path.exists()

        if not file_path.exists():
            result.errors.append(f"Required file not found: {filename}")
            result.is_valid = False
            continue

        logger.info(f"Validating {filename}...")

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            file_result = {
                "file_size_bytes": file_path.stat().st_size,
                "total_records": len(df),
                "columns_found": list(df.columns),
                "validation_errors": [],
                "validation_warnings": [],
                "sample_validated": False
            }

            # Check if file is empty
            if len(df) == 0:
                error_msg = f"{filename}: File is empty"
                result.errors.append(error_msg)
                file_result["validation_errors"].append(error_msg)
                result.is_valid = False
                continue

            # Check minimum number of records
            if len(df) < file_config["min_records"]:
                warning_msg = f"{filename}: Only {len(df)} records found, expected at least {file_config['min_records']}"
                result.warnings.append(warning_msg)
                file_result["validation_warnings"].append(warning_msg)

            # Check column names
            expected_cols = set(file_config["expected_columns"])
            actual_cols = set(df.columns)

            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols

            if missing_cols:
                error_msg = f"{filename}: Missing required columns: {sorted(missing_cols)}"
                result.errors.append(error_msg)
                file_result["validation_errors"].append(error_msg)
                result.is_valid = False

            if extra_cols:
                warning_msg = f"{filename}: Extra columns found: {sorted(extra_cols)}"
                result.warnings.append(warning_msg)
                file_result["validation_warnings"].append(warning_msg)

            # Validate data format using Pydantic schema
            if not missing_cols:  # Only validate if all required columns are present
                schema = file_config["schema"]
                sample_df = df.head(sample_size) if sample_size else df

                validation_errors = 0
                for row_num, (idx, row) in enumerate(sample_df.iterrows()):
                    try:
                        # Convert row to dict and validate with Pydantic
                        record_dict = row.to_dict()
                        schema(**record_dict)
                    except ValidationError as e:
                        validation_errors += 1
                        if validation_errors <= 5:  # Only show first 5 errors
                            error_detail = f"{filename} row {row_num + 2}: {str(e)}"
                            result.errors.append(error_detail)
                            file_result["validation_errors"].append(error_detail)

                if validation_errors > 0:
                    result.is_valid = False
                    if validation_errors > 5:
                        additional_msg = f"{filename}: {validation_errors - 5} additional validation errors not shown"
                        result.errors.append(additional_msg)
                        file_result["validation_errors"].append(additional_msg)
                else:
                    file_result["sample_validated"] = True
                    logger.info(f"✓ {filename}: {len(sample_df)} records validated successfully")

            result.validation_results[filename] = file_result

        except Exception as e:
            error_msg = f"{filename}: Error reading/validating file: {str(e)}"
            result.errors.append(error_msg)
            result.validation_results[filename] = {"error": error_msg}
            result.is_valid = False

    # Final validation
    if strict_mode and result.warnings:
        result.is_valid = False

    # Summary logging
    if result.is_valid:
        logger.info("✅ All database files are valid!")
    else:
        logger.error(f"❌ Database validation failed with {len(result.errors)} errors")

    return result


def print_verification_report(result: DatabaseVerificationResult) -> None:
    """Print a human-readable verification report."""

    print("=" * 80)
    print("LIBSKIT DATABASE VERIFICATION REPORT")
    print("=" * 80)

    print(f"\nDatabase Directory: {result.database_path}")
    print(f"Overall Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")

    print(f"\nFiles Found:")
    for filename, found in result.files_found.items():
        status = "✅" if found else "❌"
        print(f"  {status} {filename}")

    if result.validation_results:
        print(f"\nValidation Details:")
        for filename, details in result.validation_results.items():
            if "error" in details:
                print(f"\n  ❌ {filename}: {details['error']}")
                continue

            print(f"\n  📄 {filename}:")
            print(f"    Size: {details['file_size_bytes']:,} bytes")
            print(f"    Records: {details['total_records']:,}")
            print(f"    Columns: {len(details['columns_found'])}")

            if details.get('sample_validated'):
                print(f"    Validation: ✅ Passed")
            elif details['validation_errors']:
                print(f"    Validation: ❌ {len(details['validation_errors'])} errors")

    if result.errors:
        print(f"\n🚨 ERRORS ({len(result.errors)}):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. {error}")

    if result.warnings:
        print(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    """Command-line interface for database verification."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Verify libskit atomic databases")
    parser.add_argument("database_directory",
                       help="Path to directory containing database CSV files")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of records to validate per file (default: 1000)")
    parser.add_argument("--strict", action="store_true",
                       help="Treat warnings as errors")
    parser.add_argument("--quiet", action="store_true",
                       help="Only show summary, not detailed logging")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Run verification
    result = verify_databases(args.database_directory,
                             sample_size=args.sample_size,
                             strict_mode=args.strict)

    # Print report
    print_verification_report(result)

    # Exit with appropriate code
    sys.exit(0 if result.is_valid else 1)
