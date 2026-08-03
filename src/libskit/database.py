#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Database — explicit handle to libskit's atomic data.

Internally libskit maintains three C++ globals (``LINES``, ``LEVELS``,
``IONIZATION_ENERGIES``) that every :class:`PlasmaModel` and
:class:`Experiment` reads from.  The :class:`Database` class is the
public, declarative way to populate that state.

Each instance remembers how it was constructed (bundled CSVs, an
arbitrary path, or one of the extended / filtered / Stark-only variants)
and can re-activate itself on demand. The bundled database is activated
automatically on ``import libskit``, so simple usage requires no setup::

    import libskit
    model = libskit.LTEModel({"Fe": 1.0}, 10000, 1e23, 1e-3)

To compare results across several databases, construct each one
explicitly and pass it via the keyword-only ``db=`` argument that every
model and :class:`Experiment` constructor accepts::

    db_full     = libskit.Database.bundled()
    db_extended = libskit.Database.extended(top_n=100)

    m1 = libskit.LTEModel({"Fe": 1.0}, 10000, 1e23, 1e-3, db=db_full)
    m2 = libskit.LTEModel({"Fe": 1.0}, 10000, 1e23, 1e-3, db=db_extended)
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from . import _libskit

# When installed via pip the CSVs live next to this file in <package>/data/.
# In a source checkout (e.g. CI builds that only run cmake) that directory
# does not exist, but the same files are available at <repo>/databases/.
# Fall back to the repo-root copy so 'import libskit' works without an install.
_REQUIRED_DB_FILES = (
    "lines_processed_all.csv",
    "ionization_energies_processed.csv",
    "levels_processed.csv",
)


def _resolve_bundled_db_path() -> Path:
    wheel_dir = Path(__file__).parent / "data"
    if all((wheel_dir / f).is_file() for f in _REQUIRED_DB_FILES):
        return wheel_dir
    # src/libskit/database.py -> src/libskit -> src -> <repo>
    repo_dir = Path(__file__).resolve().parents[2] / "databases"
    if all((repo_dir / f).is_file() for f in _REQUIRED_DB_FILES):
        return repo_dir
    return wheel_dir  # original location (will fail on activate, handled by caller)


_BUNDLED_DB_PATH: Path = _resolve_bundled_db_path()

# Tracks the Database instance whose state is currently mirrored in
# libskit's C++ globals. ``activate()`` is a no-op when the requested
# Database already matches this reference, which keeps repeated model
# construction cheap.
_ACTIVE_DB: Optional["Database"] = None


def _set_active(db: "Database") -> None:
    global _ACTIVE_DB
    _ACTIVE_DB = db


def _active_db() -> Optional["Database"]:
    return _ACTIVE_DB


class Database:
    """Handle to a specific atomic-data configuration.

    Construct with one of the classmethods (:meth:`bundled`,
    :meth:`from_path`, :meth:`extended`, :meth:`filtered`,
    :meth:`filtered_stark`); the constructor itself is reserved for
    internal use.

    Parameters
    ----------
    path :
        Directory containing ``lines_processed_all.csv``,
        ``ionization_energies_processed.csv`` and ``levels_processed.csv``.
        For derived databases this is a temporary directory created by
        the corresponding classmethod.
    description :
        Short human-readable label, surfaced in :meth:`__repr__`.
    source_path :
        Original (non-temp) directory the data was derived from, used by
        derived classmethods so that, e.g., ``Database.filtered`` knows
        where to find ``nist_asd_lines_vacuum.csv``.
    """

    __slots__ = ("_path", "_description", "_source_path")

    def __init__(
        self,
        path: Path,
        description: str,
        source_path: Optional[Path] = None,
    ) -> None:
        # Resolve so that two Database instances built from the same logical
        # location (e.g. via bundled() and from_path(_BUNDLED_DB_PATH))
        # compare equal under ``is_active`` / ``activate``.
        self._path = Path(path).resolve()
        self._description = description
        self._source_path = (
            Path(source_path).resolve() if source_path is not None else self._path
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def bundled(cls) -> "Database":
        """The atomic-data CSVs that ship inside the libskit wheel."""
        return cls(_BUNDLED_DB_PATH, description="bundled")

    @classmethod
    def active(cls) -> Optional["Database"]:
        """Return the :class:`Database` currently mirrored into C++ globals.

        ``None`` only if no database has been activated yet (typically
        because the bundled data was missing at import time).
        """
        return _ACTIVE_DB

    @classmethod
    def from_path(cls, path: os.PathLike[str] | str) -> "Database":
        """Use a user-provided directory of pre-processed CSVs.

        The directory must contain ``lines_processed_all.csv``,
        ``ionization_energies_processed.csv`` and ``levels_processed.csv``.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Database path does not exist or is not a directory: {resolved}"
            )
        return cls(resolved, description=f"from_path({resolved})")

    @classmethod
    def extended(
        cls,
        *,
        default_stark_fwhm: float = 0.01,
        top_n: Optional[int] = None,
        source: Optional["Database"] = None,
        verbose: int = 0,
    ) -> "Database":
        """Stark-matched lines plus every NIST line with imputed Stark FWHM.

        Stark-matched lines plus every NIST line with imputed Stark FWHM,
        materialised into a temporary directory that persists for the
        lifetime of the process.
        """
        from .generate_databases import generate_extended_lines_database

        src = (source or cls.bundled())._source_path
        df = generate_extended_lines_database(
            str(src),
            default_stark_fwhm=default_stark_fwhm,
            top_n=top_n,
            verbose=verbose,
        )
        tmp = Path(tempfile.mkdtemp(prefix="libskit_extended_"))
        df.to_csv(tmp / "lines_processed_all.csv", index=False)
        for fname in ("ionization_energies_processed.csv", "levels_processed.csv"):
            shutil.copy2(src / fname, tmp / fname)
        descr = f"extended(default_stark_fwhm={default_stark_fwhm}, top_n={top_n})"
        return cls(tmp, description=descr, source_path=src)

    @classmethod
    def filtered(
        cls,
        top_n: int,
        *,
        default_stark_fwhm: float = 0.01,
        source: Optional["Database"] = None,
        verbose: int = 0,
    ) -> "Database":
        """Top-``top_n`` strongest lines per ion across the extended DB.

        Top-``top_n`` strongest lines per ion across the extended
        database. Lines are ranked by ``Aki * exp(-Ek)``.
        """
        import numpy as np
        import pandas as pd

        ext = cls.extended(
            default_stark_fwhm=default_stark_fwhm,
            source=source,
            verbose=verbose,
        )
        df = pd.read_csv(ext._path / "lines_processed_all.csv")
        df["_rank"] = df["Aki(s^-1)"].astype(float) * np.exp(
            -df["Ek(eV)"].astype(float)
        )
        df["_ion"] = df["element"].astype(str) + "_" + df["sp_num"].astype(str)
        df = (
            df.sort_values("_rank", ascending=False)
            .groupby("_ion", sort=False)
            .head(top_n)
            .drop(columns=["_rank", "_ion"])
            .reset_index(drop=True)
        )
        tmp = Path(tempfile.mkdtemp(prefix="libskit_filtered_"))
        df.to_csv(tmp / "lines_processed_all.csv", index=False)
        for fname in ("ionization_energies_processed.csv", "levels_processed.csv"):
            shutil.copy2(ext._path / fname, tmp / fname)
        descr = f"filtered(top_n={top_n}, default_stark_fwhm={default_stark_fwhm})"
        return cls(tmp, description=descr, source_path=ext._source_path)

    @classmethod
    def filtered_stark(
        cls,
        top_n: int,
        *,
        source: Optional["Database"] = None,
        verbose: int = 0,
    ) -> "Database":
        """Top-``top_n`` strongest **Stark-matched** lines per ion.

        No imputation is performed: only lines with measured Stark
        broadening parameters survive a per-ion ``Aki * exp(-Ek)``
        cutoff applied to the standard ``lines_processed_all.csv``.
        """
        import numpy as np
        import pandas as pd

        src = (source or cls.bundled())._source_path
        df = pd.read_csv(src / "lines_processed_all.csv")
        df["_rank"] = df["Aki(s^-1)"].astype(float) * np.exp(
            -df["Ek(eV)"].astype(float)
        )
        df["_ion"] = df["element"].astype(str) + "_" + df["sp_num"].astype(str)
        df = (
            df.sort_values("_rank", ascending=False)
            .groupby("_ion", sort=False)
            .head(top_n)
            .drop(columns=["_rank", "_ion"])
            .reset_index(drop=True)
        )
        tmp = Path(tempfile.mkdtemp(prefix="libskit_stark_filtered_"))
        df.to_csv(tmp / "lines_processed_all.csv", index=False)
        for fname in ("ionization_energies_processed.csv", "levels_processed.csv"):
            shutil.copy2(src / fname, tmp / fname)
        if verbose:
            print(
                f"Stark-filtered database: {len(df)} total lines "
                f"({top_n} per ion max, Stark-matched only)"
            )
        return cls(tmp, description=f"filtered_stark(top_n={top_n})", source_path=src)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def activate(self, verbose: int = 0) -> None:
        """Mirror this Database into libskit's C++ globals (idempotent).

        If the C++ state is already configured against an instance with
        the same on-disk path the call is a no-op; otherwise the C++
        ``set_database_path`` + ``load_databases`` pair is invoked.
        """
        global _ACTIVE_DB
        if _ACTIVE_DB is not None and _ACTIVE_DB._path == self._path:
            _ACTIVE_DB = self
            return
        _libskit.set_database_path(str(self._path))
        _libskit.load_databases(verbose)
        _ACTIVE_DB = self

    @property
    def path(self) -> Path:
        return self._path

    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_active(self) -> bool:
        return _ACTIVE_DB is not None and _ACTIVE_DB._path == self._path

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Database({self._description!r}, path={self._path!s})"
