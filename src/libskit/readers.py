#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Readers for experimental spectrometer data files."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .spectrum import Spectrum
from .wavelengths import air_to_vacuum


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class OrderData:
    """Data for a single echelle order."""
    order: int
    wavelengths: np.ndarray
    intensities: np.ndarray
    pixels: np.ndarray


# ── AryReader ────────────────────────────────────────────────────────

# Default bilinear wavelength-correction coefficients for the Aryelle 200.
# Fitted from 148 Fe I/II lines matched to correct vacuum→air references
# (taleb21 5 µs late-delay data, all pixels):
#   Δλ(p, λ) = a·p + b·λ + c   [nm]
# where Δλ = λ_measured − λ_true, so  λ_corrected = λ_measured − Δλ.
# Residual σ = 0.0105 nm.
_DEFAULT_WL_COEFF_A = 4.468845e-06   # nm / pixel
_DEFAULT_WL_COEFF_B = -4.802578e-05  # dimensionless
_DEFAULT_WL_COEFF_C = -1.890736e-02  # nm


class AryReader:
    """Reader for Aryelle .ary (LTB Berlin) echelle spectrometer files.

    Parses the ZIP-archived binary data, resolves per-order arrays with
    pixel positions, and provides wavelength-corrected + stitched spectra.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ary`` file.
    wl_correction : (float, float, float) or None, optional
        Bilinear wavelength-correction coefficients ``(a, b, c)`` such
        that ``Δλ = a·pixel + b·λ + c``.  The corrected wavelength is
        ``λ_corr = λ_raw − Δλ``.  Pass ``None`` to skip correction
        (return raw manufacturer calibration).  Defaults to the
        Aryelle 200 coefficients fitted from Fe lamp lines.

    Examples
    --------
    >>> reader = AryReader("measurement.ary")
    >>> spectrum = reader.spectrum()          # stitched, corrected
    >>> for order in reader.orders:           # per-order access
    ...     print(order.order, order.wavelengths[0])
    """

    def __init__(
        self,
        path: str | Path,
        wl_correction: Optional[tuple[float, float, float]] = (
            _DEFAULT_WL_COEFF_A,
            _DEFAULT_WL_COEFF_B,
            _DEFAULT_WL_COEFF_C,
        ),
    ):
        self.path = Path(path)
        self.wl_correction = wl_correction
        self._wl_raw: np.ndarray
        self._irr_raw: np.ndarray
        self._pix: np.ndarray
        self._order_ids: np.ndarray
        self._aif: np.ndarray
        self._metadata_raw: Optional[Dict[str, str]] = None
        self._load()

    # ── Loading ──────────────────────────────────────────────────────

    _AIF_DTYPE = np.dtype([
        ("indStart", np.uint32), ("indEnd", np.uint32),
        ("order", np.uint16),    ("pixStart", np.uint16),
        ("pixEnd", np.uint16),   ("empty", np.uint16),
        ("xStart", np.float32),  ("xEnd", np.float32),
    ])

    _TMP_DTYPE = np.dtype([("y", np.float32), ("x", np.float32)])

    def _load(self) -> None:
        with zipfile.ZipFile(str(self.path)) as zf:
            basename = zf.namelist()[0].rsplit(".", 1)[0]
            tmp = np.frombuffer(zf.read(basename + ".~tmp"), dtype=self._TMP_DTYPE)
            self._aif = np.frombuffer(zf.read(basename + ".~aif"), dtype=self._AIF_DTYPE)

        n = len(tmp)
        self._wl_raw = np.array(tmp["x"], dtype=np.float64)
        self._irr_raw = np.array(tmp["y"], dtype=np.float64)
        self._pix = np.full(n, np.nan)
        self._order_ids = np.zeros(n, dtype=np.int32)

        for row in self._aif:
            i0, i1 = int(row["indStart"]), int(row["indEnd"]) + 1
            p0, p1 = int(row["pixStart"]), int(row["pixEnd"])
            npix = p1 - p0 + 1
            if npix == (i1 - i0):
                self._pix[i0:i1] = np.arange(p0, p1 + 1)
            else:
                self._pix[i0:i1] = np.linspace(p0, p1, i1 - i0)
            self._order_ids[i0:i1] = int(row["order"])

    # ── Metadata ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, str]:
        """Acquisition metadata from the .~rep file (lazy-loaded)."""
        if self._metadata_raw is None:
            self._metadata_raw = {}
            with zipfile.ZipFile(str(self.path)) as zf:
                rep_files = [f for f in zf.namelist() if f.endswith(".~rep")]
                if rep_files:
                    raw = zf.read(rep_files[0]).decode("ascii")
                    for line in raw.split("\r\n"):
                        if "=" in line:
                            key, _, val = line.partition("=")
                            self._metadata_raw[key] = val
        return self._metadata_raw

    # ── Per-order access ─────────────────────────────────────────────

    def _correct_wavelengths(self, wl: np.ndarray, pix: np.ndarray) -> np.ndarray:
        """Apply bilinear correction: λ_corr = λ_raw − (a·pix + b·λ + c)."""
        if self.wl_correction is None:
            return wl.copy()
        a, b, c = self.wl_correction
        return wl - (a * pix + b * wl + c)

    @property
    def orders(self) -> List[OrderData]:
        """Per-order spectra, wavelength-corrected and sorted by ascending λ."""
        result = []
        for row in self._aif:
            i0, i1 = int(row["indStart"]), int(row["indEnd"]) + 1
            wl = self._wl_raw[i0:i1].copy()
            irr = self._irr_raw[i0:i1].copy()
            pix = self._pix[i0:i1].copy()

            # Sort ascending by wavelength
            idx = np.argsort(wl)
            wl, irr, pix = wl[idx], irr[idx], pix[idx]

            # Apply wavelength correction
            wl = self._correct_wavelengths(wl, pix)

            result.append(OrderData(
                order=int(row["order"]),
                wavelengths=wl,
                intensities=irr,
                pixels=pix,
            ))

        result.sort(key=lambda o: np.median(o.wavelengths))
        return result

    @property
    def orders_raw(self) -> List[OrderData]:
        """Per-order spectra without wavelength correction."""
        result = []
        for row in self._aif:
            i0, i1 = int(row["indStart"]), int(row["indEnd"]) + 1
            wl = self._wl_raw[i0:i1].copy()
            irr = self._irr_raw[i0:i1].copy()
            pix = self._pix[i0:i1].copy()

            idx = np.argsort(wl)
            wl, irr, pix = wl[idx], irr[idx], pix[idx]

            result.append(OrderData(
                order=int(row["order"]),
                wavelengths=wl,
                intensities=irr,
                pixels=pix,
            ))

        result.sort(key=lambda o: np.median(o.wavelengths))
        return result

    # ── Stitching ────────────────────────────────────────────────────

    @staticmethod
    def _dissolve_orders(order_list: List[OrderData]) -> tuple[np.ndarray, np.ndarray]:
        """Stitch overlapping orders into a single monotonic spectrum.

        For each pair of adjacent orders the crossover is placed at the
        midpoint of their overlap region.
        """
        n = len(order_list)

        # Crossover wavelengths between adjacent orders
        crossovers = []
        for i in range(n - 1):
            wl_end_left = order_list[i].wavelengths[-1]
            wl_start_right = order_list[i + 1].wavelengths[0]
            crossovers.append(0.5 * (wl_end_left + wl_start_right))

        wl_parts: list[np.ndarray] = []
        irr_parts: list[np.ndarray] = []

        for i, o in enumerate(order_list):
            lo = crossovers[i - 1] if i > 0 else o.wavelengths[0]
            hi = crossovers[i] if i < n - 1 else o.wavelengths[-1]
            mask = (o.wavelengths >= lo) & (o.wavelengths <= hi)
            wl_parts.append(o.wavelengths[mask])
            irr_parts.append(o.intensities[mask])

        return np.concatenate(wl_parts), np.concatenate(irr_parts)

    def spectrum(self) -> Spectrum:
        """Return a stitched, wavelength-corrected Spectrum (air wavelengths).

        Orders are corrected with the bilinear model, then dissolved at
        the midpoints of their overlap regions into a single monotonic
        wavelength array.
        """
        wl, irr = self._dissolve_orders(self.orders)
        return Spectrum(wl, irr)

    def spectrum_raw(self) -> Spectrum:
        """Return a stitched Spectrum without wavelength correction."""
        wl, irr = self._dissolve_orders(self.orders_raw)
        return Spectrum(wl, irr)


# ── ChemCamReader ────────────────────────────────────────────────────

class ChemCamReader:
    """Reader for ChemCam CCS (calibration standard) CSV data files.

    Parses the PDS-style CSV format produced by MCCOPS with ``#``-prefixed
    metadata headers.  Data rows contain wavelength (vacuum, nm), 50 shot
    intensities, median, and mean columns.

    The first data row is typically a detector-edge artifact and is
    dropped by default.

    Parameters
    ----------
    path : str or Path
        Path to the CCS ``.csv`` file.
    drop_first : bool, optional
        If ``True`` (default), drop the first data row (edge artifact).
    to_vacuum : bool, optional
        If ``True``, convert wavelengths from air to vacuum.  Defaults
        to ``False`` because CCS files already use vacuum wavelengths.

    Examples
    --------
    >>> reader = ChemCamReader("bir1a.csv")
    >>> spec = reader.spectrum          # mean intensities, vacuum λ
    >>> reader.shots.shape              # (N_wavelengths, 50)
    """

    def __init__(
        self,
        path: str | Path,
        drop_first: bool = True,
        to_vacuum: bool = False,
    ):
        self.path = Path(path)
        self._drop_first = drop_first
        self._to_vacuum = to_vacuum
        self._metadata: Dict[str, str] = {}
        self._wavelengths: np.ndarray
        self._shots: np.ndarray
        self._median: np.ndarray
        self._mean: np.ndarray
        self._load()

    def _load(self) -> None:
        metadata: Dict[str, str] = {}
        data_lines: list[str] = []

        with open(self.path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("# wave"):
                    # Column header line – skip
                    continue
                if stripped.startswith("#"):
                    # Metadata: "# KEY = VALUE , , ..."
                    body = stripped.lstrip("# ")
                    if "=" in body:
                        key, _, val = body.partition("=")
                        # Strip trailing comma-padding
                        val = val.split(",")[0].strip()
                        metadata[key.strip()] = val
                    continue
                if stripped:
                    data_lines.append(stripped)

        self._metadata = metadata

        # Parse data rows: wavelength, shot1..shot50, median, mean
        rows = []
        for line in data_lines:
            vals = [float(x) for x in line.split(",")]
            rows.append(vals)

        arr = np.array(rows)
        if self._drop_first:
            arr = arr[1:]

        self._wavelengths = arr[:, 0]
        self._shots = arr[:, 1:51]
        self._median = arr[:, 51]
        self._mean = arr[:, 52]

        if self._to_vacuum:
            self._wavelengths = air_to_vacuum(self._wavelengths)

    @property
    def metadata(self) -> Dict[str, str]:
        """Acquisition metadata from ``#``-prefixed header lines."""
        return self._metadata

    @property
    def wavelengths(self) -> np.ndarray:
        """Wavelength array (vacuum nm by default)."""
        return self._wavelengths

    @property
    def shots(self) -> np.ndarray:
        """Per-shot intensity matrix, shape ``(n_wavelengths, 50)``."""
        return self._shots

    @property
    def median(self) -> np.ndarray:
        """Median intensity across shots."""
        return self._median

    @property
    def mean(self) -> np.ndarray:
        """Mean intensity across shots."""
        return self._mean

    @property
    def spectrum(self) -> Spectrum:
        """Spectrum constructed from mean intensities."""
        return Spectrum(self._wavelengths.copy(), self._mean.copy())
