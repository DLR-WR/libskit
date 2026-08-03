#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Tests for libskit.readers.

Aryelle .ary files are a proprietary binary format and require sample
data; only the ChemCam CSV reader is exercised here against a minimal
synthetic CSV that mimics the MCCOPS PDS layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libskit.readers import ChemCamReader
from libskit.spectrum import Spectrum


def _write_synthetic_ccs_csv(path: Path, n_rows: int = 20) -> None:
    """Write a minimal CSV mimicking the MCCOPS ChemCam CCS layout."""
    rng = np.random.default_rng(0)
    wl = np.linspace(240.0, 850.0, n_rows)
    shots = rng.uniform(10.0, 100.0, size=(n_rows, 50))
    median = np.median(shots, axis=1)
    mean = shots.mean(axis=1)

    header_cols = (
        "# wave," + ",".join(f"shot{i+1}" for i in range(50)) + ",median,mean"
    )
    metadata = [
        "# TARGET = bir1a , , , , ",
        "# DATE = 2026-01-01 , , , , ",
    ]

    with open(path, "w") as f:
        for line in metadata:
            f.write(line + "\n")
        f.write(header_cols + "\n")
        for i in range(n_rows):
            row = [f"{wl[i]:.4f}"]
            row.extend(f"{v:.4f}" for v in shots[i])
            row.append(f"{median[i]:.4f}")
            row.append(f"{mean[i]:.4f}")
            f.write(",".join(row) + "\n")


@pytest.fixture
def synthetic_ccs(tmp_path: Path) -> Path:
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_ccs_csv(csv_path)
    return csv_path


class TestChemCamReader:
    def test_metadata_parsed(self, synthetic_ccs: Path) -> None:
        reader = ChemCamReader(synthetic_ccs, drop_first=False)
        assert reader.metadata["TARGET"] == "bir1a"
        assert reader.metadata["DATE"] == "2026-01-01"

    def test_shape_arrays(self, synthetic_ccs: Path) -> None:
        reader = ChemCamReader(synthetic_ccs, drop_first=False)
        assert reader.wavelengths.shape == (20,)
        assert reader.shots.shape == (20, 50)
        assert reader.median.shape == (20,)
        assert reader.mean.shape == (20,)

    def test_drop_first_removes_one_row(self, synthetic_ccs: Path) -> None:
        full = ChemCamReader(synthetic_ccs, drop_first=False)
        dropped = ChemCamReader(synthetic_ccs, drop_first=True)
        assert len(dropped.wavelengths) == len(full.wavelengths) - 1
        np.testing.assert_array_equal(dropped.wavelengths, full.wavelengths[1:])

    def test_mean_equals_shot_average(self, synthetic_ccs: Path) -> None:
        reader = ChemCamReader(synthetic_ccs, drop_first=False)
        np.testing.assert_allclose(reader.mean, reader.shots.mean(axis=1),
                                    atol=1e-3)

    def test_spectrum_property_returns_spectrum(self, synthetic_ccs: Path) -> None:
        reader = ChemCamReader(synthetic_ccs, drop_first=False)
        spec = reader.spectrum
        assert isinstance(spec, Spectrum)
        assert len(spec.wavelengths) == len(reader.wavelengths)
        np.testing.assert_array_equal(spec.irradiance, reader.mean)

    def test_to_vacuum_shifts_wavelengths(self, synthetic_ccs: Path) -> None:
        air = ChemCamReader(synthetic_ccs, drop_first=False, to_vacuum=False)
        vac = ChemCamReader(synthetic_ccs, drop_first=False, to_vacuum=True)
        # Vacuum wavelengths > air wavelengths everywhere
        assert np.all(vac.wavelengths > air.wavelengths)
