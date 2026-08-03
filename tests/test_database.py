#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Smoke + behaviour tests for libskit.Database (Phase-1 wrapper)."""
from __future__ import annotations

import pytest

import libskit
from libskit.database import Database


def test_bundled_round_trip():
    db = libskit.Database.bundled()
    assert db.is_active, "import-time auto-loader should mark bundled as active"
    db.activate()  # must be idempotent
    assert db.is_active


def test_from_path_uses_bundled_dir():
    db = libskit.Database.from_path(libskit.Database.bundled().path)
    db.activate()
    assert db.is_active
    # Round-tripping back to the bundled accessor should not require a reload.
    db2 = libskit.Database.bundled()
    assert db2.is_active


def test_from_path_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        Database.from_path(tmp_path / "does_not_exist")


def test_filtered_stark_subsets_lines():
    db = libskit.Database.filtered_stark(top_n=3, verbose=0)
    db.activate()
    assert db.is_active
    # The new database is in a temp dir distinct from bundled.
    assert db.path != libskit.Database.bundled().path
    # Restore bundled at end so other tests get the full DB.
    libskit.Database.bundled().activate()


def test_model_accepts_db_kwarg_and_works_without_it():
    db = libskit.Database.bundled()
    m_with = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3, db=db)
    m_without = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
    assert m_with.temperature == m_without.temperature
    # Wrapper IS-A pybind base, so isinstance checks against the public class
    # surface succeed.
    assert isinstance(m_with, libskit.LTEModel)


def test_layers_and_time_integrated_accept_db():
    db = libskit.Database.bundled()
    layers = libskit.LTELayers(
        [{"Fe": 1.0}], [10000.0], [1e23], [1e-3], db=db
    )
    assert layers is not None
    ti = libskit.LTETimeIntegrated(
        {"Fe": 1.0},
        10000.0, 10000.0, 1e-6,
        1e23, 1e23, 1e-6,
        1e-3, 1e-3, 1e-6,
        1e-7,
        db=db,
    )
    assert ti is not None


def test_database_repr_contains_description():
    db = libskit.Database.bundled()
    assert "bundled" in repr(db)
