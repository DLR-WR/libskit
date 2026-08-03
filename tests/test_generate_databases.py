#
# Copyright (c) 2025 Christoph H. Egerland
# SPDX-License-Identifier: MIT
#
"""Tests for libskit.generate_databases utilities (pure helpers only)."""

from __future__ import annotations

import pytest

from libskit.generate_databases import ACCURACY_MAP, acc_code_to_fraction


class TestAccuracyCodeMapping:
    @pytest.mark.parametrize(
        "code,expected",
        [("AAA", 0.003), ("AA", 0.01), ("A+", 0.02), ("A", 0.03),
         ("B+", 0.07), ("B", 0.10), ("C+", 0.18), ("C", 0.25),
         ("D+", 0.40), ("D", 0.50), ("E", 0.50)],
    )
    def test_known_codes(self, code: str, expected: float) -> None:
        assert acc_code_to_fraction(code) == expected

    def test_strips_trailing_prime(self) -> None:
        assert acc_code_to_fraction("A'") == ACCURACY_MAP["A"]
        assert acc_code_to_fraction("AA'") == ACCURACY_MAP["AA"]

    def test_strips_whitespace(self) -> None:
        assert acc_code_to_fraction("  B  ") == ACCURACY_MAP["B"]

    def test_unknown_code_returns_conservative_default(self) -> None:
        assert acc_code_to_fraction("ZZZ") == 0.50

    def test_empty_string_returns_default(self) -> None:
        assert acc_code_to_fraction("") == 0.50
        assert acc_code_to_fraction("   ") == 0.50

    def test_non_string_returns_default(self) -> None:
        # NIST CSVs sometimes have NaN floats in this column
        assert acc_code_to_fraction(float("nan")) == 0.50
        assert acc_code_to_fraction(0.0) == 0.50
