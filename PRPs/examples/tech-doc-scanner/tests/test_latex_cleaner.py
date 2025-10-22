"""Tests for LaTeX cleaner."""

import pytest
from tech_doc_scanner.latex import LaTeXCleaner


class TestLaTeXCleaner:
    """Test suite for LaTeXCleaner class."""

    def test_balance_braces_add_closing(self):
        """Test adding missing closing braces."""
        cleaner = LaTeXCleaner()
        result = cleaner.balance_braces(r"\frac{1{2")
        assert result.count("{") == result.count("}")

    def test_balance_braces_already_balanced(self):
        """Test that already balanced braces are unchanged."""
        cleaner = LaTeXCleaner()
        input_str = r"\frac{1}{2}"
        result = cleaner.balance_braces(input_str)
        assert result == input_str

    def test_fix_array_environments_add_end(self):
        """Test adding missing \\end for array environments."""
        cleaner = LaTeXCleaner()
        result = cleaner.fix_array_environments(r"\begin{array} x")
        assert r"\end{array}" in result

    def test_fix_left_right_add_right(self):
        """Test adding missing \\right."""
        cleaner = LaTeXCleaner()
        result = cleaner.fix_left_right(r"\left( x")
        assert r"\right." in result

    def test_clean_whitespace_multiple_spaces(self):
        """Test removing multiple spaces."""
        cleaner = LaTeXCleaner()
        result = cleaner.clean_whitespace("x    y")
        assert result == "x y"

    def test_fix_common_errors_trailing_subscript(self):
        """Test removing trailing subscript."""
        cleaner = LaTeXCleaner()
        result = cleaner.fix_common_errors("x_")
        assert not result.endswith("_")

    def test_clean_iterative(self):
        """Test that clean applies fixes iteratively."""
        cleaner = LaTeXCleaner()
        input_str = r"\frac{1{2  _"
        result = cleaner.clean(input_str)
        assert result.count("{") == result.count("}")
        assert not result.endswith("_")

    def test_clean_empty_input(self):
        """Test clean with empty input."""
        cleaner = LaTeXCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) is None
