"""Tests for LaTeX validator."""

import pytest
from tech_doc_scanner.latex import LaTeXValidator


class TestLaTeXValidator:
    """Test suite for LaTeXValidator class."""

    def test_validate_empty_input(self):
        """Test validation with empty input."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate("")
        assert not is_valid
        assert "Empty" in error

    def test_validate_unbalanced_braces(self):
        """Test detection of unbalanced braces."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate(r"\frac{1{2")
        assert not is_valid
        assert "braces" in error.lower()

    def test_validate_incomplete_command(self):
        """Test detection of incomplete command at end."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate("x_")
        assert not is_valid
        assert "Incomplete" in error

    def test_validate_valid_simple(self):
        """Test validation of simple valid LaTeX."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate(r"\frac{1}{2}")
        assert is_valid
        assert error is None

    def test_validate_valid_complex(self):
        """Test validation of complex valid LaTeX."""
        validator = LaTeXValidator()
        latex = r"\left( \frac{a + b}{c} \right) = \sqrt{x^2 + y^2}"
        is_valid, error = validator.validate(latex)
        assert is_valid
        assert error is None
