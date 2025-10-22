"""Extended tests for LaTeX validator."""

import pytest
from tech_doc_scanner.latex import LaTeXValidator


class TestLaTeXValidatorExtended:
    """Extended test suite for LaTeXValidator class."""

    def test_validate_unbalanced_environments(self):
        """Test detection of unbalanced environments."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate(r"\begin{array} x y")
        assert not is_valid
        assert "environment" in error.lower()

    def test_validate_unbalanced_left_right(self):
        """Test detection of unbalanced left/right."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate(r"\left( x")
        assert not is_valid
        assert "left" in error.lower() or "right" in error.lower()

    def test_validate_infinite_pattern(self):
        """Test detection of infinite patterns."""
        validator = LaTeXValidator()
        long_pattern = "x" * 150
        is_valid, error = validator.validate(long_pattern)
        assert not is_valid
        assert "infinite" in error.lower() or "pattern" in error.lower()

    def test_validate_too_long(self):
        """Test detection of excessively long LaTeX."""
        validator = LaTeXValidator()
        long_latex = r"\frac{1}{2} " * 2000
        is_valid, error = validator.validate(long_latex)
        assert not is_valid
        assert "long" in error.lower()

    def test_validate_none_input(self):
        """Test validation with None input."""
        validator = LaTeXValidator()
        is_valid, error = validator.validate(None)
        assert not is_valid
        assert "Empty" in error or "Invalid" in error
