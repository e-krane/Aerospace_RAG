"""Extended tests for LaTeX cleaner."""

import pytest
from tech_doc_scanner.latex import LaTeXCleaner


class TestLaTeXCleanerExtended:
    """Extended test suite for LaTeXCleaner class."""

    def test_balance_braces_remove_closing(self):
        """Test removing extra closing braces."""
        cleaner = LaTeXCleaner()
        result = cleaner.balance_braces(r"\frac{1}{2}}")
        assert result.count("{") == result.count("}")

    def test_fix_array_environments_already_closed(self):
        """Test that already closed environments are unchanged."""
        cleaner = LaTeXCleaner()
        input_str = r"\begin{array}{cc} a & b \end{array}"
        result = cleaner.fix_array_environments(input_str)
        assert result == input_str

    def test_fix_left_right_already_balanced(self):
        """Test that already balanced left/right are unchanged."""
        cleaner = LaTeXCleaner()
        input_str = r"\left( x \right)"
        result = cleaner.fix_left_right(input_str)
        assert result == input_str

    def test_remove_infinite_patterns_actual_pattern(self):
        """Test removal of actual infinite pattern."""
        cleaner = LaTeXCleaner()
        input_str = "x" * 120
        result = cleaner.remove_infinite_patterns(input_str)
        assert len(result) < len(input_str)

    def test_fix_common_errors_trailing_superscript(self):
        """Test removing trailing superscript."""
        cleaner = LaTeXCleaner()
        result = cleaner.fix_common_errors("x^")
        assert not result.endswith("^")

    def test_fix_common_errors_trailing_backslash(self):
        """Test removing trailing backslash."""
        cleaner = LaTeXCleaner()
        result = cleaner.fix_common_errors("x \\")
        assert not result.endswith("\\")

    def test_clean_convergence(self):
        """Test that clean converges after max iterations."""
        cleaner = LaTeXCleaner()
        # Create problematic input
        input_str = "\\frac{1{2  _  ^  \\"
        result = cleaner.clean(input_str)
        # Should terminate even if not perfect
        assert result is not None
        assert len(result) > 0
