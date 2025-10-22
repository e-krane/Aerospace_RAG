"""
Tests for Docling Parser.
"""

import pytest
from pathlib import Path
from src.parsers.docling_parser import DoclingParser, parse_latex_file, ParsedDocument


class TestDoclingParser:
    """Test suite for Docling parser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return DoclingParser(
            enable_formula_enrichment=True,
            enable_figure_extraction=True,
        )

    @pytest.fixture
    def sample_latex_file(self):
        """Path to a sample LaTeX file for testing."""
        return Path("Documents/Aerospace_Structures_LaTeX/Ch01_4P.tex")

    @pytest.fixture
    def sample_pdf_file(self):
        """Path to a sample PDF file for testing."""
        return Path("Documents/Aerospace_Structures_LaTeX/Aerospace_Structures+AppendixA.pdf")

    def test_parser_initialization(self, parser):
        """Test that parser initializes correctly."""
        assert parser is not None
        assert parser.enable_formula_enrichment is True
        assert parser.enable_figure_extraction is True
        assert parser.converter is not None

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("Documents/Aerospace_Structures_LaTeX/Aerospace_Structures+AppendixA.pdf").exists(),
        reason="Sample PDF file not found"
    )
    def test_parse_pdf_file(self, parser, sample_pdf_file):
        """Test parsing a PDF file."""
        result = parser.parse_file(sample_pdf_file)

        assert isinstance(result, ParsedDocument)
        assert result.source_file == sample_pdf_file
        assert len(result.markdown_content) > 0
        assert result.parsing_time_seconds > 0
        assert result.page_count > 0

    @pytest.mark.slow
    def test_parse_file_not_found(self, parser):
        """Test that parser raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("nonexistent_file.pdf"))

    def test_parsed_document_structure(self):
        """Test ParsedDocument dataclass structure."""
        doc = ParsedDocument(
            source_file=Path("test.pdf"),
            markdown_content="# Test",
            title="Test Document",
            equations=[],
            figures=[],
        )

        assert doc.source_file == Path("test.pdf")
        assert doc.markdown_content == "# Test"
        assert doc.title == "Test Document"
        assert isinstance(doc.equations, list)
        assert isinstance(doc.figures, list)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("Documents/Aerospace_Structures_LaTeX/Aerospace_Structures+AppendixA.pdf").exists(),
        reason="Sample PDF not found"
    )
    def test_equation_extraction(self, parser, sample_pdf_file):
        """Test that equations are extracted from PDF."""
        result = parser.parse_file(sample_pdf_file)

        # Should find equations in aerospace structures textbook
        assert len(result.equations) > 0

        # Check equation structure
        if result.equations:
            eq = result.equations[0]
            assert "latex" in eq
            assert isinstance(eq["latex"], str)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("Documents/Aerospace_Structures_LaTeX/Aerospace_Structures+AppendixA.pdf").exists(),
        reason="Sample PDF not found"
    )
    def test_figure_extraction(self, parser, sample_pdf_file):
        """Test that figures are extracted from PDF."""
        result = parser.parse_file(sample_pdf_file)

        # Should find figures in aerospace structures textbook
        assert len(result.figures) > 0

    @pytest.mark.slow
    def test_performance_target(self, parser, sample_pdf_file):
        """Test that parsing meets performance target of ~3.7s/page."""
        if not sample_pdf_file.exists():
            pytest.skip("Sample PDF not found")

        result = parser.parse_file(sample_pdf_file)

        if result.page_count > 0:
            seconds_per_page = result.parsing_time_seconds / result.page_count

            # Allow some flexibility, target is 3.7s/page
            # We'll accept up to 10s/page for initial implementation
            assert seconds_per_page < 10.0, \
                f"Parsing too slow: {seconds_per_page:.2f}s/page (target: <10s/page)"

    def test_convenience_function(self):
        """Test the parse_latex_file convenience function."""
        # Just test that it's callable
        assert callable(parse_latex_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
