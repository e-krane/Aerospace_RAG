"""
Tests for citation tracker.

Tests:
- Citation tracking and management
- Multiple citation styles
- Equation-to-source linking
- Confidence scoring
- Citation formatting
- Source attribution
"""

import pytest

from src.llm.citation_tracker import (
    CitationTracker,
    Citation,
    ChunkReference,
    EquationSource,
    CitationStyle,
    create_citation_tracker,
)


class TestChunkReference:
    """Test ChunkReference dataclass."""

    def test_chunk_reference_creation(self):
        """Test creating a chunk reference."""
        chunk = ChunkReference(
            chunk_id="chunk_001",
            content="The moment of inertia is...",
            relevance_score=0.95,
            metadata={"section": "3.2", "page": 45},
        )

        assert chunk.chunk_id == "chunk_001"
        assert chunk.relevance_score == 0.95
        assert chunk.usage_count == 1
        assert chunk.confidence == 1.0

    def test_chunk_reference_validation(self):
        """Test validation of score ranges."""
        # Invalid relevance score
        with pytest.raises(ValueError):
            ChunkReference(
                chunk_id="test",
                content="test",
                relevance_score=1.5,  # >1
            )

        # Invalid confidence score
        with pytest.raises(ValueError):
            ChunkReference(
                chunk_id="test",
                content="test",
                relevance_score=0.9,
                confidence=-0.1,  # <0
            )


class TestCitationTracker:
    """Test CitationTracker functionality."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = CitationTracker(
            citation_style=CitationStyle.IEEE,
            min_relevance_threshold=0.6,
        )

        assert tracker.citation_style == CitationStyle.IEEE
        assert tracker.min_relevance_threshold == 0.6
        assert len(tracker.chunks) == 0

    def test_add_chunk(self):
        """Test adding chunks."""
        tracker = CitationTracker()

        tracker.add_chunk(
            chunk_id="chunk_001",
            content="Test content",
            relevance_score=0.9,
            metadata={"page": 10},
        )

        assert len(tracker.chunks) == 1
        assert "chunk_001" in tracker.chunks
        assert tracker.chunks["chunk_001"].relevance_score == 0.9

    def test_add_duplicate_chunk(self):
        """Test adding same chunk multiple times increments usage."""
        tracker = CitationTracker()

        tracker.add_chunk("chunk_001", "Content", 0.9)
        tracker.add_chunk("chunk_001", "Content", 0.9)

        assert len(tracker.chunks) == 1
        assert tracker.chunks["chunk_001"].usage_count == 2

    def test_get_chunks_above_threshold(self):
        """Test filtering chunks by relevance threshold."""
        tracker = CitationTracker(min_relevance_threshold=0.7)

        tracker.add_chunk("chunk_001", "Content 1", 0.9)
        tracker.add_chunk("chunk_002", "Content 2", 0.6)  # Below threshold
        tracker.add_chunk("chunk_003", "Content 3", 0.8)

        chunks = tracker.get_chunks_above_threshold()

        assert len(chunks) == 2
        assert all(c.relevance_score >= 0.7 for c in chunks)

    def test_chunks_sorted_by_relevance(self):
        """Test chunks are sorted by relevance."""
        tracker = CitationTracker()

        tracker.add_chunk("chunk_001", "Content 1", 0.7)
        tracker.add_chunk("chunk_002", "Content 2", 0.9)
        tracker.add_chunk("chunk_003", "Content 3", 0.8)

        chunks = tracker.get_chunks_above_threshold()

        assert chunks[0].relevance_score == 0.9
        assert chunks[1].relevance_score == 0.8
        assert chunks[2].relevance_score == 0.7


class TestCitationGeneration:
    """Test citation generation and formatting."""

    def test_generate_citations_ieee_style(self):
        """Test IEEE style citations."""
        tracker = CitationTracker(citation_style=CitationStyle.IEEE)

        tracker.add_chunk(
            "chunk_001",
            "Content",
            0.9,
            metadata={"section": "3.2", "page": 45}
        )
        tracker.add_chunk(
            "chunk_002",
            "Content",
            0.8,
            metadata={"section": "4.1", "page": 67}
        )

        citations = tracker.generate_citations()

        assert len(citations) == 2
        assert citations[0].formatted_text == "[1]"
        assert citations[1].formatted_text == "[2]"
        assert "Section 3.2" in citations[0].source_info
        assert "p. 45" in citations[0].source_info

    def test_generate_citations_numeric_style(self):
        """Test numeric style citations."""
        tracker = CitationTracker(citation_style=CitationStyle.NUMERIC)

        tracker.add_chunk("chunk_001", "Content", 0.9)

        citations = tracker.generate_citations()

        assert citations[0].formatted_text == "[Source 1]"

    def test_generate_citations_inline_style(self):
        """Test inline style citations."""
        tracker = CitationTracker(citation_style=CitationStyle.INLINE)

        tracker.add_chunk(
            "chunk_001",
            "Content",
            0.9,
            metadata={"section": "3.2", "page": 45}
        )

        citations = tracker.generate_citations()

        # Inline style wraps source info in parentheses
        assert "(" in citations[0].formatted_text
        assert ")" in citations[0].formatted_text
        assert "Section 3.2" in citations[0].formatted_text

    def test_source_info_building(self):
        """Test building human-readable source info."""
        tracker = CitationTracker()

        # With section and page
        tracker.add_chunk(
            "chunk_001",
            "Content",
            0.9,
            metadata={"section": "3.2", "page": 45}
        )

        # With chapter and figure
        tracker.add_chunk(
            "chunk_002",
            "Content",
            0.9,
            metadata={"chapter": "5", "figure": "5.3"}
        )

        citations = tracker.generate_citations()

        assert "Section 3.2" in citations[0].source_info
        assert "p. 45" in citations[0].source_info

        assert "Chapter 5" in citations[1].source_info
        assert "Figure 5.3" in citations[1].source_info

    def test_view_source_link_from_url(self):
        """Test view source link from URL metadata."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "Content",
            0.9,
            metadata={"url": "https://example.com/doc.pdf#page=45"}
        )

        citations = tracker.generate_citations()

        assert citations[0].view_source_link == "https://example.com/doc.pdf#page=45"

    def test_view_source_link_from_file_path(self):
        """Test view source link from file path and page."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "Content",
            0.9,
            metadata={
                "document_path": "/path/to/document.pdf",
                "page": 45
            }
        )

        citations = tracker.generate_citations()

        assert "file:///path/to/document.pdf#page=45" in citations[0].view_source_link


class TestAnswerFormatting:
    """Test answer formatting with citations."""

    def test_format_answer_with_citations(self):
        """Test adding citations to answer."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "Content",
            0.9,
            metadata={"section": "3.2"}
        )

        answer = "The moment of inertia is calculated using the formula."
        formatted = tracker.format_answer_with_citations(answer)

        assert "The moment of inertia" in formatted
        assert "**Sources:**" in formatted
        assert "[1]" in formatted
        assert "Section 3.2" in formatted

    def test_format_answer_no_citations(self):
        """Test formatting when no citations available."""
        tracker = CitationTracker()

        answer = "Test answer"
        formatted = tracker.format_answer_with_citations(answer)

        # Should return original answer unchanged
        assert formatted == answer


class TestEquationLinking:
    """Test equation-to-source linking."""

    def test_link_equations_basic(self):
        """Test linking equations to sources."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "The formula is $I = \\frac{bh^3}{12}$ for rectangular sections.",
            0.9,
            metadata={"section": "3.2", "page": 45}
        )

        answer = "The moment of inertia is $I = \\frac{bh^3}{12}$."

        equation_sources = tracker.link_equations(answer)

        assert len(equation_sources) == 1
        assert "I = \\frac{bh^3}{12}" in equation_sources[0].equation
        assert equation_sources[0].source_chunk_id == "chunk_001"
        assert equation_sources[0].source_section == "3.2"
        assert equation_sources[0].source_page == 45

    def test_link_equations_double_dollar(self):
        """Test linking equations with $$ delimiters."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "The stress formula: $$\\sigma = \\frac{F}{A}$$",
            0.9,
        )

        answer = "Stress is calculated as $$\\sigma = \\frac{F}{A}$$."

        equation_sources = tracker.link_equations(answer)

        assert len(equation_sources) == 1
        assert "\\sigma = \\frac{F}{A}" in equation_sources[0].equation

    def test_link_equations_not_found(self):
        """Test linking when equation not in sources."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "Different content without the equation",
            0.9,
        )

        answer = "The formula is $E = mc^2$."

        equation_sources = tracker.link_equations(answer)

        # Should return empty list or handle missing equations
        assert len(equation_sources) == 0

    def test_link_multiple_equations(self):
        """Test linking multiple equations."""
        tracker = CitationTracker()

        tracker.add_chunk(
            "chunk_001",
            "Stress: $\\sigma = F/A$ and strain: $\\epsilon = \\Delta L / L_0$",
            0.9,
        )

        answer = "We have $\\sigma = F/A$ and $\\epsilon = \\Delta L / L_0$."

        equation_sources = tracker.link_equations(answer)

        assert len(equation_sources) == 2


class TestConfidenceScoring:
    """Test confidence scoring."""

    def test_confidence_scores_basic(self):
        """Test basic confidence score calculation."""
        tracker = CitationTracker()

        tracker.add_chunk("chunk_001", "Content", 0.9, confidence=0.95)
        tracker.add_chunk("chunk_002", "Content", 0.8, confidence=0.85)

        scores = tracker.get_confidence_scores()

        assert "overall_confidence" in scores
        assert "avg_relevance" in scores
        assert "avg_chunk_confidence" in scores
        assert "num_sources" in scores

        assert scores["num_sources"] == 2
        assert abs(scores["avg_relevance"] - 0.85) < 0.001  # (0.9 + 0.8) / 2

    def test_confidence_scores_empty(self):
        """Test confidence scores with no chunks."""
        tracker = CitationTracker()

        scores = tracker.get_confidence_scores()

        assert scores["overall_confidence"] == 0.0
        assert scores["num_sources"] == 0

    def test_high_low_confidence_counts(self):
        """Test counting high and low confidence sources."""
        # Include low confidence sources in calculations
        tracker = CitationTracker(include_low_confidence=True)

        tracker.add_chunk("chunk_001", "Content", 0.9, confidence=0.9)  # High
        tracker.add_chunk("chunk_002", "Content", 0.8, confidence=0.4)  # Low
        tracker.add_chunk("chunk_003", "Content", 0.85, confidence=0.85)  # High

        scores = tracker.get_confidence_scores()

        assert scores["high_confidence_sources"] == 2
        assert scores["low_confidence_sources"] == 1


class TestCitationSummary:
    """Test citation summary statistics."""

    def test_citation_summary(self):
        """Test getting citation summary."""
        tracker = CitationTracker(min_relevance_threshold=0.7)

        tracker.add_chunk("chunk_001", "Content", 0.9)
        tracker.add_chunk("chunk_002", "Content", 0.6)  # Below threshold
        tracker.add_chunk("chunk_003", "Content", 0.8)

        summary = tracker.get_citation_summary()

        assert summary["total_citations"] == 2  # Only above threshold
        assert summary["total_chunks_tracked"] == 3
        assert summary["chunks_above_threshold"] == 2
        assert summary["citation_style"] == "ieee"
        assert "confidence_scores" in summary


class TestTrackerReset:
    """Test tracker reset functionality."""

    def test_reset(self):
        """Test resetting tracker clears all data."""
        tracker = CitationTracker()

        tracker.add_chunk("chunk_001", "Content", 0.9)
        tracker.add_chunk("chunk_002", "Content", 0.8)

        assert len(tracker.chunks) == 2

        tracker.reset()

        assert len(tracker.chunks) == 0
        assert len(tracker.citation_order) == 0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_create_citation_tracker(self):
        """Test convenience function creates tracker."""
        tracker = create_citation_tracker(
            citation_style="ieee",
            min_relevance=0.6
        )

        assert isinstance(tracker, CitationTracker)
        assert tracker.citation_style == CitationStyle.IEEE
        assert tracker.min_relevance_threshold == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
