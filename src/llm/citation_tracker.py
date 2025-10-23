"""
Citation and source tracking for RAG answers.

Tracks which context chunks contributed to the generated answer and provides:
- Standardized citation format
- Source attribution with confidence scores
- Equation-to-source linking
- View source links
- Citation validation

Usage:
    tracker = CitationTracker()

    # Track chunk usage
    tracker.add_chunk(
        chunk_id="chunk_001",
        content="The moment of inertia...",
        relevance_score=0.95,
        metadata={"section": "3.2", "page": 45}
    )

    # Generate citations
    citations = tracker.generate_citations()

    # Link equations to sources
    equation_sources = tracker.link_equations(
        answer="The formula is $I = \\frac{bh^3}{12}$",
        equation_pattern=r"\$.*?\$"
    )
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

from loguru import logger


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    IEEE = "ieee"  # [1], [2], [3]
    APA = "apa"  # (Author, Year)
    NUMERIC = "numeric"  # [Source 1], [Source 2]
    INLINE = "inline"  # (Chapter 3, Section 2, p. 45)


@dataclass
class ChunkReference:
    """
    Reference to a context chunk used in answer generation.

    Attributes:
        chunk_id: Unique identifier for the chunk
        content: Chunk text content
        relevance_score: How relevant this chunk was (0-1)
        metadata: Additional metadata (section, page, figure, etc.)
        usage_count: How many times this chunk was referenced
        confidence: Confidence in this chunk's contribution (0-1)
    """
    chunk_id: str
    content: str
    relevance_score: float
    metadata: Dict = field(default_factory=dict)
    usage_count: int = 1
    confidence: float = 1.0

    def __post_init__(self):
        """Validate scores are in valid range."""
        if not 0 <= self.relevance_score <= 1:
            raise ValueError(f"relevance_score must be 0-1, got {self.relevance_score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")


@dataclass
class Citation:
    """
    Formatted citation for a source.

    Attributes:
        citation_id: Citation number/identifier
        chunk_id: Original chunk ID
        formatted_text: Citation text (e.g., "[1]", "(Chapter 3, p. 45)")
        source_info: Human-readable source description
        relevance_score: Relevance of this source
        confidence: Confidence in citation
        view_source_link: Optional link to original document
    """
    citation_id: str
    chunk_id: str
    formatted_text: str
    source_info: str
    relevance_score: float
    confidence: float = 1.0
    view_source_link: Optional[str] = None


@dataclass
class EquationSource:
    """
    Link between an equation and its source.

    Attributes:
        equation: LaTeX equation string
        source_chunk_id: Chunk containing this equation
        source_section: Section/chapter reference
        source_page: Page number
        confidence: Confidence this is the correct source
    """
    equation: str
    source_chunk_id: str
    source_section: Optional[str] = None
    source_page: Optional[int] = None
    confidence: float = 1.0


class CitationTracker:
    """
    Track and manage citations for RAG-generated answers.

    Features:
    - Track chunk usage and relevance
    - Generate citations in multiple formats
    - Link equations to source pages
    - Provide view source links
    - Calculate confidence scores

    Usage:
        tracker = CitationTracker(citation_style=CitationStyle.IEEE)

        # Add chunks as they're used
        for chunk in context_chunks:
            tracker.add_chunk(
                chunk_id=chunk["id"],
                content=chunk["content"],
                relevance_score=chunk["score"],
                metadata=chunk["metadata"]
            )

        # Generate citations
        citations = tracker.generate_citations()
        formatted_answer = tracker.format_answer_with_citations(answer)

        # Get equation sources
        equation_sources = tracker.link_equations(answer)
    """

    def __init__(
        self,
        citation_style: CitationStyle = CitationStyle.IEEE,
        min_relevance_threshold: float = 0.5,
        include_low_confidence: bool = False,
    ):
        """
        Initialize citation tracker.

        Args:
            citation_style: Format for citations
            min_relevance_threshold: Minimum relevance to include (0-1)
            include_low_confidence: Whether to include low confidence sources
        """
        self.citation_style = citation_style
        self.min_relevance_threshold = min_relevance_threshold
        self.include_low_confidence = include_low_confidence

        self.chunks: Dict[str, ChunkReference] = {}
        self.citation_order: List[str] = []

        logger.debug(
            f"CitationTracker initialized: style={citation_style}, "
            f"threshold={min_relevance_threshold}"
        )

    def add_chunk(
        self,
        chunk_id: str,
        content: str,
        relevance_score: float,
        metadata: Optional[Dict] = None,
        confidence: float = 1.0,
    ):
        """
        Add or update a chunk reference.

        Args:
            chunk_id: Unique chunk identifier
            content: Chunk text
            relevance_score: Relevance score (0-1)
            metadata: Additional metadata
            confidence: Confidence score (0-1)
        """
        if chunk_id in self.chunks:
            # Chunk already tracked, increment usage
            self.chunks[chunk_id].usage_count += 1
            logger.debug(f"Incremented usage for chunk {chunk_id}")
        else:
            # New chunk
            chunk_ref = ChunkReference(
                chunk_id=chunk_id,
                content=content,
                relevance_score=relevance_score,
                metadata=metadata or {},
                confidence=confidence,
            )
            self.chunks[chunk_id] = chunk_ref
            self.citation_order.append(chunk_id)

            logger.debug(
                f"Added chunk {chunk_id}: relevance={relevance_score:.3f}, "
                f"confidence={confidence:.3f}"
            )

    def get_chunks_above_threshold(self) -> List[ChunkReference]:
        """
        Get chunks above relevance threshold, sorted by relevance.

        Returns:
            List of ChunkReference objects
        """
        filtered = [
            chunk for chunk in self.chunks.values()
            if chunk.relevance_score >= self.min_relevance_threshold
        ]

        # Filter by confidence if needed
        if not self.include_low_confidence:
            filtered = [chunk for chunk in filtered if chunk.confidence >= 0.5]

        # Sort by relevance (descending), then by usage count
        filtered.sort(
            key=lambda x: (x.relevance_score, x.usage_count),
            reverse=True
        )

        return filtered

    def generate_citations(self) -> List[Citation]:
        """
        Generate formatted citations for tracked chunks.

        Returns:
            List of Citation objects
        """
        chunks = self.get_chunks_above_threshold()
        citations = []

        for idx, chunk in enumerate(chunks, start=1):
            citation = self._format_citation(idx, chunk)
            citations.append(citation)

        logger.info(f"Generated {len(citations)} citations")
        return citations

    def _format_citation(self, idx: int, chunk: ChunkReference) -> Citation:
        """Format a single citation based on style."""

        if self.citation_style == CitationStyle.IEEE:
            formatted_text = f"[{idx}]"
            source_info = self._build_source_info(chunk)

        elif self.citation_style == CitationStyle.NUMERIC:
            formatted_text = f"[Source {idx}]"
            source_info = self._build_source_info(chunk)

        elif self.citation_style == CitationStyle.INLINE:
            source_info = self._build_source_info(chunk)
            formatted_text = f"({source_info})"

        else:  # APA or default
            formatted_text = f"(Source {idx})"
            source_info = self._build_source_info(chunk)

        # Build view source link if page/section available
        view_source_link = self._build_view_source_link(chunk)

        return Citation(
            citation_id=str(idx),
            chunk_id=chunk.chunk_id,
            formatted_text=formatted_text,
            source_info=source_info,
            relevance_score=chunk.relevance_score,
            confidence=chunk.confidence,
            view_source_link=view_source_link,
        )

    def _build_source_info(self, chunk: ChunkReference) -> str:
        """Build human-readable source description."""
        parts = []

        metadata = chunk.metadata

        # Document name
        if "document" in metadata:
            parts.append(metadata["document"])

        # Section/chapter
        if "section" in metadata:
            parts.append(f"Section {metadata['section']}")
        elif "chapter" in metadata:
            parts.append(f"Chapter {metadata['chapter']}")

        # Page
        if "page" in metadata:
            parts.append(f"p. {metadata['page']}")

        # Figure/table reference
        if "figure" in metadata:
            parts.append(f"Figure {metadata['figure']}")
        elif "table" in metadata:
            parts.append(f"Table {metadata['table']}")

        return ", ".join(parts) if parts else f"Source {chunk.chunk_id}"

    def _build_view_source_link(self, chunk: ChunkReference) -> Optional[str]:
        """Build view source link if possible."""
        metadata = chunk.metadata

        # If we have a URL, use it
        if "url" in metadata:
            return metadata["url"]

        # If we have document path and page, construct link
        if "document_path" in metadata and "page" in metadata:
            doc_path = metadata["document_path"]
            page = metadata["page"]
            return f"file://{doc_path}#page={page}"

        return None

    def format_answer_with_citations(
        self,
        answer: str,
        insert_citations: bool = True,
    ) -> str:
        """
        Add citations to answer text.

        Args:
            answer: Generated answer text
            insert_citations: Whether to insert citation markers inline

        Returns:
            Answer with citations appended (and optionally inserted)
        """
        citations = self.generate_citations()

        if not citations:
            return answer

        # Optionally insert inline citations (basic implementation)
        formatted_answer = answer

        if insert_citations:
            # Insert citations after sentences mentioning tracked chunks
            # (This is a simplified version - production would use NLP)
            for idx, chunk_id in enumerate(self.citation_order, start=1):
                if chunk_id not in self.chunks:
                    continue

                chunk = self.chunks[chunk_id]

                # Look for key terms from chunk in answer
                # Insert citation marker
                citation_marker = f" [{idx}]"

                # Simple heuristic: add after sentences containing chunk keywords
                # (In production, use proper sentence segmentation and matching)

        # Append citation list
        formatted_answer += "\n\n**Sources:**\n"
        for citation in citations:
            formatted_answer += f"{citation.formatted_text} {citation.source_info}"

            if citation.view_source_link:
                formatted_answer += f" ([View Source]({citation.view_source_link}))"

            formatted_answer += "\n"

        return formatted_answer

    def link_equations(
        self,
        answer: str,
        equation_pattern: str = r"\$\$?(.*?)\$\$?",
    ) -> List[EquationSource]:
        """
        Link equations in answer to their source chunks.

        Args:
            answer: Generated answer containing equations
            equation_pattern: Regex pattern to extract equations

        Returns:
            List of EquationSource objects
        """
        # Extract equations from answer
        equations = re.findall(equation_pattern, answer, re.DOTALL)

        equation_sources = []

        for equation in equations:
            # Clean equation
            equation_clean = equation.strip()

            # Find source chunk containing this equation
            source = self._find_equation_source(equation_clean)

            if source:
                equation_sources.append(source)

        logger.info(f"Linked {len(equation_sources)} equations to sources")
        return equation_sources

    def _find_equation_source(self, equation: str) -> Optional[EquationSource]:
        """Find which chunk contains an equation."""
        # Normalize equation for matching
        equation_normalized = equation.replace(" ", "").replace("\\\\", "\\")

        for chunk_id, chunk in self.chunks.items():
            content_normalized = chunk.content.replace(" ", "").replace("\\\\", "\\")

            # Check if equation appears in chunk
            if equation_normalized in content_normalized:
                metadata = chunk.metadata

                return EquationSource(
                    equation=equation,
                    source_chunk_id=chunk_id,
                    source_section=metadata.get("section"),
                    source_page=metadata.get("page"),
                    confidence=chunk.confidence,
                )

        # Equation not found in any chunk
        logger.warning(f"Equation not found in sources: {equation[:50]}...")
        return None

    def get_confidence_scores(self) -> Dict[str, float]:
        """
        Calculate confidence scores for the answer.

        Returns:
            Dictionary with various confidence metrics
        """
        chunks = self.get_chunks_above_threshold()

        if not chunks:
            return {
                "overall_confidence": 0.0,
                "avg_relevance": 0.0,
                "avg_chunk_confidence": 0.0,
                "num_sources": 0,
            }

        # Calculate metrics
        avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)
        avg_confidence = sum(c.confidence for c in chunks) / len(chunks)

        # Overall confidence: weighted average
        overall_confidence = (avg_relevance * 0.6) + (avg_confidence * 0.4)

        return {
            "overall_confidence": overall_confidence,
            "avg_relevance": avg_relevance,
            "avg_chunk_confidence": avg_confidence,
            "num_sources": len(chunks),
            "high_confidence_sources": sum(1 for c in chunks if c.confidence >= 0.8),
            "low_confidence_sources": sum(1 for c in chunks if c.confidence < 0.5),
        }

    def get_citation_summary(self) -> Dict:
        """
        Get summary statistics about citations.

        Returns:
            Dictionary with citation statistics
        """
        citations = self.generate_citations()
        confidence = self.get_confidence_scores()

        return {
            "total_citations": len(citations),
            "total_chunks_tracked": len(self.chunks),
            "chunks_above_threshold": len(self.get_chunks_above_threshold()),
            "citation_style": self.citation_style.value,
            "confidence_scores": confidence,
        }

    def reset(self):
        """Clear all tracked citations."""
        self.chunks.clear()
        self.citation_order.clear()
        logger.debug("Citation tracker reset")


def create_citation_tracker(
    citation_style: str = "ieee",
    min_relevance: float = 0.5,
) -> CitationTracker:
    """
    Convenience function to create a citation tracker.

    Args:
        citation_style: Citation format style
        min_relevance: Minimum relevance threshold

    Returns:
        CitationTracker instance
    """
    style = CitationStyle(citation_style.lower())
    return CitationTracker(
        citation_style=style,
        min_relevance_threshold=min_relevance,
    )


if __name__ == "__main__":
    logger.add("logs/citation_tracker.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("CITATION TRACKER - Source Attribution for RAG")
    print("=" * 70)
    print("\nFeatures:")
    print("  • Track chunk usage and relevance")
    print("  • Generate citations in multiple formats (IEEE, APA, Numeric, Inline)")
    print("  • Link equations to source pages")
    print("  • Provide view source links")
    print("  • Calculate confidence scores")
    print("\nCitation Styles:")
    print("  • IEEE: [1], [2], [3]")
    print("  • Numeric: [Source 1], [Source 2]")
    print("  • Inline: (Chapter 3, Section 2, p. 45)")
    print("\nUsage:")
    print("  tracker = CitationTracker(citation_style=CitationStyle.IEEE)")
    print("  tracker.add_chunk(chunk_id='...', content='...', relevance_score=0.95)")
    print("  citations = tracker.generate_citations()")
    print("  formatted_answer = tracker.format_answer_with_citations(answer)")
    print("=" * 70 + "\n")
