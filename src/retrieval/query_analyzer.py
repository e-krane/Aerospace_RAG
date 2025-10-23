"""
Query analysis and routing for intelligent retrieval.

Classifies queries and adjusts fusion parameters dynamically.
"""

from typing import Dict, List, Literal
from dataclasses import dataclass
import re

from loguru import logger


QueryType = Literal["factual", "conceptual", "equation", "procedural", "exploratory"]


@dataclass
class QueryAnalysis:
    """Analysis result for a query."""

    query: str
    query_type: QueryType
    alpha: float  # Fusion weight (0.0=semantic, 1.0=BM25)
    filters: Dict
    expanded_query: str
    reasoning: str


class QueryAnalyzer:
    """
    Intelligent query analysis and routing.

    Classifies queries and determines optimal retrieval strategy.
    """

    # Technical term patterns
    TECHNICAL_TERMS = {
        "abbreviations": [
            "FEM",
            "CFD",
            "HNSW",
            "BM25",
            "RRF",
            "MTEB",
            "LaTeX",
            "PDF",
        ],
        "aerospace": [
            "fuselage",
            "wing",
            "airfoil",
            "stress",
            "strain",
            "shear",
            "moment",
            "beam",
            "plate",
            "shell",
        ],
        "math": [
            "equation",
            "formula",
            "theorem",
            "proof",
            "derivative",
            "integral",
            "matrix",
            "tensor",
        ],
    }

    # Conceptual keywords
    CONCEPTUAL_KEYWORDS = [
        "explain",
        "why",
        "how",
        "concept",
        "theory",
        "principle",
        "understand",
        "meaning",
        "significance",
    ]

    # Procedural keywords
    PROCEDURAL_KEYWORDS = [
        "calculate",
        "compute",
        "solve",
        "derive",
        "step",
        "procedure",
        "method",
        "algorithm",
        "process",
    ]

    # Exploratory keywords
    EXPLORATORY_KEYWORDS = [
        "overview",
        "survey",
        "introduction",
        "background",
        "review",
        "summary",
        "discuss",
    ]

    def __init__(self):
        """Initialize query analyzer."""
        pass

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query and determine retrieval strategy.

        Args:
            query: User query string

        Returns:
            QueryAnalysis with type, alpha, filters, and reasoning
        """
        query_lower = query.lower()

        # Detect filters
        filters = self._detect_filters(query)

        # Classify query type
        query_type = self._classify_query(query_lower)

        # Determine optimal alpha
        alpha = self._determine_alpha(query_type, query_lower)

        # Expand query with technical terms
        expanded_query = self._expand_query(query)

        # Generate reasoning
        reasoning = self._explain_strategy(query_type, alpha, filters)

        logger.info(f"Query analysis: type={query_type}, alpha={alpha:.2f}")

        return QueryAnalysis(
            query=query,
            query_type=query_type,
            alpha=alpha,
            filters=filters,
            expanded_query=expanded_query,
            reasoning=reasoning,
        )

    def _classify_query(self, query_lower: str) -> QueryType:
        """
        Classify query into one of five types.

        Args:
            query_lower: Lowercased query

        Returns:
            QueryType classification
        """
        # Check for equation queries
        if any(
            keyword in query_lower for keyword in ["equation", "formula", "$$", "latex"]
        ):
            return "equation"

        # Check for procedural queries
        if any(keyword in query_lower for keyword in self.PROCEDURAL_KEYWORDS):
            return "procedural"

        # Check for exploratory queries
        if any(keyword in query_lower for keyword in self.EXPLORATORY_KEYWORDS):
            return "exploratory"

        # Check for conceptual queries
        if any(keyword in query_lower for keyword in self.CONCEPTUAL_KEYWORDS):
            return "conceptual"

        # Default to factual (technical term lookup)
        return "factual"

    def _determine_alpha(self, query_type: QueryType, query_lower: str) -> float:
        """
        Determine optimal fusion weight (alpha).

        Args:
            query_type: Classified query type
            query_lower: Lowercased query

        Returns:
            Alpha value (0.0-1.0)
            - 0.0: Pure semantic search
            - 0.5: Balanced hybrid
            - 1.0: Pure BM25 search
        """
        # Count technical terms
        tech_term_count = sum(
            1
            for category in self.TECHNICAL_TERMS.values()
            for term in category
            if term.lower() in query_lower
        )

        # Base alpha by query type
        base_alpha = {
            "factual": 0.7,  # Favor BM25 for exact terms
            "equation": 0.8,  # Strong BM25 for LaTeX matching
            "procedural": 0.6,  # Balanced with slight BM25 bias
            "conceptual": 0.3,  # Favor semantic understanding
            "exploratory": 0.2,  # Strong semantic for broad queries
        }[query_type]

        # Adjust based on technical term density
        if tech_term_count >= 3:
            base_alpha += 0.1  # More BM25 for technical queries
        elif tech_term_count == 0:
            base_alpha -= 0.1  # More semantic for natural language

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, base_alpha))

    def _detect_filters(self, query: str) -> Dict:
        """
        Detect metadata filters from natural language.

        Args:
            query: User query

        Returns:
            Dictionary of filters
        """
        filters = {}

        # Chapter/section detection
        chapter_match = re.search(r"(?:chapter|section)\s+(\d+)", query, re.IGNORECASE)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            filters["document_id"] = f"chapter_{chapter_num}"

        # Equation filter
        if re.search(r"with\s+equations?", query, re.IGNORECASE):
            filters["has_equations"] = True

        # Page range detection
        page_match = re.search(r"pages?\s+(\d+)(?:\s*-\s*(\d+))?", query, re.IGNORECASE)
        if page_match:
            start_page = int(page_match.group(1))
            end_page = int(page_match.group(2)) if page_match.group(2) else start_page
            filters["page_number"] = {"gte": start_page, "lte": end_page}

        # Figure filter
        if re.search(r"with\s+figures?|diagrams?", query, re.IGNORECASE):
            filters["has_figures"] = True

        if filters:
            logger.info(f"Detected filters: {filters}")

        return filters

    def _expand_query(self, query: str) -> str:
        """
        Expand query with technical term variations.

        Args:
            query: Original query

        Returns:
            Expanded query string
        """
        expanded = query

        # Expand common abbreviations
        abbreviation_map = {
            "FEM": "finite element method",
            "CFD": "computational fluid dynamics",
            "HNSW": "hierarchical navigable small world",
            "BM25": "best matching 25",
            "RRF": "reciprocal rank fusion",
        }

        for abbrev, full in abbreviation_map.items():
            if abbrev in query:
                expanded += f" {full}"

        return expanded

    def _explain_strategy(
        self, query_type: QueryType, alpha: float, filters: Dict
    ) -> str:
        """
        Generate human-readable explanation of retrieval strategy.

        Args:
            query_type: Classified type
            alpha: Fusion weight
            filters: Detected filters

        Returns:
            Explanation string
        """
        strategy_explanation = {
            "factual": "Using BM25-heavy search for precise term matching",
            "equation": "Using strong BM25 for LaTeX equation matching",
            "procedural": "Using balanced hybrid search for step-by-step content",
            "conceptual": "Using semantic-heavy search for conceptual understanding",
            "exploratory": "Using strong semantic search for broad exploration",
        }

        explanation = f"Query type: {query_type}. {strategy_explanation[query_type]}."

        if filters:
            filter_str = ", ".join(f"{k}={v}" for k, v in filters.items())
            explanation += f" Applying filters: {filter_str}."

        explanation += f" Fusion weight: {alpha:.2f} (0=semantic, 1=BM25)."

        return explanation


def route_query(query: str, analyzer: QueryAnalyzer = None) -> QueryAnalysis:
    """
    Convenience function to analyze and route a query.

    Args:
        query: User query string
        analyzer: Optional QueryAnalyzer instance

    Returns:
        QueryAnalysis result
    """
    if analyzer is None:
        analyzer = QueryAnalyzer()

    return analyzer.analyze(query)


if __name__ == "__main__":
    logger.add("logs/query_analyzer.log", rotation="10 MB")

    # Example usage
    analyzer = QueryAnalyzer()

    # Test queries
    test_queries = [
        "What is the stress-strain relationship?",  # Conceptual
        "Calculate the moment of inertia for a rectangular beam",  # Procedural
        "Show me equations for beam deflection",  # Equation
        "FEM analysis of wing structures",  # Factual
        "Overview of aerospace structures in chapter 3",  # Exploratory + Filter
    ]

    print("\n" + "=" * 70)
    print("QUERY ANALYZER EXAMPLES")
    print("=" * 70)

    for query in test_queries:
        analysis = analyzer.analyze(query)
        print(f"\nQuery: {query}")
        print(f"Type: {analysis.query_type}")
        print(f"Alpha: {analysis.alpha:.2f}")
        print(f"Filters: {analysis.filters}")
        print(f"Reasoning: {analysis.reasoning}")
