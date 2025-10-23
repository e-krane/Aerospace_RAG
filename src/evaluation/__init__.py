"""
Evaluation framework for RAG system quality assessment.

Provides RAGAS integration and synthetic test data generation.
"""

from src.evaluation.ragas_evaluator import (
    RAGASEvaluator,
    EvaluationResult,
    generate_synthetic_testcases,
)

__all__ = [
    "RAGASEvaluator",
    "EvaluationResult",
    "generate_synthetic_testcases",
]
