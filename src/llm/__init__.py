"""
LLM integration for answer generation.

This module provides multi-provider LLM support for generating answers
from retrieved context in the RAG pipeline.
"""

from src.llm.client import LLMClient, LLMResponse, LLMProvider, get_llm_client
from src.llm.prompts import (
    PromptBuilder,
    PromptType,
    get_prompt_builder,
    build_technical_qa_prompt,
    build_equation_explanation_prompt,
    build_procedural_steps_prompt,
    build_conceptual_prompt,
    build_comparative_prompt,
)
from src.llm.citation_tracker import (
    CitationTracker,
    Citation,
    ChunkReference,
    EquationSource,
    CitationStyle,
    create_citation_tracker,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMProvider",
    "get_llm_client",
    "PromptBuilder",
    "PromptType",
    "get_prompt_builder",
    "build_technical_qa_prompt",
    "build_equation_explanation_prompt",
    "build_procedural_steps_prompt",
    "build_conceptual_prompt",
    "build_comparative_prompt",
    "CitationTracker",
    "Citation",
    "ChunkReference",
    "EquationSource",
    "CitationStyle",
    "create_citation_tracker",
]
