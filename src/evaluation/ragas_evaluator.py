"""
RAGAS integration for RAG system evaluation.

Evaluates:
- Context Precision: How relevant are the retrieved chunks?
- Context Recall: Did we retrieve all necessary information?
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: How relevant is the answer to the question?
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Install with: pip install ragas")


@dataclass
class EvaluationResult:
    """Result of RAGAS evaluation."""

    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    # Statistics
    num_samples: int
    passed_threshold: bool

    # Details
    per_sample_scores: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metrics": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
            },
            "statistics": {
                "num_samples": self.num_samples,
                "passed_threshold": self.passed_threshold,
            },
            "per_sample_scores": self.per_sample_scores,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"RAGAS Evaluation Results ({self.num_samples} samples)\n"
            f"  Context Precision: {self.context_precision:.3f}\n"
            f"  Context Recall: {self.context_recall:.3f}\n"
            f"  Faithfulness: {self.faithfulness:.3f}\n"
            f"  Answer Relevancy: {self.answer_relevancy:.3f}\n"
            f"  Passed: {'✅' if self.passed_threshold else '❌'}"
        )


class RAGASEvaluator:
    """
    RAGAS-based RAG system evaluator.

    Uses RAGAS framework to evaluate retrieval and generation quality.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4",
        embeddings_model: Optional[Any] = None,
        threshold_precision: float = 0.8,
        threshold_recall: float = 0.8,
        threshold_faithfulness: float = 0.9,
        threshold_relevancy: float = 0.8,
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            llm_provider: LLM provider (openai, anthropic, ollama)
            llm_model: Model name (gpt-4, claude-3-opus, etc)
            embeddings_model: Optional embeddings model (uses Qwen3 by default)
            threshold_precision: Minimum context precision
            threshold_recall: Minimum context recall
            threshold_faithfulness: Minimum faithfulness
            threshold_relevancy: Minimum answer relevancy
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not installed. Install with: pip install ragas")

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embeddings_model = embeddings_model

        self.thresholds = {
            "context_precision": threshold_precision,
            "context_recall": threshold_recall,
            "faithfulness": threshold_faithfulness,
            "answer_relevancy": threshold_relevancy,
        }

        # Initialize LLM
        self.llm = self._setup_llm()

        # Initialize embeddings (use Qwen3 by default)
        if embeddings_model is None:
            self.embeddings = self._setup_qwen_embeddings()
        else:
            self.embeddings = embeddings_model

        logger.info(
            f"RAGASEvaluator initialized with {llm_provider}/{llm_model}"
        )

    def _setup_llm(self):
        """Setup LLM for RAGAS evaluation."""
        if self.llm_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=self.llm_model, temperature=0)
            except ImportError:
                logger.error("langchain-openai not installed")
                raise

        elif self.llm_provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model=self.llm_model, temperature=0)
            except ImportError:
                logger.error("langchain-anthropic not installed")
                raise

        elif self.llm_provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(model=self.llm_model, temperature=0)
            except ImportError:
                logger.error("langchain-community not installed")
                raise

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _setup_qwen_embeddings(self):
        """Setup Qwen3 embeddings for RAGAS."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            logger.warning(f"Failed to setup Qwen3 embeddings: {e}")
            logger.info("Falling back to sentence-transformers default")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings()

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG system using RAGAS metrics.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved context chunks (list of lists)
            ground_truths: Optional ground truth answers for context recall

        Returns:
            EvaluationResult with all metrics
        """
        if len(questions) != len(answers) != len(contexts):
            raise ValueError("Questions, answers, and contexts must have same length")

        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            if len(ground_truths) != len(questions):
                raise ValueError("Ground truths must match questions length")
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Run evaluation
        logger.info(f"Evaluating {len(questions)} samples with RAGAS...")

        metrics_to_use = [
            context_precision,
            faithfulness,
            answer_relevancy,
        ]

        # Only use context recall if ground truths provided
        if ground_truths:
            metrics_to_use.append(context_recall)

        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        # Extract scores
        scores = {
            "context_precision": result["context_precision"],
            "context_recall": result.get("context_recall", 0.0),
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
        }

        # Check thresholds
        passed = all(
            scores[metric] >= threshold
            for metric, threshold in self.thresholds.items()
            if scores[metric] > 0  # Skip if metric not computed
        )

        # Get per-sample scores if available
        per_sample = []
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            per_sample = df.to_dict("records")

        evaluation_result = EvaluationResult(
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            num_samples=len(questions),
            passed_threshold=passed,
            per_sample_scores=per_sample,
        )

        logger.info(f"Evaluation complete:\n{evaluation_result}")

        return evaluation_result

    def evaluate_from_file(
        self,
        test_file: Path,
    ) -> EvaluationResult:
        """
        Evaluate from a test file.

        Args:
            test_file: Path to JSON file with test cases

        Expected format:
        {
            "test_cases": [
                {
                    "question": "...",
                    "answer": "...",
                    "contexts": ["...", "..."],
                    "ground_truth": "..." (optional)
                }
            ]
        }

        Returns:
            EvaluationResult
        """
        with open(test_file) as f:
            data = json.load(f)

        test_cases = data["test_cases"]

        questions = [tc["question"] for tc in test_cases]
        answers = [tc["answer"] for tc in test_cases]
        contexts = [tc["contexts"] for tc in test_cases]

        ground_truths = None
        if all("ground_truth" in tc for tc in test_cases):
            ground_truths = [tc["ground_truth"] for tc in test_cases]

        return self.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )

    def save_results(
        self,
        result: EvaluationResult,
        output_path: Path,
    ):
        """
        Save evaluation results to file.

        Args:
            result: EvaluationResult to save
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Results saved to {output_path}")


def generate_synthetic_testcases(
    contexts: List[str],
    num_questions: int = 10,
    complexity: str = "simple",
    llm_provider: str = "anthropic",
    llm_model: str = "claude-3-haiku-20240307",
) -> List[Dict]:
    """
    Generate synthetic test cases from contexts.

    Args:
        contexts: List of context passages
        num_questions: Number of questions to generate
        complexity: Complexity level (simple, reasoning, multi-context, conditional)
        llm_provider: LLM provider for generation
        llm_model: Model name (use Haiku for cost efficiency)

    Returns:
        List of test cases with questions and ground truth answers
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS not installed")

    try:
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context
    except ImportError:
        logger.error("RAGAS testset generation not available")
        raise

    logger.info(
        f"Generating {num_questions} {complexity} test cases "
        f"using {llm_provider}/{llm_model}"
    )

    # Setup LLM
    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=llm_model, temperature=0.7)
    elif llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=llm_model, temperature=0.7)
    else:
        raise ValueError(f"Unsupported provider: {llm_provider}")

    # Setup embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
    )

    # Create generator
    generator = TestsetGenerator.from_langchain(
        llm,
        embeddings,
    )

    # Prepare documents
    from langchain.docstore.document import Document
    documents = [Document(page_content=ctx) for ctx in contexts]

    # Generate test set
    distributions = {
        "simple": {simple: 1.0},
        "reasoning": {reasoning: 1.0},
        "multi_context": {multi_context: 1.0},
        "mixed": {simple: 0.4, reasoning: 0.4, multi_context: 0.2},
    }

    distribution = distributions.get(complexity, distributions["simple"])

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=num_questions,
        distributions=distribution,
    )

    # Convert to list of dicts
    test_cases = []
    for item in testset.to_pandas().to_dict("records"):
        test_cases.append({
            "question": item["question"],
            "ground_truth": item.get("ground_truth", ""),
            "contexts": item.get("contexts", []),
            "complexity": complexity,
        })

    logger.info(f"Generated {len(test_cases)} test cases")

    return test_cases


if __name__ == "__main__":
    logger.add("logs/ragas_evaluator.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("RAGAS EVALUATOR")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Context Precision: Relevance of retrieved chunks")
    print("  - Context Recall: Completeness of retrieval")
    print("  - Faithfulness: Answer grounded in context")
    print("  - Answer Relevancy: Answer matches question")
    print("\nSupported LLMs:")
    print("  - OpenAI (GPT-4, GPT-3.5)")
    print("  - Anthropic (Claude 3 Opus, Sonnet, Haiku)")
    print("  - Ollama (local models)")
    print("\nSynthetic Test Generation:")
    print("  - Use Claude Haiku for cost efficiency (~$2.80 for 500 questions)")
    print("  - Complexity levels: simple, reasoning, multi-context, mixed")
    print("  - 90% time savings vs manual creation")
    print("=" * 70)
