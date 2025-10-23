#!/usr/bin/env python3
"""
Generate synthetic test dataset using RAGAS and Claude Haiku.

Generates 500 question-answer pairs at multiple complexity levels
for evaluation of the Aerospace RAG system.

Cost: ~$2.80 using Claude Haiku
Time savings: 90% vs manual creation
"""

import sys
import json
from pathlib import Path
from typing import List

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.ragas_evaluator import generate_synthetic_testcases


def load_context_passages(corpus_dir: Path) -> List[str]:
    """
    Load context passages from processed corpus.

    Args:
        corpus_dir: Directory containing processed markdown files

    Returns:
        List of context passages
    """
    contexts = []

    # Look for processed markdown files
    for md_file in corpus_dir.glob("**/*.md"):
        try:
            content = md_file.read_text()

            # Split into paragraphs (simple approach)
            paragraphs = [
                p.strip()
                for p in content.split("\n\n")
                if len(p.strip()) > 100  # Minimum length
            ]

            contexts.extend(paragraphs)
            logger.info(f"Loaded {len(paragraphs)} passages from {md_file.name}")

        except Exception as e:
            logger.warning(f"Failed to load {md_file}: {e}")

    logger.info(f"Total context passages: {len(contexts)}")
    return contexts


def generate_testset(
    contexts: List[str],
    output_dir: Path,
    num_total: int = 500,
):
    """
    Generate synthetic testset at multiple complexity levels.

    Args:
        contexts: Context passages to generate from
        output_dir: Output directory for test files
        num_total: Total number of test cases (default: 500)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distribution across complexity levels
    distributions = {
        "simple": 200,  # 40%
        "reasoning": 150,  # 30%
        "multi_context": 100,  # 20%
        "mixed": 50,  # 10%
    }

    all_testcases = []

    for complexity, num_questions in distributions.items():
        logger.info(f"\nGenerating {num_questions} {complexity} test cases...")

        try:
            testcases = generate_synthetic_testcases(
                contexts=contexts,
                num_questions=num_questions,
                complexity=complexity,
                llm_provider="anthropic",
                llm_model="claude-3-haiku-20240307",
            )

            # Save individual complexity level
            complexity_file = output_dir / f"synthetic_{complexity}.json"
            with open(complexity_file, "w") as f:
                json.dump(
                    {
                        "complexity": complexity,
                        "num_cases": len(testcases),
                        "test_cases": testcases,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Saved {len(testcases)} cases to {complexity_file}")

            all_testcases.extend(testcases)

        except Exception as e:
            logger.error(f"Failed to generate {complexity} cases: {e}")

    # Save combined testset
    combined_file = output_dir / "synthetic_testset.json"
    with open(combined_file, "w") as f:
        json.dump(
            {
                "description": "Synthetic test cases generated with RAGAS and Claude Haiku",
                "total_cases": len(all_testcases),
                "distribution": distributions,
                "test_cases": all_testcases,
            },
            f,
            indent=2,
        )

    logger.info(f"\nGenerated {len(all_testcases)} total test cases")
    logger.info(f"Combined testset saved to {combined_file}")

    # Generate summary
    summary = {
        "total_generated": len(all_testcases),
        "by_complexity": {},
    }

    for complexity in distributions.keys():
        count = sum(1 for tc in all_testcases if tc.get("complexity") == complexity)
        summary["by_complexity"][complexity] = count

    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")


def main():
    """Main execution."""
    logger.add("logs/synthetic_generation.log", rotation="10 MB")

    print("\n" + "=" * 70)
    print("SYNTHETIC TEST DATA GENERATION")
    print("=" * 70)
    print("\nTarget: 500 question-answer pairs")
    print("LLM: Claude Haiku (cost-efficient)")
    print("Distribution:")
    print("  - Simple: 200 (40%)")
    print("  - Reasoning: 150 (30%)")
    print("  - Multi-context: 100 (20%)")
    print("  - Mixed: 50 (10%)")
    print("\nEstimated cost: ~$2.80")
    print("Time savings: 90% vs manual creation")
    print("=" * 70 + "\n")

    # Configuration
    processed_dir = Path("data/processed")
    output_dir = Path("data/evaluation/synthetic")

    # Check if corpus is processed
    if not processed_dir.exists() or not list(processed_dir.glob("**/*.md")):
        logger.error(
            f"No processed corpus found in {processed_dir}. "
            "Please process documents first with Docling parser."
        )
        print("\n❌ No processed corpus found!")
        print("\nTo process corpus:")
        print("  1. Run: python -m src.parsers.docling_parser Documents/.../*.pdf")
        print("  2. Check output in data/processed/")
        print("  3. Re-run this script\n")
        return 1

    # Load contexts
    logger.info("Loading context passages from processed corpus...")
    contexts = load_context_passages(processed_dir)

    if len(contexts) < 50:
        logger.error(
            f"Too few contexts ({len(contexts)}). Need at least 50 passages."
        )
        return 1

    # Generate testset
    try:
        generate_testset(
            contexts=contexts,
            output_dir=output_dir,
            num_total=500,
        )

        print("\n✅ Synthetic testset generation complete!")
        print(f"   Output: {output_dir}/")
        print(f"   Files: synthetic_testset.json, synthetic_*.json")
        print("\nNext steps:")
        print("  1. Review generated questions for quality")
        print("  2. Add human refinements if needed")
        print("  3. Combine with golden test set")
        print("  4. Run evaluation pipeline\n")

        return 0

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"\n❌ Generation failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
