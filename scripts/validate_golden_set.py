#!/usr/bin/env python3
"""
Validate and expand golden test set.

Validates:
- JSON schema correctness
- Required fields present
- Distribution matches target
- No duplicate IDs

Provides utilities for expanding the test set to 200 cases.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import Counter

from loguru import logger


REQUIRED_FIELDS = [
    "id",
    "category",
    "difficulty",
    "question",
    "expected_chunks",
    "answer_keywords",
    "ground_truth",
]

VALID_CATEGORIES = [
    "equation_heavy",
    "conceptual_understanding",
    "cross_reference",
    "figure_interpretation",
    "procedural",
]

TARGET_DISTRIBUTION = {
    "equation_heavy": 50,
    "conceptual_understanding": 50,
    "cross_reference": 30,
    "figure_interpretation": 30,
    "procedural": 40,
}


def validate_test_case(test_case: Dict, case_idx: int) -> List[str]:
    """
    Validate a single test case.

    Args:
        test_case: Test case dictionary
        case_idx: Index of test case (for error reporting)

    Returns:
        List of validation errors
    """
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in test_case:
            errors.append(f"Case {case_idx}: Missing required field '{field}'")

    # Check category validity
    if "category" in test_case:
        if test_case["category"] not in VALID_CATEGORIES:
            errors.append(
                f"Case {case_idx}: Invalid category '{test_case['category']}'. "
                f"Must be one of {VALID_CATEGORIES}"
            )

    # Check difficulty validity
    if "difficulty" in test_case:
        if test_case["difficulty"] not in ["easy", "medium", "hard"]:
            errors.append(
                f"Case {case_idx}: Invalid difficulty '{test_case['difficulty']}'"
            )

    # Check data types
    if "expected_chunks" in test_case:
        if not isinstance(test_case["expected_chunks"], list):
            errors.append(f"Case {case_idx}: expected_chunks must be a list")

    if "answer_keywords" in test_case:
        if not isinstance(test_case["answer_keywords"], list):
            errors.append(f"Case {case_idx}: answer_keywords must be a list")

    # Check minimum content length
    if "question" in test_case:
        if len(test_case["question"].strip()) < 10:
            errors.append(f"Case {case_idx}: Question too short")

    if "ground_truth" in test_case:
        if len(test_case["ground_truth"].strip()) < 20:
            errors.append(f"Case {case_idx}: Ground truth too short")

    return errors


def validate_golden_set(golden_set_path: Path) -> Dict:
    """
    Validate entire golden test set.

    Args:
        golden_set_path: Path to golden_set.json

    Returns:
        Validation report dictionary
    """
    logger.info(f"Validating golden set: {golden_set_path}")

    try:
        with open(golden_set_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return {"valid": False, "errors": [f"JSON load failed: {e}"]}

    errors = []

    # Check top-level structure
    if "test_cases" not in data:
        errors.append("Missing 'test_cases' field")
        return {"valid": False, "errors": errors}

    test_cases = data["test_cases"]

    # Check each test case
    for idx, test_case in enumerate(test_cases):
        case_errors = validate_test_case(test_case, idx)
        errors.extend(case_errors)

    # Check for duplicate IDs
    ids = [tc.get("id") for tc in test_cases if "id" in tc]
    duplicate_ids = [id for id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate IDs found: {duplicate_ids}")

    # Check distribution
    category_counts = Counter(tc.get("category") for tc in test_cases)
    distribution_report = {}

    for category, target_count in TARGET_DISTRIBUTION.items():
        actual_count = category_counts.get(category, 0)
        distribution_report[category] = {
            "target": target_count,
            "actual": actual_count,
            "progress": f"{actual_count}/{target_count} ({actual_count/target_count*100:.0f}%)",
        }

    # Generate report
    valid = len(errors) == 0

    report = {
        "valid": valid,
        "total_cases": len(test_cases),
        "target_cases": sum(TARGET_DISTRIBUTION.values()),
        "progress": f"{len(test_cases)}/{sum(TARGET_DISTRIBUTION.values())} ({len(test_cases)/sum(TARGET_DISTRIBUTION.values())*100:.0f}%)",
        "distribution": distribution_report,
        "errors": errors,
    }

    return report


def print_report(report: Dict):
    """Print validation report."""
    print("\n" + "=" * 70)
    print("GOLDEN TEST SET VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal cases: {report['total_cases']}/{report['target_cases']}")
    print(f"Progress: {report['progress']}")

    print("\nDistribution by category:")
    for category, stats in report["distribution"].items():
        status = "✅" if stats["actual"] >= stats["target"] else "⚠️"
        print(f"  {status} {category:30s} {stats['progress']}")

    if report["errors"]:
        print(f"\n❌ Validation errors ({len(report['errors'])}):")
        for error in report["errors"][:10]:  # Show first 10
            print(f"  • {error}")
        if len(report["errors"]) > 10:
            print(f"  ... and {len(report['errors']) - 10} more errors")
    else:
        print("\n✅ No validation errors!")

    print("\n" + "=" * 70)

    if report["valid"]:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED - Fix errors above")

    print("=" * 70 + "\n")


def generate_template_cases(category: str, num_cases: int) -> List[Dict]:
    """
    Generate template test cases for a category.

    Args:
        category: Category name
        num_cases: Number of template cases to generate

    Returns:
        List of template test cases
    """
    templates = []

    for i in range(num_cases):
        case_id = f"{category[:4]}-{i+1:03d}"

        template = {
            "id": case_id,
            "category": category,
            "difficulty": "medium",
            "question": f"[TODO: Add {category} question {i+1}]",
            "expected_chunks": ["chapter_X_section_Y"],
            "answer_keywords": ["keyword1", "keyword2"],
            "ground_truth": "[TODO: Add ground truth answer]",
            "notes": f"Template for {category} case {i+1}",
        }

        # Add category-specific fields
        if category == "equation_heavy":
            template["expected_equations"] = ["$$[TODO: Add LaTeX equation]$$"]
        elif category == "figure_interpretation":
            template["expected_figures"] = ["Figure X.Y"]

        templates.append(template)

    return templates


def expand_golden_set(
    golden_set_path: Path,
    output_path: Optional[Path] = None,
):
    """
    Expand golden set with templates to reach target distribution.

    Args:
        golden_set_path: Path to existing golden_set.json
        output_path: Path to save expanded set (default: same as input)
    """
    if output_path is None:
        output_path = golden_set_path

    with open(golden_set_path) as f:
        data = json.load(f)

    test_cases = data["test_cases"]

    # Check current distribution
    category_counts = Counter(tc.get("category") for tc in test_cases)

    print("\nExpanding golden set with templates...")

    # Add templates for categories below target
    for category, target_count in TARGET_DISTRIBUTION.items():
        actual_count = category_counts.get(category, 0)

        if actual_count < target_count:
            needed = target_count - actual_count
            print(f"  Adding {needed} templates for {category}")

            templates = generate_template_cases(category, needed)
            test_cases.extend(templates)

    # Update data
    data["test_cases"] = test_cases
    data["total_cases"] = len(test_cases)

    # Save
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Expanded golden set saved to {output_path}")
    print(f"   Total cases: {len(test_cases)}")
    print("\nNext steps:")
    print("  1. Review and fill in [TODO] placeholders")
    print("  2. Customize questions and answers")
    print("  3. Validate again with this script\n")


def main():
    """Main execution."""
    logger.add("logs/golden_set_validation.log", rotation="10 MB")

    golden_set_path = Path("data/evaluation/golden_set.json")

    if not golden_set_path.exists():
        print(f"\n❌ Golden set not found: {golden_set_path}\n")
        return 1

    # Validate
    report = validate_golden_set(golden_set_path)
    print_report(report)

    # Offer to expand if incomplete
    if report["total_cases"] < report["target_cases"]:
        print("The golden set is incomplete.")
        response = input("Generate templates to reach 200 cases? (y/n): ")

        if response.lower() == "y":
            expand_golden_set(golden_set_path)

            # Validate again
            print("\nRe-validating expanded set...")
            report = validate_golden_set(golden_set_path)
            print_report(report)

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
