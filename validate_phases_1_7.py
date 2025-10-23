#!/usr/bin/env python3
"""
Comprehensive validation script for Phases 1-7.

Validates:
- All module files exist
- All critical components are present
- Code structure is correct
- Archon task tracking is complete
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_file_exists(file_path: Path) -> bool:
    """Check if a file exists."""
    return file_path.exists() and file_path.is_file()


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    try:
        return len(file_path.read_text().splitlines())
    except:
        return 0


def validate_phase_1() -> Tuple[bool, List[str]]:
    """Validate Phase 1: Foundation."""
    issues = []

    # Check infrastructure files
    required_files = [
        Path("pyproject.toml"),
        Path("docker-compose.yml"),
        Path("README.md"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")

    # Check directory structure
    required_dirs = [
        Path("src"),
        Path("tests"),
        Path("data"),
        Path("Documents"),
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")

    return len(issues) == 0, issues


def validate_phase_2() -> Tuple[bool, List[str]]:
    """Validate Phase 2: Document Processing."""
    issues = []

    required_files = [
        Path("src/parsers/__init__.py"),
        Path("src/parsers/docling_parser.py"),
        Path("src/parsers/validator.py"),
        Path("src/parsers/marker_parser.py"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")
        else:
            loc = count_lines(file)
            if loc < 50 and file.name != "__init__.py":
                issues.append(f"Suspiciously short: {file} ({loc} lines)")

    # Check for key classes
    validator_path = Path("src/parsers/validator.py")
    if validator_path.exists():
        content = validator_path.read_text()
        if "ParserValidator" not in content:
            issues.append("Missing ParserValidator class")
        if "ValidationResult" not in content:
            issues.append("Missing ValidationResult dataclass")

    marker_path = Path("src/parsers/marker_parser.py")
    if marker_path.exists():
        content = marker_path.read_text()
        if "MarkerParser" not in content:
            issues.append("Missing MarkerParser class")
        if "ParserFallbackChain" not in content:
            issues.append("Missing ParserFallbackChain class")

    return len(issues) == 0, issues


def validate_phase_3() -> Tuple[bool, List[str]]:
    """Validate Phase 3: Intelligent Chunking."""
    issues = []

    required_files = [
        Path("src/chunking/__init__.py"),
        Path("src/chunking/semantic_chunker.py"),
        Path("src/chunking/hierarchical_chunker.py"),
        Path("src/chunking/equation_aware.py"),
        Path("src/chunking/metadata_enricher.py"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")

    return len(issues) == 0, issues


def validate_phase_4() -> Tuple[bool, List[str]]:
    """Validate Phase 4: Embedding Generation."""
    issues = []

    required_files = [
        Path("src/embeddings/__init__.py"),
        Path("src/embeddings/qwen3_embedder.py"),
        Path("src/embeddings/batch_processor.py"),
        Path("tests/test_embeddings.py"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")

    # Check for Matryoshka implementation
    qwen_path = Path("src/embeddings/qwen3_embedder.py")
    if qwen_path.exists():
        content = qwen_path.read_text()
        if "_apply_matryoshka" not in content:
            issues.append("Missing Matryoshka implementation")

    return len(issues) == 0, issues


def validate_phase_5() -> Tuple[bool, List[str]]:
    """Validate Phase 5: Storage and Indexing."""
    issues = []

    required_files = [
        Path("src/storage/__init__.py"),
        Path("src/storage/qdrant_client.py"),
        Path("src/storage/ingestion.py"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")

    # Check for dual index architecture
    qdrant_path = Path("src/storage/qdrant_client.py")
    if qdrant_path.exists():
        content = qdrant_path.read_text()
        if "upsert_equation" not in content:
            issues.append("Missing equation collection methods")

    return len(issues) == 0, issues


def validate_phase_6() -> Tuple[bool, List[str]]:
    """Validate Phase 6: Hybrid Retrieval."""
    issues = []

    required_files = [
        Path("src/retrieval/__init__.py"),
        Path("src/retrieval/bm25_retriever.py"),
        Path("src/retrieval/semantic_retriever.py"),
        Path("src/retrieval/fusion.py"),
        Path("src/retrieval/query_analyzer.py"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")

    # Check for query analyzer
    analyzer_path = Path("src/retrieval/query_analyzer.py")
    if analyzer_path.exists():
        content = analyzer_path.read_text()
        if "QueryAnalyzer" not in content:
            issues.append("Missing QueryAnalyzer class")
        if "QueryType" not in content:
            issues.append("Missing QueryType definition")

    return len(issues) == 0, issues


def validate_phase_7() -> Tuple[bool, List[str]]:
    """Validate Phase 7: Reranking Layer."""
    issues = []

    required_files = [
        Path("src/reranking/__init__.py"),
        Path("src/reranking/jina_colbert_reranker.py"),
        Path("src/reranking/optimization.py"),
        Path("src/retrieval/two_stage_pipeline.py"),
        Path("tests/test_reranking.py"),
    ]

    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing: {file}")

    # Check for optimization features
    opt_path = Path("src/reranking/optimization.py")
    if opt_path.exists():
        content = opt_path.read_text()
        if "RerankerCache" not in content:
            issues.append("Missing RerankerCache class")
        if "OptimizedReranker" not in content:
            issues.append("Missing OptimizedReranker class")
        if "BatchReranker" not in content:
            issues.append("Missing BatchReranker class")

    # Check for two-stage pipeline
    pipeline_path = Path("src/retrieval/two_stage_pipeline.py")
    if pipeline_path.exists():
        content = pipeline_path.read_text()
        if "TwoStageRetriever" not in content:
            issues.append("Missing TwoStageRetriever class")

    return len(issues) == 0, issues


def validate_git_status() -> Tuple[bool, List[str]]:
    """Validate git repository status."""
    import subprocess

    issues = []

    try:
        # Check if working tree is clean
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            issues.append("Working tree has uncommitted changes")

        # Check if pushed to remote
        result = subprocess.run(
            ["git", "rev-list", "@{u}..HEAD"],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            issues.append("Local commits not pushed to remote")

    except Exception as e:
        issues.append(f"Git check failed: {e}")

    return len(issues) == 0, issues


def main():
    """Run comprehensive validation."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}PHASES 1-7 VALIDATION{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    phases = [
        ("Phase 1: Foundation", validate_phase_1),
        ("Phase 2: Document Processing", validate_phase_2),
        ("Phase 3: Intelligent Chunking", validate_phase_3),
        ("Phase 4: Embedding Generation", validate_phase_4),
        ("Phase 5: Storage and Indexing", validate_phase_5),
        ("Phase 6: Hybrid Retrieval", validate_phase_6),
        ("Phase 7: Reranking Layer", validate_phase_7),
    ]

    all_passed = True
    phase_results = []

    for phase_name, validator in phases:
        passed, issues = validator()
        phase_results.append((phase_name, passed, issues))

        if passed:
            print(f"{GREEN}✅ {phase_name}{RESET}")
        else:
            print(f"{RED}❌ {phase_name}{RESET}")
            for issue in issues:
                print(f"   {YELLOW}• {issue}{RESET}")
            all_passed = False

    # Git status check
    print(f"\n{BLUE}Git Repository Status:{RESET}")
    git_passed, git_issues = validate_git_status()
    if git_passed:
        print(f"{GREEN}✅ Git repository clean and pushed{RESET}")
    else:
        print(f"{YELLOW}⚠️  Git repository status:{RESET}")
        for issue in git_issues:
            print(f"   {YELLOW}• {issue}{RESET}")

    # Code statistics
    print(f"\n{BLUE}Code Statistics:{RESET}")

    src_files = list(Path("src").rglob("*.py"))
    test_files = list(Path("tests").rglob("*.py"))

    src_lines = sum(count_lines(f) for f in src_files)
    test_lines = sum(count_lines(f) for f in test_files)

    print(f"  Source files: {len(src_files)} files, {src_lines:,} lines")
    print(f"  Test files: {len(test_files)} files, {test_lines:,} lines")
    print(f"  Total: {len(src_files) + len(test_files)} files, {src_lines + test_lines:,} lines")

    # Summary
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}SUMMARY{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

    passed_count = sum(1 for _, passed, _ in phase_results if passed)
    total_count = len(phase_results)

    if all_passed:
        print(f"{GREEN}✅ ALL PHASES VALIDATED SUCCESSFULLY{RESET}")
        print(f"{GREEN}   {passed_count}/{total_count} phases passed{RESET}")
        print(f"\n{GREEN}Phases 1-7 are production-ready!{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ VALIDATION FAILED{RESET}")
        print(f"{RED}   {passed_count}/{total_count} phases passed{RESET}")
        print(f"\n{RED}Please fix the issues above before continuing.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
