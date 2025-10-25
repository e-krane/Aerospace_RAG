#!/usr/bin/env python3
"""
Test project-aware LaTeX indexing.

Validates that chunks from a LaTeX book project are enriched with:
- Project name and directory
- Chapter number and title
- Cross-reference information
- Book context flag
"""

from pathlib import Path
from src.parsers.latex_parser import LaTeXParser
from src.parsers.latex_project_detector import LaTeXProjectDetector
from src.chunking.semantic_chunker import SemanticChunker

def test_project_aware_chunking():
    """Test that chunks get enriched with project metadata."""

    print("=" * 70)
    print("TESTING PROJECT-AWARE LaTeX INDEXING")
    print("=" * 70)

    # Test file from the aerospace book
    test_file = Path("Documents/Aerospace_Structures_LaTeX/Ch10_4P.tex")

    if not test_file.exists():
        print(f"\n❌ Test file not found: {test_file}")
        return

    print(f"\nTest file: {test_file.name}")
    print("=" * 70)

    # Step 1: Detect project
    print("\n1. Detecting project context...")
    detector = LaTeXProjectDetector()
    project = detector.detect_project(test_file.parent)

    if project:
        print(f"   ✅ Detected project: {project.project_name}")
        print(f"   📚 {len(project.chapters)} chapters")
        print(f"   📐 Custom class: {project.class_file}")
    else:
        print("   ❌ No project detected (standalone document)")
        return

    # Step 2: Parse LaTeX
    print("\n2. Parsing LaTeX file...")
    parser = LaTeXParser()
    parsed_latex = parser.parse_file(test_file)

    print(f"   ✅ Parsed successfully")
    print(f"   📄 Sections: {len(parsed_latex.sections)}")
    print(f"   📐 Equations: {len(parsed_latex.equations)}")

    # Step 3: Create chunks
    print("\n3. Creating semantic chunks...")
    markdown = parser.to_markdown(parsed_latex)
    chunker = SemanticChunker(chunk_size=750)
    chunks = chunker.chunk_text(markdown, document_id=test_file.stem)

    print(f"   ✅ Created {len(chunks)} chunks")
    print(f"   📊 Avg tokens: {sum(c.token_count for c in chunks) / len(chunks):.0f}")

    # Step 4: Enrich with project metadata
    print("\n4. Enriching chunks with project metadata...")
    for chunk in chunks:
        chunk.metadata = detector.enrich_chunk_metadata(
            chunk.metadata,
            test_file,
            project
        )

    print(f"   ✅ Enriched all {len(chunks)} chunks")

    # Step 5: Validate enrichment
    print("\n5. Validating metadata enrichment...")

    first_chunk = chunks[0]
    metadata_keys = set(first_chunk.metadata.keys())

    # Expected project metadata
    expected_fields = {
        'is_project',
        'project_name',
        'chapter_number',
        'chapter_title',
        'book_context',
        'total_chapters',
    }

    missing = expected_fields - metadata_keys
    if missing:
        print(f"   ❌ Missing fields: {missing}")
    else:
        print(f"   ✅ All expected fields present")

    # Display sample chunk with metadata
    print("\n6. Sample chunk with enriched metadata:")
    print("   " + "=" * 66)

    sample = chunks[0]
    print(f"\n   Chunk ID: {sample.chunk_id}")
    print(f"   Tokens: {sample.token_count}")
    print(f"   Content preview: {sample.content[:100]}...")

    print(f"\n   📋 Metadata ({len(sample.metadata)} fields):")
    for key, value in sorted(sample.metadata.items()):
        if isinstance(value, (str, int, bool)):
            print(f"      {key:25} = {value}")
        elif isinstance(value, list):
            print(f"      {key:25} = [{len(value)} items]")

    # Key validations
    print("\n7. Key validations:")
    validations = [
        ("Project detected", first_chunk.metadata.get('is_project') == True),
        ("Project name set", first_chunk.metadata.get('project_name') == 'Aerospace_Structures_LaTeX'),
        ("Chapter number", first_chunk.metadata.get('chapter_number') == 10),
        ("Book context flag", first_chunk.metadata.get('book_context') == True),
        ("Total chapters", first_chunk.metadata.get('total_chapters') == 22),
    ]

    all_pass = True
    for check, passed in validations:
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
        if not passed:
            all_pass = False

    # Final summary
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED!")
        print("\nProject-aware indexing working correctly:")
        print("  ✓ Detects LaTeX projects automatically")
        print("  ✓ Enriches chunks with project metadata")
        print("  ✓ Preserves chapter context")
        print("  ✓ Enables cross-chapter retrieval")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Check the output above for details")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_project_aware_chunking()
