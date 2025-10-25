"""
LaTeX Project Detector and Metadata Enrichment.

Detects when a directory contains a LaTeX book/project (multiple chapters)
and enriches chunks with project-level metadata for better RAG retrieval.

Features:
- Detects book projects vs. standalone documents
- Extracts chapter numbers and titles
- Identifies cross-references between chapters
- Adds project-level metadata to chunks
- Enables cross-chapter context
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class LaTeXProject:
    """Represents a LaTeX book/project with multiple chapters."""

    project_dir: Path
    project_name: str
    chapters: list[dict] = field(default_factory=list)
    has_custom_class: bool = False
    class_file: str | None = None
    shared_files: list[str] = field(default_factory=list)
    cross_references: dict[str, list[str]] = field(default_factory=dict)

    def get_chapter_info(self, file_path: Path) -> dict | None:
        """Get chapter metadata for a specific file."""
        for chapter in self.chapters:
            if chapter['file_path'] == file_path:
                return chapter
        return None


class LaTeXProjectDetector:
    r"""
    Detects LaTeX book projects and enriches metadata.

    A LaTeX project is detected when a directory contains:
    - Multiple .tex files (chapters)
    - Custom .cls file (document class)
    - Cross-reference patterns (\myexternaldocument)
    """

    def __init__(self):
        """Initialize project detector."""
        self.chapter_pattern = re.compile(r'Ch(\d+)_.*\.tex', re.IGNORECASE)
        self.external_doc_pattern = re.compile(r'\\myexternaldocument\{([^\}]+)\}')

    def detect_project(self, directory: Path) -> LaTeXProject | None:
        """
        Detect if directory contains a LaTeX book project.

        Args:
            directory: Directory to analyze

        Returns:
            LaTeXProject if detected, None if standalone documents
        """
        directory = Path(directory)

        if not directory.is_dir():
            return None

        # Find all .tex files
        tex_files = sorted(directory.glob("*.tex"))

        if len(tex_files) < 3:  # Need at least 3 files for a "project"
            return None

        # Check for custom class file
        cls_files = list(directory.glob("*.cls"))
        has_custom_class = len(cls_files) > 0
        class_file = cls_files[0].name if cls_files else None

        # Check for cross-references (sign of multi-file project)
        has_cross_refs = False
        for tex_file in tex_files[:3]:  # Check first 3 files
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            if '\\myexternaldocument' in content or '\\input' in content:
                has_cross_refs = True
                break

        # Must have either custom class OR cross-references to be a project
        if not (has_custom_class or has_cross_refs):
            return None

        # Extract project name from directory
        project_name = directory.name

        # Analyze chapters
        chapters = []
        for tex_file in tex_files:
            chapter_info = self._extract_chapter_info(tex_file)
            if chapter_info:
                chapters.append(chapter_info)

        # Sort chapters by number (handle None values)
        chapters.sort(key=lambda c: c.get('chapter_number') or 999)

        # Find shared files
        shared_files = []
        for pattern in ['crosslink.tex', '*.bib', 'latexmkrc']:
            shared_files.extend([f.name for f in directory.glob(pattern)])

        # Extract cross-references
        cross_refs = self._extract_cross_references(directory, tex_files)

        project = LaTeXProject(
            project_dir=directory,
            project_name=project_name,
            chapters=chapters,
            has_custom_class=has_custom_class,
            class_file=class_file,
            shared_files=shared_files,
            cross_references=cross_refs,
        )

        return project

    def _extract_chapter_info(self, tex_file: Path) -> dict | None:
        """Extract chapter metadata from a .tex file."""
        try:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')

            # Extract chapter number from filename (Ch10_4P.tex -> 10)
            chapter_match = self.chapter_pattern.match(tex_file.name)
            chapter_number = int(chapter_match.group(1)) if chapter_match else None

            # Extract chapter title from \chapter{...}
            title_match = re.search(r'\\chapter(?:\[[^\]]*\])?\{([^\}]+)\}', content)
            chapter_title = title_match.group(1).strip() if title_match else None

            # Check if it's an appendix
            is_appendix = 'app' in tex_file.name.lower()

            # Count equations as indicator of content type
            equation_count_approx = content.count('\\begin{equation}') + content.count('$')

            return {
                'file_path': tex_file,
                'file_name': tex_file.name,
                'chapter_number': chapter_number,
                'chapter_title': chapter_title,
                'is_appendix': is_appendix,
                'has_equations': equation_count_approx > 10,
                'file_size': tex_file.stat().st_size,
            }

        except Exception as e:
            return None

    def _extract_cross_references(
        self,
        directory: Path,
        tex_files: list[Path]
    ) -> dict[str, list[str]]:
        """
        Extract cross-reference graph between chapters.

        Returns:
            Dict mapping file stems to list of referenced file stems
        """
        cross_refs = {}

        for tex_file in tex_files:
            try:
                content = tex_file.read_text(encoding='utf-8', errors='ignore')

                # Find all \myexternaldocument{...} references
                refs = self.external_doc_pattern.findall(content)

                if refs:
                    cross_refs[tex_file.stem] = refs

            except Exception:
                continue

        return cross_refs

    def enrich_chunk_metadata(
        self,
        chunk_metadata: dict,
        file_path: Path,
        project: LaTeXProject | None,
    ) -> dict:
        """
        Enrich chunk metadata with project-level information.

        Args:
            chunk_metadata: Existing chunk metadata
            file_path: Source file path
            project: Detected project (or None for standalone)

        Returns:
            Enriched metadata dict
        """
        if project is None:
            # Standalone document - no enrichment
            chunk_metadata['is_project'] = False
            return chunk_metadata

        # Get chapter info
        chapter_info = project.get_chapter_info(file_path)

        # Add project-level metadata
        enriched = {
            **chunk_metadata,
            # Project identification
            'is_project': True,
            'project_name': project.project_name,
            'project_dir': str(project.project_dir),

            # Chapter information
            'chapter_number': chapter_info.get('chapter_number') if chapter_info else None,
            'chapter_title': chapter_info.get('chapter_title') if chapter_info else None,
            'is_appendix': chapter_info.get('is_appendix', False) if chapter_info else False,

            # Document type indicators
            'has_custom_class': project.has_custom_class,
            'document_class': project.class_file,

            # Cross-reference support
            'has_cross_refs': len(project.cross_references) > 0,
            'references_chapters': project.cross_references.get(file_path.stem, []),

            # Book context flag (important for retrieval)
            'book_context': True,
            'total_chapters': len(project.chapters),
        }

        return enriched


def detect_latex_project(directory: Path) -> LaTeXProject | None:
    """
    Convenience function to detect LaTeX projects.

    Args:
        directory: Directory to check

    Returns:
        LaTeXProject if detected, None otherwise

    Example:
        >>> project = detect_latex_project(Path("Documents/Aerospace_Structures_LaTeX"))
        >>> if project:
        ...     print(f"Found project: {project.project_name}")
        ...     print(f"  Chapters: {len(project.chapters)}")
        ...     print(f"  Custom class: {project.class_file}")
    """
    detector = LaTeXProjectDetector()
    return detector.detect_project(directory)


if __name__ == "__main__":
    # Test the detector
    import sys

    test_dir = Path("Documents/Aerospace_Structures_LaTeX")
    if len(sys.argv) > 1:
        test_dir = Path(sys.argv[1])

    print(f"Testing LaTeX project detector on: {test_dir}")
    print("=" * 70)

    detector = LaTeXProjectDetector()
    project = detector.detect_project(test_dir)

    if project:
        print(f"\n✅ Detected LaTeX Project!")
        print(f"  Project Name: {project.project_name}")
        print(f"  Directory: {project.project_dir}")
        print(f"  Custom Class: {project.class_file or 'None'}")
        print(f"  Total Chapters: {len(project.chapters)}")
        print(f"  Shared Files: {', '.join(project.shared_files)}")
        print(f"  Cross-References: {len(project.cross_references)} files")

        print(f"\n  Chapters:")
        for ch in project.chapters[:10]:  # Show first 10
            ch_num = ch.get('chapter_number', '?')
            ch_title = ch.get('chapter_title', 'Unknown')
            has_eq = '📐' if ch.get('has_equations') else '📝'
            print(f"    {has_eq} Ch {ch_num:2}: {ch_title}")

        if len(project.chapters) > 10:
            print(f"    ... and {len(project.chapters) - 10} more")

        print(f"\n  Cross-Reference Graph (sample):")
        for file_stem, refs in list(project.cross_references.items())[:3]:
            print(f"    {file_stem} → {len(refs)} refs")

        # Test metadata enrichment
        print(f"\n  Testing metadata enrichment:")
        test_file = project.chapters[0]['file_path']
        test_metadata = {'chunk_index': 0, 'chunking_method': 'semantic'}
        enriched = detector.enrich_chunk_metadata(test_metadata, test_file, project)

        print(f"    Original metadata keys: {list(test_metadata.keys())}")
        print(f"    Enriched metadata keys: {list(enriched.keys())}")
        print(f"    Added: {set(enriched.keys()) - set(test_metadata.keys())}")

    else:
        print(f"\n❌ No LaTeX project detected in {test_dir}")
        print(f"   (Directory may contain standalone documents)")
