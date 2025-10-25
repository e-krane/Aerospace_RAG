#!/usr/bin/env python3
"""
Document Indexing CLI for Aerospace RAG System.

Indexes aerospace documents (PDFs) into the vector database using:
- Docling parser for document extraction
- Semantic chunking with equation preservation
- Qwen3-embedding:8b for vector generation
- Qdrant storage with binary quantization

Usage:
    # Index single document
    python scripts/index_documents.py --input data/raw/aerospace_textbook.pdf

    # Index directory
    python scripts/index_documents.py --input data/raw/ --recursive

    # Custom batch size
    python scripts/index_documents.py --input document.pdf --batch-size 64

    # Force reindex
    python scripts/index_documents.py --input document.pdf --force
"""

import sys
from pathlib import Path
from typing import List, Optional
import time
import click
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.rag_pipeline import RAGPipeline, IndexingResult


console = Console()


def find_document_files(path: Path, recursive: bool = False) -> List[Path]:
    """
    Find all document files (PDF and LaTeX) in a path.

    Args:
        path: File or directory path
        recursive: Search subdirectories

    Returns:
        List of document file paths (PDF and .tex)
    """
    if path.is_file():
        if path.suffix.lower() in [".pdf", ".tex"]:
            return [path]
        else:
            console.print(f"[yellow]Warning: {path} is not a PDF or LaTeX file[/yellow]")
            return []

    if path.is_dir():
        if recursive:
            pdf_files = list(path.rglob("*.pdf"))
            tex_files = list(path.rglob("*.tex"))
        else:
            pdf_files = list(path.glob("*.pdf"))
            tex_files = list(path.glob("*.tex"))

        all_files = pdf_files + tex_files
        return sorted(all_files)

    console.print(f"[red]Error: {path} not found[/red]")
    return []


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


def format_time(ms: float) -> str:
    """Format milliseconds as human-readable time."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}min"


def display_summary(results: List[IndexingResult], total_time_ms: float):
    """Display indexing summary table."""
    # Calculate totals
    total_chunks = sum(r.chunks_indexed for r in results)
    total_pages = sum(r.pages_processed for r in results)
    total_equations = sum(r.equations_preserved for r in results)
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    # Create summary table
    table = Table(title="Indexing Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Documents Processed", str(len(results)))
    table.add_row("Successful", f"[green]{successful}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Total Pages", str(total_pages))
    table.add_row("Equations Preserved", str(total_equations))
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Total Time", format_time(total_time_ms))
    table.add_row("Chunks/Second", f"{total_chunks / (total_time_ms / 1000):.1f}")

    console.print()
    console.print(table)


def display_document_result(result: IndexingResult):
    """Display results for a single document."""
    doc_name = Path(result.document_path).name

    if result.success:
        # Success message
        console.print(
            f"  [green]✓[/green] {doc_name}: "
            f"{result.chunks_indexed} chunks in {format_time(result.total_time_ms)}"
        )

        # Detailed breakdown
        console.print(
            f"    Parsing: {format_time(result.parsing_time_ms)}, "
            f"Chunking: {format_time(result.chunking_time_ms)}, "
            f"Embedding: {format_time(result.embedding_time_ms)}, "
            f"Indexing: {format_time(result.indexing_time_ms)}"
        )

        if result.equations_preserved > 0:
            console.print(
                f"    [cyan]{result.equations_preserved} equations preserved[/cyan]"
            )
    else:
        # Error message
        console.print(f"  [red]✗[/red] {doc_name}: Failed")
        for error in result.errors:
            console.print(f"    [red]Error: {error}[/red]")


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="PDF/LaTeX file or directory to index",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recursively search directories for documents",
)
@click.option(
    "--batch-size",
    "-b",
    default=32,
    type=int,
    help="Embedding batch size (default: 32)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force reindex even if document already indexed",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default="logs/indexing.log",
    help="Log file path",
)
def index_documents(
    input: str,
    recursive: bool,
    batch_size: int,
    force: bool,
    verbose: bool,
    log_file: str,
):
    """
    Index aerospace documents into the RAG system.

    Processes PDF documents through the complete indexing pipeline:
    parsing, chunking, embedding, and storage in Qdrant.
    """
    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        log_file,
        rotation="10 MB",
        level="DEBUG" if verbose else "INFO",
    )
    if verbose:
        logger.add(sys.stderr, level="DEBUG")

    # Display header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Aerospace RAG - Document Indexing[/bold cyan]\n"
            f"Using: qwen3-embedding:8b (4.7GB VRAM)",
            border_style="cyan",
        )
    )
    console.print()

    # Find document files
    input_path = Path(input)
    doc_files = find_document_files(input_path, recursive)

    if not doc_files:
        console.print("[red]No document files found![/red]")
        sys.exit(1)

    # Count file types
    pdf_count = sum(1 for f in doc_files if f.suffix.lower() == ".pdf")
    tex_count = sum(1 for f in doc_files if f.suffix.lower() == ".tex")

    console.print(f"Found {len(doc_files)} file(s) to index:")
    if pdf_count > 0:
        console.print(f"  • {pdf_count} PDF file(s)")
    if tex_count > 0:
        console.print(f"  • {tex_count} LaTeX file(s)")
    console.print()

    # Initialize pipeline
    console.print("[cyan]Initializing RAG pipeline...[/cyan]")
    try:
        with console.status("[bold green]Loading models..."):
            pipeline = RAGPipeline(preload_models=True)
        console.print("[green]✓[/green] Pipeline ready")
        console.print()
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize pipeline: {e}[/red]")
        logger.exception("Pipeline initialization failed")
        sys.exit(1)

    # Index documents
    results: List[IndexingResult] = []
    total_start = time.time()

    console.print("[bold]Indexing documents:[/bold]")
    console.print()

    for doc_file in doc_files:
        try:
            # Get file info
            file_size = doc_file.stat().st_size

            console.print(
                f"[bold]{doc_file.name}[/bold] ({format_size(file_size)})"
            )

            # Index with progress indicator
            with console.status(f"[bold green]Processing...") as status:
                result = pipeline.index_document(
                    str(doc_file),
                    batch_size=batch_size,
                    show_progress=False,  # Disable internal progress
                )

            results.append(result)
            display_document_result(result)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Indexing interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]✗ Error processing {doc_file.name}: {e}[/red]")
            logger.exception(f"Error processing {doc_file}")
            console.print()

    total_time_ms = (time.time() - total_start) * 1000

    # Display summary
    if results:
        display_summary(results, total_time_ms)
    else:
        console.print("[yellow]No documents were successfully indexed[/yellow]")

    # Exit code
    failed_count = sum(1 for r in results if not r.success)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    index_documents()
