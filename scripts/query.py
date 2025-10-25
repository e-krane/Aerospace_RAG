#!/usr/bin/env python3
"""
Query CLI for Aerospace RAG System.

Interactive query interface for the RAG system.

Usage:
    # Single query
    python scripts/query.py "What is the Euler buckling formula?"

    # Interactive mode
    python scripts/query.py --interactive

    # JSON output
    python scripts/query.py "Explain beam bending" --format json

    # Show sources
    python scripts/query.py "What is stress?" --show-sources
"""

import sys
from pathlib import Path
import json
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.rag_pipeline import RAGPipeline


console = Console()


def display_response_text(response, show_sources: bool = True):
    """Display response in text format."""
    # Answer
    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="[bold cyan]Answer[/bold cyan]",
        border_style="cyan",
    ))

    # Metrics
    console.print()
    console.print(f"[dim]Retrieved {response.sources_count} chunks in {response.retrieval_time_ms:.0f}ms[/dim]")
    console.print(f"[dim]Generated answer in {response.generation_time_ms:.0f}ms ({response.tokens_used} tokens)[/dim]")
    console.print(f"[dim]Total time: {response.total_time_ms:.0f}ms[/dim]")

    # Sources
    if show_sources and response.citations:
        console.print()
        table = Table(title="Sources", show_header=True, header_style="bold magenta")
        table.add_column("#", width=3)
        table.add_column("Document")
        table.add_column("Relevance", justify="right")

        for i, citation in enumerate(response.citations[:5], 1):
            doc = citation.get("document", "Unknown")
            score = citation.get("relevance_score", 0.0)
            table.add_row(str(i), doc, f"{score:.2f}")

        console.print(table)


def display_response_json(response):
    """Display response in JSON format."""
    output = {
        "answer": response.answer,
        "query": response.query,
        "model": response.model_used,
        "tokens_used": response.tokens_used,
        "sources_count": response.sources_count,
        "metrics": {
            "retrieval_ms": response.retrieval_time_ms,
            "generation_ms": response.generation_time_ms,
            "total_ms": response.total_time_ms,
        },
        "citations": response.citations,
    }
    console.print_json(json.dumps(output, indent=2))


def display_response_markdown(response):
    """Display response in markdown format."""
    md = f"""# Query: {response.query}

## Answer

{response.answer}

## Metadata

- **Model**: {response.model_used}
- **Tokens**: {response.tokens_used}
- **Sources**: {response.sources_count}
- **Retrieval**: {response.retrieval_time_ms:.0f}ms
- **Generation**: {response.generation_time_ms:.0f}ms
- **Total**: {response.total_time_ms:.0f}ms

## Sources

"""
    for i, citation in enumerate(response.citations[:5], 1):
        doc = citation.get("document", "Unknown")
        chunk = citation.get("chunk_id", "")
        score = citation.get("relevance_score", 0.0)
        md += f"{i}. {doc} (chunk {chunk}, score: {score:.2f})\n"

    console.print(Markdown(md))


@click.command()
@click.argument("query", required=False)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive query mode",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format",
)
@click.option(
    "--show-sources/--no-sources",
    default=True,
    help="Show source citations",
)
@click.option(
    "--max-results",
    "-n",
    default=5,
    type=int,
    help="Maximum chunks to retrieve",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def query(
    query: str,
    interactive: bool,
    format: str,
    show_sources: bool,
    max_results: int,
    verbose: bool,
):
    """
    Query the Aerospace RAG system.

    Ask questions about aerospace engineering and get answers
    from the indexed document corpus.
    """
    # Setup logging
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    logger.add("logs/query.log", rotation="10 MB", level="DEBUG" if verbose else "INFO")

    # Display header
    if not interactive and format == "text":
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Aerospace RAG - Query System[/bold cyan]\n"
            f"Using: qwen3:latest (5.2GB VRAM)",
            border_style="cyan",
        ))
        console.print()

    # Initialize pipeline
    if format == "text":
        console.print("[cyan]Initializing RAG pipeline...[/cyan]")

    try:
        with console.status("[bold green]Loading models...") if format == "text" else console.status():
            pipeline = RAGPipeline(preload_models=True)

        if format == "text":
            console.print("[green]✓[/green] Pipeline ready")
            console.print()
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize pipeline: {e}[/red]")
        logger.exception("Pipeline initialization failed")
        sys.exit(1)

    # Interactive mode
    if interactive:
        console.print("[bold]Interactive Query Mode[/bold]")
        console.print("Type 'exit' or 'quit' to exit\n")

        while True:
            try:
                user_query = console.input("[bold cyan]Query>[/bold cyan] ")

                if user_query.lower() in ["exit", "quit", "q"]:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not user_query.strip():
                    continue

                # Process query
                with console.status("[bold green]Processing..."):
                    response = pipeline.query(
                        user_query,
                        max_results=max_results,
                        include_sources=show_sources,
                    )

                # Display response
                if format == "json":
                    display_response_json(response)
                elif format == "markdown":
                    display_response_markdown(response)
                else:
                    display_response_text(response, show_sources)

                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Query failed")

    # Single query mode
    elif query:
        try:
            # Process query
            with console.status("[bold green]Processing...") if format == "text" else console.status():
                response = pipeline.query(
                    query,
                    max_results=max_results,
                    include_sources=show_sources,
                )

            # Display response
            if format == "json":
                display_response_json(response)
            elif format == "markdown":
                display_response_markdown(response)
            else:
                display_response_text(response, show_sources)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Query failed")
            sys.exit(1)

    else:
        console.print("[yellow]No query provided. Use --interactive or provide a query argument.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    query()
