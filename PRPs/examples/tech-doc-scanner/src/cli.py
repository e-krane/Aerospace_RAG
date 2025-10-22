"""
Command-line interface for Tech Doc Scanner.

Provides a unified CLI for converting technical PDFs to Markdown with
equation recognition.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import Config, DoclingConfig, OutputConfig, Pix2TextConfig
from .converter import DocumentConverterWrapper
from .stats import BatchStatistics, ConversionStatistics

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="tech-doc-scanner")
def cli():
    """
    Tech Doc Scanner - High-quality PDF to Markdown conversion.

    Convert technical PDFs to Markdown with state-of-the-art equation recognition.
    """
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="./output",
    help="Output directory for converted files",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML configuration file",
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Choice(["standard", "pix2text"], case_sensitive=False),
    default="pix2text",
    help="Pipeline type to use",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Device for Pix2Text (cpu or cuda)",
)
@click.option("--no-validation", is_flag=True, help="Disable LaTeX validation")
@click.option("--no-fallback", is_flag=True, help="Disable OCR fallback on validation failure")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def convert(
    input_file: Path,
    output_dir: Path,
    config: Optional[Path],
    pipeline: str,
    device: str,
    no_validation: bool,
    no_fallback: bool,
    verbose: bool,
):
    """
    Convert a single PDF file to Markdown.

    Example:
        tech-doc-scanner convert input.pdf -o ./output
        tech-doc-scanner convert input.pdf --device cuda --verbose
    """
    setup_logging(verbose)

    try:
        # Load configuration
        if config:
            app_config = Config.from_yaml(config)
            console.print(f"[green]Loaded configuration from: {config}[/green]")
        else:
            # Create default config with CLI overrides
            pix2text_config = Pix2TextConfig(
                device=device, validate=not no_validation, fallback_to_ocr=not no_fallback
            )
            app_config = Config(
                pix2text=pix2text_config, output=OutputConfig(base_dir=output_dir)
            )

        console.print(f"\n[bold cyan]Converting:[/bold cyan] {input_file}")
        console.print(f"[bold cyan]Pipeline:[/bold cyan] {pipeline}")
        console.print(f"[bold cyan]Device:[/bold cyan] {device}")
        console.print(f"[bold cyan]Output:[/bold cyan] {output_dir}\n")

        # Create converter
        converter = DocumentConverterWrapper(
            pipeline_type=pipeline,
            docling_config=app_config.docling,
            pix2text_config=app_config.pix2text,
            output_config=app_config.output,
        )

        # Convert with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Converting PDF...", total=None)
            result = converter.convert_file(input_file, output_dir)
            progress.update(task, completed=True)

        # Display results
        if result.success:
            console.print(f"\n[bold green]✓ Conversion successful![/bold green]")
            console.print(f"[green]Time: {result.elapsed_time:.1f}s[/green]")
            console.print(f"\n[bold]Output files:[/bold]")
            for file in result.output_files:
                console.print(f"  • {file}")

            # Display formula stats if available
            if result.stats:
                console.print(f"\n[bold]Formula Recognition:[/bold]")
                console.print(f"  Total: {result.stats.total}")
                console.print(f"  Recognized: {result.stats.recognized}")
                console.print(f"  Success rate: {result.stats.success_rate:.1f}%")

            sys.exit(0)
        else:
            console.print(f"\n[bold red]✗ Conversion failed![/bold red]")
            console.print(f"[red]Error: {result.error}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("input_pattern", type=str)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="./output",
    help="Output directory for converted files",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML configuration file",
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Choice(["standard", "pix2text"], case_sensitive=False),
    default="pix2text",
    help="Pipeline type to use",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    help="Device for Pix2Text (cpu or cuda)",
)
@click.option("--report", "-r", is_flag=True, help="Generate HTML report")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def batch(
    input_pattern: str,
    output_dir: Path,
    config: Optional[Path],
    pipeline: str,
    device: str,
    report: bool,
    verbose: bool,
):
    """
    Batch convert multiple PDF files.

    INPUT_PATTERN can be a glob pattern like "./pdfs/*.pdf"

    Example:
        tech-doc-scanner batch "./pdfs/*.pdf" -o ./output
        tech-doc-scanner batch "*.pdf" --device cuda --report
    """
    setup_logging(verbose)

    try:
        # Find matching files
        import glob

        input_files = [Path(f) for f in glob.glob(input_pattern)]
        if not input_files:
            console.print(f"[yellow]No files found matching pattern: {input_pattern}[/yellow]")
            sys.exit(1)

        console.print(f"\n[bold cyan]Found {len(input_files)} file(s) to convert[/bold cyan]\n")

        # Load configuration
        if config:
            app_config = Config.from_yaml(config)
        else:
            app_config = Config(
                pix2text=Pix2TextConfig(device=device), output=OutputConfig(base_dir=output_dir)
            )

        # Create converter
        converter = DocumentConverterWrapper(
            pipeline_type=pipeline,
            docling_config=app_config.docling,
            pix2text_config=app_config.pix2text,
            output_config=app_config.output,
        )

        # Convert with progress bar
        batch_stats = BatchStatistics()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Converting files...", total=len(input_files))

            for input_file in input_files:
                progress.update(task, description=f"[cyan]Converting {input_file.name}...")
                result = converter.convert_file(input_file, output_dir)

                # Collect statistics
                conv_stats = ConversionStatistics(
                    input_file=input_file.name,
                    success=result.success,
                    elapsed_time=result.elapsed_time,
                    output_formats=[f.suffix[1:] for f in result.output_files],
                    formula_stats=result.stats.to_dict() if result.stats else None,
                    error=result.error,
                )
                batch_stats.add_conversion(conv_stats)

                progress.advance(task)

        # Display summary
        batch_stats.print_summary()

        # Generate report if requested
        if report:
            report_path = output_dir / "conversion_report.html"
            batch_stats.generate_html_report(report_path)
            console.print(f"\n[bold green]Report saved:[/bold green] {report_path}")

        # Exit with appropriate code
        sys.exit(0 if batch_stats.failed == 0 else 1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="tech-doc-scanner.yaml",
    help="Output path for configuration file",
)
def config_generate(output: Path):
    """
    Generate a sample configuration file.

    Example:
        tech-doc-scanner config-generate -o my-config.yaml
    """
    try:
        config = Config()
        config.to_yaml(output)
        console.print(f"[green]✓ Configuration saved to:[/green] {output}")
        console.print("\n[bold]Edit the file to customize settings, then use:[/bold]")
        console.print(f"  tech-doc-scanner convert input.pdf --config {output}")
    except Exception as e:
        console.print(f"[red]Error generating config:[/red] {e}")
        sys.exit(1)


@cli.command(name="config-validate")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
def config_validate(config_file: Path):
    """
    Validate a configuration file.

    Example:
        tech-doc-scanner config-validate my-config.yaml
    """
    try:
        config = Config.from_yaml(config_file)
        console.print(f"[green]✓ Configuration is valid:[/green] {config_file}")
        console.print("\n[bold]Configuration summary:[/bold]")
        console.print(f"  Pipeline: {config.pix2text.device}")
        console.print(f"  OCR: {config.docling.ocr_languages}")
        console.print(f"  Output formats: {config.output.formats}")
    except Exception as e:
        console.print(f"[red]✗ Configuration is invalid:[/red] {e}")
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
