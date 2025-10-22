"""
Batch PDF processing example.

This example demonstrates how to convert multiple PDFs in a batch
with consolidated statistics and reporting.
"""

from pathlib import Path
import glob
from tech_doc_scanner import DocumentConverterWrapper, DoclingConfig, Pix2TextConfig, OutputConfig
from tech_doc_scanner.stats import BatchStatistics, ConversionStatistics


def main():
    """Batch convert multiple PDFs."""
    # Find all PDFs matching pattern
    pdf_pattern = "*.pdf"
    pdf_files = [Path(f) for f in glob.glob(pdf_pattern)]
    
    if not pdf_files:
        print(f"No PDF files found matching: {pdf_pattern}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to convert\n")
    
    # Output directory
    output_dir = Path("./output/batch_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create converter
    converter = DocumentConverterWrapper(
        pipeline_type="pix2text",
        docling_config=DoclingConfig(),
        pix2text_config=Pix2TextConfig(device="cpu"),
        output_config=OutputConfig(base_dir=output_dir, formats=["md", "html"]),
    )
    
    # Collect batch statistics
    batch_stats = BatchStatistics()
    
    # Process each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Converting: {pdf_file.name}")
        
        result = converter.convert_file(pdf_file, output_dir)
        
        # Add to batch statistics
        stats = ConversionStatistics(
            input_file=pdf_file.name,
            success=result.success,
            elapsed_time=result.elapsed_time,
            output_formats=[f.suffix[1:] for f in result.output_files],
            formula_stats=result.stats.to_dict() if result.stats else None,
            error=result.error,
        )
        batch_stats.add_conversion(stats)
        
        if result.success:
            print(f"  ✓ Completed in {result.elapsed_time:.1f}s")
        else:
            print(f"  ✗ Failed: {result.error}")
        print()
    
    # Display summary
    print("=" * 60)
    batch_stats.print_summary()
    
    # Generate HTML report
    report_path = output_dir / "batch_report.html"
    batch_stats.generate_html_report(report_path)
    print(f"\nHTML report saved: {report_path}")


if __name__ == "__main__":
    main()
