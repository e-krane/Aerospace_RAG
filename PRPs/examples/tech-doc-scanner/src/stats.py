"""
Statistics collection and reporting module.

Provides unified statistics tracking and reporting for document conversions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .enrichment.base import EnrichmentStats


@dataclass
class ConversionStatistics:
    """
    Statistics for a single conversion.

    Attributes:
        input_file: Input file name
        success: Whether conversion succeeded
        elapsed_time: Conversion time in seconds
        output_formats: List of generated output formats
        formula_stats: Formula recognition statistics (if applicable)
        error: Error message if conversion failed
    """

    input_file: str
    success: bool
    elapsed_time: float
    output_formats: List[str] = field(default_factory=list)
    formula_stats: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "input_file": self.input_file,
            "success": self.success,
            "elapsed_time": self.elapsed_time,
            "output_formats": self.output_formats,
            "formula_stats": self.formula_stats,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class BatchStatistics:
    """
    Statistics for a batch of conversions.

    Attributes:
        conversions: List of individual conversion statistics
        total_files: Total number of files processed
        successful: Number of successful conversions
        failed: Number of failed conversions
        total_time: Total processing time in seconds
        average_time: Average time per conversion
    """

    conversions: List[ConversionStatistics] = field(default_factory=list)

    def add_conversion(self, stats: ConversionStatistics) -> None:
        """Add conversion statistics."""
        self.conversions.append(stats)

    @property
    def total_files(self) -> int:
        """Total number of files processed."""
        return len(self.conversions)

    @property
    def successful(self) -> int:
        """Number of successful conversions."""
        return sum(1 for c in self.conversions if c.success)

    @property
    def failed(self) -> int:
        """Number of failed conversions."""
        return sum(1 for c in self.conversions if not c.success)

    @property
    def total_time(self) -> float:
        """Total processing time in seconds."""
        return sum(c.elapsed_time for c in self.conversions)

    @property
    def average_time(self) -> float:
        """Average time per conversion."""
        return self.total_time / self.total_files if self.total_files > 0 else 0.0

    @property
    def total_formulas(self) -> int:
        """Total formulas processed across all conversions."""
        total = 0
        for c in self.conversions:
            if c.formula_stats and "total" in c.formula_stats:
                total += c.formula_stats["total"]
        return total

    @property
    def total_recognized(self) -> int:
        """Total formulas successfully recognized."""
        total = 0
        for c in self.conversions:
            if c.formula_stats and "recognized" in c.formula_stats:
                total += c.formula_stats["recognized"]
        return total

    @property
    def overall_success_rate(self) -> float:
        """Overall formula recognition success rate."""
        if self.total_formulas == 0:
            return 0.0

        successful = 0
        for c in self.conversions:
            if c.formula_stats:
                recognized = c.formula_stats.get("recognized", 0)
                fallback = c.formula_stats.get("fallback_used", 0)
                successful += recognized - fallback

        return (successful / self.total_formulas) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_files": self.total_files,
                "successful": self.successful,
                "failed": self.failed,
                "total_time": self.total_time,
                "average_time": self.average_time,
                "total_formulas": self.total_formulas,
                "total_recognized": self.total_recognized,
                "success_rate": self.overall_success_rate,
            },
            "conversions": [c.to_dict() for c in self.conversions],
        }

    def print_summary(self) -> None:
        """Print statistics summary to console."""
        print("\n" + "=" * 70)
        print("Batch Conversion Statistics")
        print("=" * 70)
        print(f"Total files:      {self.total_files}")
        print(f"Successful:       {self.successful} ({self.successful/self.total_files*100:.1f}%)")
        print(f"Failed:           {self.failed}")
        print(f"Total time:       {self.total_time:.1f}s")
        print(f"Average time:     {self.average_time:.1f}s per file")

        if self.total_formulas > 0:
            print("\nFormula Recognition:")
            print(f"  Total formulas:   {self.total_formulas}")
            print(f"  Recognized:       {self.total_recognized}")
            print(f"  Success rate:     {self.overall_success_rate:.1f}%")

        print("=" * 70 + "\n")

    def generate_markdown_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate Markdown report.

        Args:
            output_path: Optional path to save report

        Returns:
            Markdown report as string
        """
        lines = [
            "# Conversion Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total files:** {self.total_files}",
            f"- **Successful:** {self.successful} ({self.successful/self.total_files*100:.1f}%)",
            f"- **Failed:** {self.failed}",
            f"- **Total time:** {self.total_time:.1f}s",
            f"- **Average time:** {self.average_time:.1f}s per file",
            "",
        ]

        if self.total_formulas > 0:
            lines.extend(
                [
                    "## Formula Recognition",
                    "",
                    f"- **Total formulas:** {self.total_formulas}",
                    f"- **Recognized:** {self.total_recognized}",
                    f"- **Success rate:** {self.overall_success_rate:.1f}%",
                    "",
                ]
            )

        lines.extend(
            [
                "## Individual Conversions",
                "",
                "| File | Status | Time (s) | Formulas | Recognized | Success Rate |",
                "|------|--------|----------|----------|------------|--------------|",
            ]
        )

        for conv in self.conversions:
            status = "✓" if conv.success else "✗"
            formulas = conv.formula_stats.get("total", 0) if conv.formula_stats else 0
            recognized = conv.formula_stats.get("recognized", 0) if conv.formula_stats else 0
            fallback = conv.formula_stats.get("fallback_used", 0) if conv.formula_stats else 0

            if formulas > 0:
                success_rate = ((recognized - fallback) / formulas) * 100
                rate_str = f"{success_rate:.1f}%"
            else:
                rate_str = "N/A"

            lines.append(
                f"| {conv.input_file} | {status} | {conv.elapsed_time:.1f} | "
                f"{formulas} | {recognized} | {rate_str} |"
            )

        report = "\n".join(lines)

        if output_path:
            output_path.write_text(report, encoding="utf-8")

        return report

    def generate_html_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate HTML report.

        Args:
            output_path: Optional path to save report

        Returns:
            HTML report as string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Conversion Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; margin-top: 0; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-card.failed {{ border-left-color: #f44336; }}
        .stat-card h3 {{ margin: 0; color: #666; font-size: 14px; }}
        .stat-card .value {{ font-size: 32px; font-weight: bold; color: #333; margin: 10px 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
            color: #666;
        }}
        .success {{ color: #4CAF50; }}
        .failed {{ color: #f44336; }}
        .timestamp {{ color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Conversion Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <h3>Total Files</h3>
                <div class="value">{self.total_files}</div>
            </div>
            <div class="stat-card">
                <h3>Successful</h3>
                <div class="value">{self.successful}</div>
                <small>{self.successful/self.total_files*100:.1f}%</small>
            </div>
            <div class="stat-card failed">
                <h3>Failed</h3>
                <div class="value">{self.failed}</div>
            </div>
            <div class="stat-card">
                <h3>Average Time</h3>
                <div class="value">{self.average_time:.1f}s</div>
            </div>
        </div>
"""

        if self.total_formulas > 0:
            html += f"""
        <h2>Formula Recognition</h2>
        <div class="summary">
            <div class="stat-card">
                <h3>Total Formulas</h3>
                <div class="value">{self.total_formulas}</div>
            </div>
            <div class="stat-card">
                <h3>Recognized</h3>
                <div class="value">{self.total_recognized}</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="value">{self.overall_success_rate:.1f}%</div>
            </div>
        </div>
"""

        html += """
        <h2>Individual Conversions</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Status</th>
                    <th>Time (s)</th>
                    <th>Formulas</th>
                    <th>Recognized</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        for conv in self.conversions:
            status_class = "success" if conv.success else "failed"
            status_icon = "✓" if conv.success else "✗"
            formulas = conv.formula_stats.get("total", 0) if conv.formula_stats else 0
            recognized = conv.formula_stats.get("recognized", 0) if conv.formula_stats else 0
            fallback = conv.formula_stats.get("fallback_used", 0) if conv.formula_stats else 0

            if formulas > 0:
                success_rate = ((recognized - fallback) / formulas) * 100
                rate_str = f"{success_rate:.1f}%"
            else:
                rate_str = "N/A"

            html += f"""
                <tr>
                    <td>{conv.input_file}</td>
                    <td class="{status_class}">{status_icon}</td>
                    <td>{conv.elapsed_time:.1f}</td>
                    <td>{formulas}</td>
                    <td>{recognized}</td>
                    <td>{rate_str}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        if output_path:
            output_path.write_text(html, encoding="utf-8")

        return html

    def save_json(self, output_path: Path) -> None:
        """
        Save statistics as JSON.

        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
