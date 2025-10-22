"""Pytest configuration and shared fixtures for Tech Doc Scanner tests."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.fixture
def valid_latex_samples() -> Dict[str, str]:
    """
    Provide valid LaTeX expressions for testing.

    Returns:
        Dictionary of named LaTeX samples that should be considered valid
    """
    return {
        "simple_fraction": r"\frac{1}{2}",
        "nested_fraction": r"\frac{\frac{a}{b}}{c}",
        "equation_with_delimiters": r"\left( \frac{x}{y} \right)",
        "matrix": r"\begin{matrix} a & b \\ c & d \end{matrix}",
        "complex_expression": r"E = mc^2 \cdot \sqrt{\frac{1}{1-v^2/c^2}}",
        "subscript_superscript": r"x_i^2 + y_j^3",
        "greek_letters": r"\alpha + \beta = \gamma",
        "integral": r"\int_0^{\infty} e^{-x} dx",
    }


@pytest.fixture
def invalid_latex_samples() -> Dict[str, str]:
    """
    Provide invalid LaTeX expressions for testing.

    Returns:
        Dictionary of named LaTeX samples with known issues
    """
    return {
        "unbalanced_braces_open": r"\frac{1{2",
        "unbalanced_braces_close": r"\frac{1}{2}}",
        "incomplete_frac": r"\frac{1}",
        "incomplete_subscript": r"x_",
        "incomplete_superscript": r"x^",
        "trailing_backslash": "x + y\\",  # Cannot use raw string ending with backslash
        "unbalanced_left_right": r"\left( x + y",
        "unbalanced_environment": r"\begin{matrix} a & b",
        "excessive_repetition": "a" * 150,
        "too_long": "x" * 6000,
        "incomplete_line_break": r"x + y \\",
    }


@pytest.fixture
def cleanable_latex_samples() -> Dict[str, tuple[str, str]]:
    """
    Provide LaTeX expressions that can be cleaned with expected results.

    Returns:
        Dictionary mapping names to (input, expected_output) tuples
    """
    return {
        "unbalanced_braces": (r"\frac{1{2", r"\frac{1{2}}"),
        "incomplete_frac_at_end": (r"\frac{x}", r"\frac{x}{}"),
        "trailing_backslash": ("x + y\\", "x + y"),  # Cannot use raw string ending with backslash
        "excessive_whitespace": (r"x  +   y", r"x + y"),
        "whitespace_in_braces": (r"{ x }", r"{x}"),
        "unbalanced_left_right": (r"\left( x", r"\left( x \right."),
        "incomplete_subscript": (r"x_", r"x"),
        "incomplete_superscript": (r"y^", r"y"),
        "incomplete_line_break": (r"x \\", r"x"),
    }


@pytest.fixture
def config_dict_sample() -> Dict[str, Any]:
    """
    Provide a sample configuration dictionary.

    Returns:
        Complete configuration dictionary with all sections
    """
    return {
        "docling": {
            "do_ocr": True,
            "ocr_languages": ["en", "de"],
            "do_table_structure": True,
            "table_mode": "accurate",
            "do_cell_matching": True,
            "generate_picture_images": True,
            "images_scale": 2.5,
            "do_picture_classification": False,
            "do_code_enrichment": True,
            "accelerator_device": "cuda",
            "accelerator_threads": 8,
        },
        "pix2text": {
            "enabled": True,
            "device": "cuda",
            "validate": True,
            "clean": True,
            "fallback_to_ocr": True,
            "max_clean_iterations": 5,
            "images_scale": 3.0,
            "expansion_factor": 0.15,
        },
        "output": {
            "formats": ["markdown", "html", "json"],
            "image_mode": "referenced",
            "base_dir": "./test_output",
            "create_subdirs": True,
        },
    }


@pytest.fixture
def temp_yaml_config(tmp_path: Path, config_dict_sample: Dict[str, Any]) -> Path:
    """
    Create a temporary YAML configuration file.

    Args:
        tmp_path: Pytest temporary directory fixture
        config_dict_sample: Configuration dictionary fixture

    Returns:
        Path to temporary YAML configuration file
    """
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict_sample, f)
    return config_path


@pytest.fixture
def minimal_config_dict() -> Dict[str, Any]:
    """
    Provide a minimal configuration dictionary with defaults.

    Returns:
        Minimal configuration dictionary
    """
    return {
        "docling": {},
        "pix2text": {},
        "output": {},
    }


@pytest.fixture
def invalid_config_dict() -> Dict[str, Any]:
    """
    Provide an invalid configuration dictionary.

    Returns:
        Configuration dictionary with invalid values
    """
    return {
        "docling": {
            "table_mode": "invalid_mode",
            "accelerator_device": "quantum_computer",
        },
        "pix2text": {
            "device": "magic",
        },
        "output": {
            "image_mode": "telepathy",
        },
    }


@pytest.fixture
def empty_yaml_file(tmp_path: Path) -> Path:
    """
    Create an empty YAML file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to empty YAML file
    """
    empty_path = tmp_path / "empty.yaml"
    empty_path.touch()
    return empty_path
