# Pix2Text Integration with Docling

## Overview

This integration combines the strengths of both tools:

- **Docling**: Excellent structure detection (tables, layout, document organization)
- **Pix2Text**: State-of-the-art (SOTA) mathematical formula recognition

## Why Pix2Text?

Pix2Text uses specialized models:
- **MFD (Mathematical Formula Detection) 1.5**: Detects where formulas appear
- **MFR (Mathematical Formula Recognition) 1.5**: Converts formulas to LaTeX with SOTA accuracy

The MFR 1.5 model achieves state-of-the-art results on mathematical formula recognition benchmarks.

## Installation

```bash
# Basic installation
pip install pix2text

# Or with uv
uv add pix2text

# For multilingual support
pip install pix2text[multilingual]
```

## How It Works

The custom enrichment model (`docling_with_pix2text.py`) works as follows:

1. **Docling processes the PDF**:
   - Detects document layout
   - Extracts tables with TableFormer ACCURATE mode
   - Identifies formula regions (labeled as `FORMULA`)
   - Performs OCR on text

2. **Custom Pix2Text enrichment**:
   - For each detected formula element
   - Crops the formula image with expanded context
   - Passes it to Pix2Text's MFR model
   - Replaces formula text with extracted LaTeX

3. **Export**:
   - Markdown with accurate LaTeX formulas
   - HTML with proper rendering

## Architecture

```python
class Pix2TextFormulaEnrichmentModel(BaseItemAndImageEnrichmentModel):
    """
    Custom enrichment that:
    - Inherits from Docling's base enrichment model
    - Processes only FORMULA elements
    - Uses Pix2Text for LaTeX extraction
    """

class Pix2TextPipeline(StandardPdfPipeline):
    """
    Extended pipeline that:
    - Uses StandardPdfPipeline for structure
    - Adds Pix2Text enrichment to pipeline
    - Keeps backend for image cropping
    """
```

## Configuration Options

### Pipeline Options

```python
pipeline_options = Pix2TextPipelineOptions()

# Structure (Docling's strength)
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

# OCR
pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])

# Images
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0  # Higher = better quality

# Enrichments
pipeline_options.do_picture_classification = True
pipeline_options.do_code_enrichment = True

# Pix2Text formula recognition (NEW!)
pipeline_options.do_pix2text_formulas = True
```

### Enrichment Model Tuning

In `Pix2TextFormulaEnrichmentModel`:

```python
images_scale = 2.6  # Resolution for formula images (higher = better accuracy)
expansion_factor = 0.1  # Context around formula (0.0-0.3 recommended)
```

## Usage

```bash
uv run docling_with_pix2text.py
```

Output will be in `./output/pix2text/`:
- `Bruhn_Crippling_short.md` - Markdown with Pix2Text LaTeX
- `Bruhn_Crippling_short.html` - HTML rendering

## Comparison of Approaches

| Approach | Structure | Tables | Formulas | Speed | Best For |
|----------|-----------|--------|----------|-------|----------|
| Standard Pipeline | ✓✓✓ | ✓✓✓ | ✓ | Fast | General documents |
| VLM Pipeline | ✓✓ | ✓✓ | ✓✓ | Slow | Detail-rich docs |
| With Formula Enrichment | ✓✓✓ | ✓✓✓ | ✓✓ | Medium | Math-heavy docs |
| **With Pix2Text** | **✓✓✓** | **✓✓✓** | **✓✓✓** | **Medium** | **Engineering/Technical** |

## Expected Results

### Before (Standard/VLM):
- Formulas recognized as text with OCR
- May have character recognition errors
- Special symbols often misidentified
- No proper LaTeX structure

### After (Pix2Text Integration):
- Formulas converted to proper LaTeX
- State-of-the-art accuracy on mathematical symbols
- Proper structure preservation (fractions, exponents, etc.)
- Better rendering in markdown and HTML

## Performance Considerations

- **Model Loading**: ~2-5 seconds first time
- **Per Formula**: ~0.1-0.5 seconds per formula
- **Memory**: +500MB for Pix2Text models
- **Total Overhead**: Depends on formula count (typically +30-60 seconds for a 10-page technical paper)

## Troubleshooting

### Issue: "pix2text not installed"
**Solution**: Run `pip install pix2text` or `uv add pix2text`

### Issue: Out of memory
**Solution**: Reduce `images_scale` from 2.6 to 2.0 or lower

### Issue: Formulas not detected
**Solution**:
1. Check if formulas are labeled as `FORMULA` in base Docling output
2. Increase `expansion_factor` if formulas are cropped too tightly
3. Try VLM pipeline first to improve formula detection

### Issue: Poor LaTeX quality
**Solution**:
1. Increase `images_scale` (try 3.0)
2. Ensure source PDF has good resolution
3. Check if formulas are images vs. text in PDF

## Advanced: Custom Pix2Text Configuration

You can customize Pix2Text initialization:

```python
class Pix2TextFormulaEnrichmentModel(BaseItemAndImageEnrichmentModel):
    def __init__(self, enabled: bool, custom_config: dict = None):
        self.enabled = enabled
        if self.enabled:
            # Custom Pix2Text configuration
            config = custom_config or {
                'analyzer': {'model_name': 'mfd'},
                'formula': {'model_name': 'mfr-pro'},  # Use pro model
            }
            self.p2t = Pix2Text.from_config(config)
```

## Future Enhancements

Potential improvements:
1. **Batch processing**: Process multiple formulas in one Pix2Text call
2. **Confidence scores**: Filter low-confidence results
3. **Fallback**: Use Docling's CodeFormula if Pix2Text fails
4. **Hybrid mode**: Combine results from both models
5. **Post-processing**: Validate and clean LaTeX output

## Credits

- **Docling**: IBM Research - Document understanding
- **Pix2Text**: breezedeus - Mathematical formula recognition
- **Integration**: Custom enrichment model following Docling's pipeline pattern
