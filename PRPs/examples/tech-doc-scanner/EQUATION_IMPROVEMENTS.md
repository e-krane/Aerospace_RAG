# Equation/Formula Improvements in Docling

## What You've Discovered

Your testing revealed:
- **Standard Pipeline**: Good structure/tables
- **VLM Pipeline**: Good details
- **Both**: Had issues with equations

## Solution: Formula Enrichment

Docling has a dedicated **CodeFormula model** that significantly improves equation handling.

### What Formula Enrichment Does

1. **Analyzes equations** in the document
2. **Extracts LaTeX representations** from formulas
3. **Improves rendering** in both markdown and HTML exports
4. **Uses MathML** syntax in HTML for proper math display

### How to Enable

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_formula_enrichment = True  # This is the key!
```

### Model Used

- **Model**: [CodeFormula](https://huggingface.co/ds4sd/CodeFormula) from DS4SD
- **Purpose**: Specialized in extracting LaTeX from document formulas
- **Additional Benefit**: Also handles code block parsing when `do_code_enrichment = True`

## Your Test Results

You now have outputs in:

1. **`output/with_formulas/`** - Best for equations:
   - Uses CodeFormula model for LaTeX extraction
   - Check `Bruhn_Crippling_short.html` for MathML rendering
   - Markdown has LaTeX representations

2. **`output/comparison/`** - Side-by-side comparison:
   - Standard vs VLM (without formula enrichment)

3. **`output/hybrid/`** - Combined approach:
   - Had formula enrichment enabled already

## Additional Improvements Available

### 1. Picture Description (Vision Models)

For better figure/diagram understanding:

```python
pipeline_options.do_picture_description = True

# Options:
from docling.datamodel.pipeline_options import (
    granite_picture_description,  # IBM Granite Vision
    smolvlm_picture_description,   # SmolVLM
    PictureDescriptionVlmOptions,  # Any HuggingFace VLM
    PictureDescriptionApiOptions,  # Remote API (OpenAI, etc.)
)

# Example with local model:
pipeline_options.picture_description_options = granite_picture_description

# Example with API (requires enable_remote_services=True):
pipeline_options.picture_description_options = PictureDescriptionApiOptions(
    url="http://localhost:8000/v1/chat/completions",
    params=dict(model="MODEL_NAME", max_completion_tokens=200),
    prompt="Describe the image concisely."
)
```

### 2. Code Understanding

For documents with code blocks:

```python
pipeline_options.do_code_enrichment = True
# Uses CodeFormula model to parse code and detect language
```

### 3. OCR Improvements

Try different OCR engines:

```python
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,      # Default, good balance
    TesseractOcrOptions, # More accurate for some docs
    RapidOcrOptions,     # Faster
)

pipeline_options.ocr_options = TesseractOcrOptions()
```

### 4. Table Extraction Tuning

```python
# Already using ACCURATE mode in your tests
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

# Alternative: faster but less accurate
pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# Cell matching control
pipeline_options.table_structure_options.do_cell_matching = True  # Better quality
```

### 5. Image Quality/Scale

```python
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0  # Higher = better quality, larger files
```

## Recommended Configuration for Your Use Case

Based on your document (Bruhn_Crippling.pdf - appears to be engineering/structural):

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
)
from docling_core.types.doc import ImageRefMode

pipeline_options = PdfPipelineOptions()

# Structure (what you found works well)
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.table_structure_options.do_cell_matching = True

# OCR
pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])

# Images
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

# Enrichments - THE KEY IMPROVEMENTS
pipeline_options.do_formula_enrichment = True  # For equations!
pipeline_options.do_code_enrichment = True
pipeline_options.do_picture_classification = True

# Optional: Add picture descriptions if you want diagram explanations
# pipeline_options.do_picture_description = True
# from docling.datamodel.pipeline_options import granite_picture_description
# pipeline_options.picture_description_options = granite_picture_description

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
```

## Performance vs Quality Trade-offs

- **Formula enrichment**: +5 minutes processing time, significant equation quality improvement
- **Picture description**: +significant time (depends on model), adds figure captions
- **Code enrichment**: Minimal time impact
- **TableFormer ACCURATE**: ~2x slower than FAST, much better for complex tables

## Next Steps

1. **Compare outputs**: Check `output/with_formulas/Bruhn_Crippling_short.html` to see MathML rendering
2. **If still issues**: Consider post-processing the LaTeX or using a vision model for picture descriptions
3. **For production**: Balance quality vs processing time based on your needs

## CLI Usage

You can also use these from command line:

```bash
docling Bruhn_Crippling.pdf \
  --enrich-formula \
  --enrich-code \
  --enrich-picture-classes \
  --table-mode accurate \
  --ocr \
  --to md \
  --output ./output/cli_with_formulas
```
