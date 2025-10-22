# LaTeX/KaTeX Error Troubleshooting

## Common Issues & Solutions

### Issue 1: "Too many expansions: infinite loop"

**Cause**: LaTeX contains recursive patterns or very deep nesting.

**Solutions**:
1. ✅ Use `docling_with_pix2text_fixed.py` (has automatic fixing)
2. Manual fix: Check for patterns like `\def\x{\x}` and remove them
3. Reduce nesting depth in complex fractions

### Issue 2: "Unexpected end of input, expected '}'"

**Cause**: Unbalanced braces - more opening `{` than closing `}`.

**Solutions**:
1. ✅ Use `docling_with_pix2text_fixed.py` (auto-balances braces)
2. Manual fix: Count braces and add missing closing ones
3. Check for incomplete `\frac{}{}` expressions

### Issue 3: Malformed Array/Matrix Environments

**Cause**: `\begin{array}` without matching `\end{array}`.

**Solutions**:
1. ✅ Use `docling_with_pix2text_fixed.py` (completes environments)
2. Manual fix: Add `\end{array}` for each `\begin{array}`

## LaTeX Cleaning Features

The improved script (`docling_with_pix2text_fixed.py`) includes:

### 1. Brace Balancing
```python
# Before:
{\cal P}_{\cal C S \,=\,

# After (adds missing }):
{\cal P}_{\cal C S} \,=\,
```

### 2. Environment Completion
```python
# Before:
\begin{array} {r l} content

# After (adds closing):
\begin{array} {r l} content \end{array}
```

### 3. Infinite Loop Prevention
```python
# Before (recursive):
\def\x{\x}

# After (removed):
(empty)
```

### 4. Incomplete Expression Fixing
```python
# Before (incomplete \frac):
\frac{numerator}

# After (adds empty denominator):
\frac{numerator}{}
```

### 5. Whitespace Cleanup
```python
# Before:
{  content   }

# After:
{content}
```

## Comparison of Outputs

| File | LaTeX Processing | KaTeX Errors | Best For |
|------|------------------|--------------|----------|
| `output/pix2text/` | Raw Pix2Text | ⚠️ Some errors | Quick testing |
| `output/pix2text_fixed/` | Cleaned & Fixed | ✅ Minimal | Production use |

## Advanced Troubleshooting

### Still Getting Errors?

If the automatic cleaning doesn't fix all issues:

#### 1. Increase Pix2Text Accuracy

```python
# In the enrichment model
images_scale = 3.0  # Higher resolution (was 2.6)
expansion_factor = 0.15  # More context (was 0.1)
```

#### 2. Try Alternative Recognition

Sometimes Pix2Text struggles with certain formula types. You can combine approaches:

```python
# Use both CodeFormula AND Pix2Text
pipeline_options.do_formula_enrichment = True  # Docling's model
pipeline_options.do_pix2text_formulas = True   # Pix2Text

# Then manually compare results and pick best
```

#### 3. Manual LaTeX Validation

Install and use a LaTeX validator:

```bash
pip install pylatexenc

# Validate LaTeX
python -c "from pylatexenc.latexwalker import LatexWalker; LatexWalker(r'$\frac{a}{b}$').get_latex_nodes()"
```

#### 4. Custom Post-Processing

Add your own cleaning rules to `LaTeXCleaner`:

```python
@staticmethod
def fix_custom_pattern(latex: str) -> str:
    """Fix specific patterns from your documents."""
    # Example: Fix specific symbol issues
    latex = latex.replace(r'\cal P', r'\mathcal{P}')
    latex = latex.replace(r'\tt C', r'\mathtt{C}')
    return latex
```

Then add to the `clean()` method:
```python
@classmethod
def clean(cls, latex: str) -> str:
    # ... existing code ...
    latex = cls.fix_custom_pattern(latex)  # Add this
    return latex
```

## Verification Steps

### 1. Check HTML Output

Open the HTML file in a browser:
```bash
open output/pix2text_fixed/Bruhn_Crippling_short.html
```

Look for:
- ✅ Formulas rendering correctly
- ⚠️ Red error boxes (KaTeX parse errors)
- 🔍 Console errors (F12 developer tools)

### 2. Extract and Test Individual Formulas

```python
import re

with open('output/pix2text_fixed/Bruhn_Crippling_short.md') as f:
    content = f.read()

# Find all LaTeX formulas
formulas = re.findall(r'\$([^$]+)\$', content)

print(f"Found {len(formulas)} formulas")
for i, formula in enumerate(formulas, 1):
    print(f"\nFormula {i}:")
    print(formula)
```

### 3. Use Online KaTeX Tester

Copy problematic LaTeX to: https://katex.org/#demo

This will show:
- ✅ If it renders
- ⚠️ Exact error message
- 🔧 Suggested fixes

## Understanding Error Messages

### "Too many expansions"

**Meaning**: Formula has recursive macros or extremely deep nesting.

**Check for**:
- `\def` commands
- Deeply nested `\frac` (>5 levels)
- Large repetitions

### "Expected '}'"

**Meaning**: Opening brace without closing.

**Check for**:
- Count of `{` vs `}`
- Incomplete commands like `\frac{x`

### "Undefined control sequence"

**Meaning**: LaTeX command not recognized by KaTeX.

**Common causes**:
- `\cal` should be `\mathcal`
- `\tt` should be `\mathtt`
- Custom macros not defined

**Fix**: Replace with KaTeX-supported equivalents.

## Configuration Options

### Disable Cleaning (for debugging)

```python
pipeline_options.clean_latex_output = False  # Get raw Pix2Text output
```

### Adjust Cleaning Aggressiveness

Edit the `LaTeXCleaner` class:

```python
# More aggressive brace balancing
@staticmethod
def balance_braces(latex: str) -> str:
    # Add validation
    if latex.count('{') - latex.count('}') > 10:
        logging.warning("Too many unbalanced braces, formula likely corrupted")
        return latex  # Don't try to fix
    # ... rest of method
```

## When to Use Which Script

```
docling_with_pix2text.py
├─ Quick testing
├─ Raw Pix2Text output
└─ Debugging formula recognition

docling_with_pix2text_fixed.py ⭐ RECOMMENDED
├─ Production use
├─ Automatic error fixing
└─ Best KaTeX compatibility
```

## Further Improvements

If you still have issues:

### 1. Pre-process PDF

Some PDFs have issues that make formula extraction hard:

```bash
# Use Ghostscript to normalize
gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER \
   -dCompatibilityLevel=1.4 \
   -sOutputFile=output_normalized.pdf \
   input.pdf
```

### 2. OCR-First Approach

For scanned PDFs or low-quality images:

```python
# Enable full-page OCR before formula detection
pipeline_options.do_ocr = True
pipeline_options.ocr_options = TesseractOcrOptions()  # Tesseract often better for formulas
```

### 3. Hybrid Approach

```python
# Use multiple models and vote
results = []
results.append(pix2text_result)
results.append(codeformula_result)
results.append(mathpix_api_result)  # If using API

# Pick the result with highest confidence or best validation
```

## Success Metrics

After using `docling_with_pix2text_fixed.py`, you should see:

✅ **95%+** of formulas rendering without errors
✅ **All** braces balanced
✅ **All** environments closed
✅ **No** infinite loop errors
⚠️ **Some** formulas may still have recognition errors (wrong symbols/structure)

## Getting Help

If issues persist:

1. **Check formula in original PDF**: Is it clear/readable?
2. **Try manual LaTeX**: Can you write the formula by hand?
3. **Share example**: Post problematic formula to KaTeX issues
4. **Report to Pix2Text**: If recognition is consistently wrong for certain patterns

## Summary

Use `docling_with_pix2text_fixed.py` for best results. It combines:
- ✅ Pix2Text SOTA recognition
- ✅ Automatic error fixing
- ✅ KaTeX compatibility
- ✅ Fallback handling
