# Aerospace Structures LaTeX Corpus

## Overview
This directory contains documentation and metadata for the Aerospace Structures textbook LaTeX corpus used in the RAG system.

## Corpus Statistics
- **Total Chapters**: 18
- **Total Lines**: 20,667
- **Total Size**: ~1.75 MB
- **Estimated Figures**: 395+ (PDF format)
- **Estimated Equations**: 500+
- **Estimated Pages**: 600
- **Language**: English
- **Domain**: Aerospace Engineering

## Document Structure

### Source Location
`Documents/Aerospace_Structures_LaTeX/`

### Chapter Organization
Chapters are numbered Ch01_4P.tex through Ch18_4P.tex, with the following topics:

1. **Ch01**: Function of flight vehicle structural members (introductory, 236 lines)
2. **Ch02**: Aircraft loads (foundational, 763 lines)
3. **Ch03**: Elements of a thin-walled bar theory (intermediate, 1480 lines)
4. **Ch04**: Some aspects of the structural analysis (advanced, 1691 lines)
5. **Ch05**: Work and energy methods (advanced, 858 lines)
6. **Ch06**: Applications of Castigliano's Theorems (advanced, 1106 lines)
7. **Ch07**: Arches, rings and fuselage frames (intermediate, 1028 lines)
8. **Ch08**: Laminated bars of fiber-reinforced polymer composites (advanced, 3299 lines - LARGEST)
9. **Ch09**: Failure initiation in FRP composites (advanced, 593 lines)
10. **Ch10**: Structural stability of discrete conservative systems (advanced, 846 lines)
11. **Ch11**: Buckling of columns and plates (advanced, 1328 lines)
12. **Ch12**: Introduction to aeroelasticity (advanced, 470 lines)
13. **Ch13**: Fracture of cracked members (advanced, 798 lines)
14. **Ch14**: Design of a landing strut and wing spar (practical, 438 lines)
15. **Ch15**: Direct stiffness method (computational, 718 lines)
16. **Ch16**: Applications of the direct stiffness method (computational, 1607 lines)
17. **Ch17**: Finite element method (computational, 1587 lines)
18. **Ch18**: Introduction to flexible body dynamics (advanced, 1821 lines)

### Appendix
- **App_4P.tex**: Appendix A (reference material, 102KB)

## Selected Test Chapters

For initial testing and validation, we have selected 5 representative chapters:

### 1. Chapter 01 - Function of flight vehicle structural members
- **Rationale**: Introductory chapter with basic concepts and terminology
- **Complexity**: Low
- **File**: Ch01_4P.tex (236 lines, ~6 figures)
- **Topics**: Aircraft structures basics, material selection, fabrication
- **Equation Density**: Low-Medium

### 2. Chapter 03 - Elements of a thin-walled bar theory
- **Rationale**: Core intermediate theory with extensive equations
- **Complexity**: Medium-High
- **File**: Ch03_4P.tex (1480 lines, ~24 figures)
- **Topics**: Thin-walled structures, shear flow, torsion
- **Equation Density**: High

### 3. Chapter 06 - Applications of Castigliano's Theorems
- **Rationale**: Energy methods with complex derivations
- **Complexity**: High
- **File**: Ch06_4P.tex (1106 lines, ~34 figures)
- **Topics**: Energy principles, virtual work, deflection analysis
- **Equation Density**: Very High

### 4. Chapter 08 - Laminated bars of fiber-reinforced polymer composites
- **Rationale**: Largest chapter with modern composite materials
- **Complexity**: Very High
- **File**: Ch08_4P.tex (3299 lines, ~15 figures)
- **Topics**: Composite mechanics, laminate theory, failure analysis
- **Equation Density**: Very High
- **Notes**: Most extensive chapter, excellent test of chunking strategies

### 5. Chapter 11 - Buckling of columns and plates
- **Rationale**: Critical stability topic with practical applications
- **Complexity**: High
- **File**: Ch11_4P.tex (1328 lines, ~38 figures)
- **Topics**: Column buckling, plate buckling, stability analysis
- **Equation Density**: Very High

**Total Test Corpus**: 8,447 lines (~40% of full corpus)

## Technical Details

### LaTeX Structure
- **Document Class**: AeroStructure-ERJohnson.cls (custom)
- **Cross-References**: Uses `crosslink.tex` for inter-chapter references
- **Figures**: External PDF files (Figure_##-##.pdf pattern)
- **Compilation**: Requires full LaTeX toolchain with custom class

### Dependencies
- Custom document class: AeroStructure-ERJohnson.cls
- Cross-reference system: crosslink.tex
- External figures: 395+ PDF files
- Special commands and macros defined in preambles

### Equation Characteristics
- Mix of inline math ($...$) and display math ($$...$$, equation environments)
- Complex multi-line derivations
- Matrix notation
- Vector notation with custom commands
- Extensive use of subscripts and superscripts
- Greek symbols and special mathematical operators

### Cross-References
- Chapters reference each other using `\myexternaldocument{}`
- Figure references across chapters
- Equation references across sections
- Heavy use of `\label{}` and `\ref{}` commands

## Processing Considerations

### Docling Parser
- Should preserve equation LaTeX notation
- Extract figure bounding boxes from embedded PDFs
- Maintain section hierarchy
- Handle custom LaTeX commands
- Expected speed: ~3.7 seconds/page = ~30 minutes for full corpus

### Chunking Strategy
- Respect section boundaries (chapter → section → subsection)
- Never split equations across chunks
- Keep equation + context together
- Target chunk size: 500-1000 tokens with 100-token overlap
- Special handling for long derivations in Ch06, Ch08

### Quality Targets
- **Equation Preservation**: >95% accuracy
- **Figure Extraction**: >90% success rate
- **Section Structure**: 100% preservation
- **Cross-References**: Track and maintain

## Validation Checklist
- [ ] All 18 chapter files accessible
- [ ] Appendix file accessible
- [ ] Figure PDFs present (395+ files)
- [ ] Document class file present
- [ ] Crosslink file present
- [ ] Test compilation of sample chapters
- [ ] Verify equation rendering
- [ ] Verify figure references

## Next Steps
1. Run Docling parser on 5 test chapters
2. Validate equation preservation rate
3. Assess chunking quality
4. Generate embeddings for test set
5. Build evaluation dataset from test chapters
6. Expand to full corpus after validation

## Notes
- Chapter numbering uses labels (ch1, ch2, etc.) not sequential
- Some chapters have short titles in headers `\chapter[short]{long}`
- Line breaks in titles use `\break` command
- Figures are pre-rendered as PDFs, not inline LaTeX
- Heavy mathematical content requires specialized parsing

## References
- Main PDF: `Documents/Aerospace_Structures_LaTeX/Aerospace_Structures+AppendixA.pdf`
- Manifest: `data/corpus_manifest.json`
- This document: `data/CORPUS_README.md`

---
*Document created: 2025-10-22*
*Project: Aerospace LaTeX RAG System*
*Phase: Phase 1 - Foundation*
