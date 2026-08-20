# Camera-ready paper (MLMI 2026, ID 35)

**Leakage-Controlled Resting-State Connectome Classification of Schizophrenia:
A Nested Cross-Validation Audit of the COBRE Cohort**

Current length: 9 pages (body through Disclosure plus references; limit: 8.5 + 2).

## Files

| File | Purpose |
| --- | --- |
| `paper.tex`, `references.bib` | Springer LNCS source, the format the proceedings need |
| `Paper-35_camera_ready.pdf` | Compiled manuscript, 9 pages |
| `Paper-35_camera_ready.html` | Self-contained print-ready manuscript (Ctrl+P → Save as PDF, A4, margins "None", background graphics on) |
| `figures/*.pdf`, `figures/*.png` | Vector and raster versions of all eight figures |
| `build_figures.py` | Regenerates every figure from `../results/*.csv` |
| `build_paper.py` | Regenerates the HTML manuscript with figures embedded |
| `ChangeAfterReview-35.txt` | Point-by-point response to the three reviews |

## Rebuild

```powershell
python build_figures.py   # writes figures/*.pdf and figures/*.png
python build_paper.py     # writes Paper-35_camera_ready.html
```

Every number in the figures is read from the result CSVs; none is hard-coded except the
eight leaked-model coefficients, which come from the accepted manuscript, and the two
COBRE reference values used in Fig. 8.

Fig. 8 needs the CNP 50/50 contrast and null-control CSVs. If they are missing,
regenerate them first (the null scripts need no imaging data):

```powershell
python ..\scripts\run_cnp_balanced.py     # needs OpenNeuro ds000030 on D:\Leakage_Audit_data
python ..\scripts\run_null_control.py
python ..\scripts\run_null_matched.py
python ..\scripts\run_null_connectome.py
```

## Compile the LNCS version (Overleaf)

1. Create a project from the official Springer LNCS template (`llncs.cls`).
2. Upload `paper.tex`, `references.bib`, and the `figures/` directory.
3. Set `paper.tex` as the main document; compiler **pdfLaTeX**; run BibTeX once.
4. Check that the PDF is un-anonymised and keeps the **Disclosure of Interests** block.

Do not alter LNCS margins or fonts, and avoid manual `\vspace` to fit the page limit.

## Submission checklist

- `paper.tex`, `references.bib`, `figures/`
- Compiled PDF
- `ChangeAfterReview-35.txt` (paste into a `.docx` if the portal insists on Word)
- Signed License-to-Publish (not included here)

## Code and data

Protocol and scripts: https://github.com/Ishsirjan/Leakage_Audit
Imaging data: COBRE release, Figshare acquisition 4197885.
