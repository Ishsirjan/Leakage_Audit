# Leakage Audit Protocol (LAP)

**Leakage-Controlled Resting-State Connectome Classification of Schizophrenia**  
MLMI 2026 (MICCAI Workshop) — Poster (camera-ready)

Author: **Ishsirjan Kaur Chandok**  
IGMM, CNRS UMR 5535, Université de Montpellier  
Email: ishsirjanchandok.iskc@gmail.com

Code: https://github.com/Ishsirjan/Leakage_Audit

---

## What this repo is

Code and results for a **leakage audit** of schizophrenia vs control classification from resting-state fMRI connectomes (COBRE, Schaefer-100).

**Main claim:** Under nested CV with feature construction confined to training folds, honest AUC is ≈ **0.77**. The same data can look like **0.88–0.90** if edges are selected on the full cohort first. A controlled same-selector contrast (top-20 |t| + logistic) shows optimism **Δ ≈ +0.19 AUC**.

This is **not** an apathy biomarker paper (COBRE has no AES/PANSS scores). Literature-guided seeds appear only in a secondary/leaked reference pipeline.

---

## Installable package: `leakage_audit`

```python
from leakage_audit import LAPConfig, run_safe_vs_leaked_topk, vec_upper

X, _ = vec_upper(fc)  # (n_subjects, 4950)
df = run_safe_vs_leaked_topk(X, y, subject_ids, demo=demo, k=20, cfg=LAPConfig())
print(df[["name", "auc_mean", "delta_vs_safe"]])
```

Core modules:

- `leakage_audit/protocol.py` — SAFE vs LEAKED, nested CV, residualisation
- `leakage_audit/data.py` — COBRE loaders + FC cache
- `leakage_audit/config.py` — paths via env vars

---

## Data (not in this repo)

| Dataset | Role | Where to get it |
|---------|------|-----------------|
| **COBRE** (primary) | Train / nested CV | [Figshare 4197885](https://figshare.com/articles/dataset/COBRE_preprocessed_with_NIAK_0_17_-_lightweight_release/4197885) |
| **UCLA CNP** (optional external) | Locked transfer | [OpenNeuro ds000030](https://openneuro.org/datasets/ds000030) |

```powershell
$env:LEAKAGE_AUDIT_COBRE = "C:\Users\YOU\Downloads\4197885"
$env:LEAKAGE_AUDIT_WORKDIR = "C:\Users\YOU\Downloads\lap_outputs"
```

Or edit `leakage_audit/config.py`. Cached connectomes go to `LEAKAGE_AUDIT_WORKDIR` as `subject_fc_matrices.npz`.

---

## Install

```bash
git clone https://github.com/Ishsirjan/Leakage_Audit.git
cd Leakage_Audit
conda activate quant   # or any Python 3.10+ env
pip install -r requirements.txt
pip install -e .
```

---

## Reproduce paper analyses

```bash
python scripts/run_methods_benchmark.py
python scripts/run_lap_and_residualisation.py
python scripts/make_figures.py
python scripts/run_external_ucla_cnp.py   # optional
```

---

## Key results (COBRE)

| Analysis | AUC |
|----------|-----|
| Elastic net + demo (5×10 nested CV) | **0.768 ± 0.076** |
| PCA-30 + demo | 0.771 ± 0.079 |
| Elastic net edges-only | 0.765 ± 0.079 |
| Top-20 \|t\| SAFE (controlled) | 0.711 ± 0.077 |
| Top-20 \|t\| LEAKED (controlled) | 0.903 ± 0.039 |
| Optimism Δ (same selector/classifier) | **+0.192** |
| Subject-level OOF (EN+demo, 5×3) | 0.764 |
| Primary ECE / Brier | 0.116 / 0.220 |
| Locked 70/30 COBRE split (EN) | ~0.73 |

CSV summaries are in `results/`.

---

## Repository layout

```
SchizoApathy/   # local folder name; GitHub repo is Leakage_Audit
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── leakage_audit/                # installable package (pip name: leakage-audit)
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   └── protocol.py               # LAP toolkit
├── scripts/
├── results/
└── docs/
```

**Not included:** raw NIfTI, full UCLA download (~85 GB), `nilearn_cache/`.

---

## Citation

> Chandok IK. Leakage-Controlled Resting-State Connectome Classification of Schizophrenia: A Rigorous Nested Cross-Validation Analysis of the COBRE Cohort. MLMI 2026 (MICCAI Workshop).

## License

MIT — see `LICENSE`.

## Publish / update GitHub

Double-click `push_to_github.bat` (or see `PUSH_TO_GITHUB.md`).  
Then set the repo to **Public**: https://github.com/Ishsirjan/Leakage_Audit

## Contact

Ishsirjan Kaur Chandok — ishsirjanchandok.iskc@gmail.com
