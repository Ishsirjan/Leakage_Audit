# Leakage Audit Protocol (LAP)

**Leakage-Controlled Resting-State Connectome Classification of Schizophrenia**  
MLMI 2026 (MICCAI Workshop) — Poster (camera-ready)

Author: **Ishsirjan Kaur Chandok**  
Institut de Génétique Moléculaire de Montpellier, CNRS UMR 5535,  
1919 Route de Mende, 34293 Montpellier Cedex 5, France  
Email: ishsirjanchandok.iskc@gmail.com

Code: https://github.com/Ishsirjan/Leakage_Audit

---

## What this repo is

Code and results for a **leakage audit** of schizophrenia vs control classification from resting-state fMRI connectomes (COBRE, Schaefer-100).

**Main claim:** Under nested CV with feature construction confined to training folds, honest AUC is ≈ **0.77**. A leaked twenty-edge screen on the same data reports **0.903**. A controlled same-selector contrast (top-20 |t| + logistic) shows optimism **Δ ≈ +0.19 AUC**.

**Protocol check and null control.** UCLA CNP is a second *within-cohort* schizophrenia audit on a 50/50 age/sex/FD-matched subsample (fMRIPrep, not a locked transfer): **0.679 → 0.873** (Δ 0.194, same as COBRE). On synthetic data with no signal, the leakage-free protocol returns chance while a leaked twenty-edge screen reports **0.821** on correlated connectomes and **0.938** on independent features at COBRE dimensions. The observed leaked value of 0.903 sits about 0.08 above the correlated floor. On the matched CNP connectomes, observed leaked 0.873 does not exceed a label-permutation floor of **0.893**.

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
| **UCLA CNP** (optional) | Within-cohort audit (not locked transfer) | [OpenNeuro ds000030](https://openneuro.org/datasets/ds000030); keep files on `D:\Leakage_Audit_data` |

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
python scripts/run_methods_benchmark.py         # needs COBRE
python scripts/run_lap_and_residualisation.py   # needs COBRE
python scripts/run_null_control.py              # self-contained, no data needed
python scripts/run_null_matched.py              # self-contained, no data needed
python scripts/run_null_connectome.py           # self-contained, correlated-edge null
python scripts/run_cnp_balanced.py              # 50/50 CNP match; needs D:\Leakage_Audit_data
python paper/build_figures.py                   # all paper figures from results/*.csv
python paper/build_paper.py                     # print-ready HTML
```

The null-control scripts generate their own synthetic data. UCLA CNP needs
OpenNeuro `ds000030` on `D:\Leakage_Audit_data`.

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

### Protocol check and null control

| Analysis | AUC |
|----------|-----|
| UCLA CNP 50/50 matched, top-20 SAFE | 0.679 ± 0.074 |
| UCLA CNP 50/50 matched, top-20 LEAKED | 0.873 ± 0.081 |
| UCLA CNP 50/50, full connectome, leakage-free | 0.727 |
| **Null data**, COBRE dims — leakage-free | 0.505 ± 0.068 |
| **Null data**, COBRE dims — leaked, independent edges | **0.938 ± 0.017** |
| **Null data**, COBRE dims — leaked, correlated edges | **0.821 ± 0.020** |
| **Null data**, CNP 50/50 — leaked, independent edges | 0.971 |
| **Null data**, CNP 50/50 — leaked, correlated edges | 0.873 |
| **Null data**, CNP 50/50 — leaked, label permutation | 0.893 |

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

Ishsirjan Kaur Chandok — Institut de Génétique Moléculaire de Montpellier, CNRS UMR 5535 — ishsirjanchandok.iskc@gmail.com
