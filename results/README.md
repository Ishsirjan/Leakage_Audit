# Results (paper tables)

CSV summaries from the COBRE analyses, the UCLA CNP 50/50 matched within-cohort
audit, and the null control. Raw MRI is **not** included.

| File | Contents |
|------|----------|
| `methods_benchmark_comparison.csv` | Leakage-free multi-method nested CV (5×10) |
| `lap_controlled_leakage.csv` | SAFE vs LEAKED top-20 (same selector/classifier) |
| `lap_residualisation_sensitivity.csv` | EN / top-20 ± FD residualisation |
| `lap_locked_split_cobre.csv` | Locked 70/30 within-COBRE splits |
| `lap_summary_metrics.csv` | One-row summary metrics |
| `camera_ready_primary_metrics.csv` | Numbers used in camera-ready text |
| `camera_ready_fd_group_stats.csv` | FD by diagnosis (Welch t) |
| `external_cnp_matched50_contrast.csv` | UCLA CNP 50/50: SAFE vs LEAKED top-20 |
| `external_cnp_matched50_full.csv` | UCLA CNP 50/50 full-connectome leakage-free logistic |
| `null_control_by_k.csv` | Null data, COBRE dimensions, k ∈ {5, 20, 100} |
| `null_control_by_n.csv` | Null data, p = 4,950, n = 50–600 |
| `null_control_matched.csv` | Null data (independent features) at each cohort's exact (n, p); CNP is 50/50 |
| `null_control_connectome.csv` | Null data with realistic edge dependency, including CNP 50/50 |
| `null_control_cnp_permutation.csv` | Label permutation of the observed matched CNP connectomes |

Primary paper numbers (5×10): EN+demo **0.768 ± 0.076**. Controlled Δ ≈ **+0.19** (5×3).

UCLA CNP (within-cohort 50/50 age/sex/FD match, not a locked transfer):
**0.679 → 0.873**. On data with no signal the leakage-free protocol returns chance
(0.49–0.52) while the leaked protocol reports **0.821–0.938** at COBRE dimensions.
On the matched CNP connectomes, observed leaked 0.873 does not exceed a
label-permutation floor of **0.893**. Observed leaked COBRE 0.903 sits about 0.08
above its correlated floor.
