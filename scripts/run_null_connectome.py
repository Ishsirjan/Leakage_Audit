"""Null control with realistic connectome correlation structure.

The Gaussian null of `run_null_control.py` treats the 4,950 edges as independent, which
they are not: edges sharing a node are correlated by construction, so the effective
number of independent candidate features is smaller and the manufacturable optimism
could be smaller too. Here the null is generated at the level of the time series --
random parcel signals with a shared low-rank component, then Pearson connectomes -- so
the edge dependency structure of a real connectome is reproduced while the group labels
remain pure noise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leakage_audit.protocol import LAPConfig, run_safe_vs_leaked_topk  # noqa: E402

RESULTS = ROOT / "results"
N_LATENT = 5
N_SEEDS = 10

# label, n_subjects, n_rois, n_timepoints -- matched to each analysed cohort
CELLS = [
    ("COBRE-matched, correlated edges", 146, 100, 150),
    ("ABIDE-matched, correlated edges", 844, 111, 196),
]


def null_connectomes(rng, n_subjects, n_rois, n_time):
    """Connectomes with no group structure but realistic edge dependency."""
    iu = np.triu_indices(n_rois, k=1)
    loadings = rng.normal(0, 1, size=(N_LATENT, n_rois))
    edges = []
    for _ in range(n_subjects):
        latent = rng.normal(0, 1, size=(n_time, N_LATENT))
        ts = latent @ loadings + rng.normal(0, 1.6, size=(n_time, n_rois))
        fc = np.clip(np.corrcoef(ts.T), -0.999, 0.999)
        edges.append(np.arctanh(fc[iu]))
    return np.array(edges, dtype=np.float32)


def edge_dependency(X, rng, n_pairs=4000):
    """Mean |correlation| between randomly chosen pairs of edges."""
    i = rng.integers(0, X.shape[1], n_pairs)
    j = rng.integers(0, X.shape[1], n_pairs)
    keep = i != j
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    r = (Xs[:, i[keep]] * Xs[:, j[keep]]).mean(0)
    return float(np.abs(r).mean())


def main():
    rows = []
    for label, n_subjects, n_rois, n_time in CELLS:
        safes, leakeds, deps = [], [], []
        t0 = time.time()
        for s in range(N_SEEDS):
            rng = np.random.default_rng(9000 + s)
            X = null_connectomes(rng, n_subjects, n_rois, n_time)
            y = np.zeros(n_subjects, dtype=int)
            y[rng.choice(n_subjects, size=n_subjects // 2, replace=False)] = 1
            deps.append(edge_dependency(X, rng))
            d = run_safe_vs_leaked_topk(
                X, y, np.arange(n_subjects), demo=None, k=20,
                cfg=LAPConfig(n_splits=5, n_repeats=3, random_state=9000 + s, n_jobs=2),
            )
            safes.append(float(d.loc[d.name.str.contains("SAFE"), "auc_mean"].iloc[0]))
            leakeds.append(float(d.loc[d.name.str.contains("LEAKED"), "auc_mean"].iloc[0]))
            print(f"{label[:5]} seed {s}: safe {safes[-1]:.3f}  leaked {leakeds[-1]:.3f}  "
                  f"mean |r| between edges {deps[-1]:.3f}", flush=True)

        safes, leakeds = np.array(safes), np.array(leakeds)
        rows.append({
            "cell": label,
            "n": n_subjects, "p": n_rois * (n_rois - 1) // 2, "k": 20,
            "n_rois": n_rois, "n_timepoints": n_time,
            "mean_abs_edge_corr": float(np.mean(deps)),
            "auc_safe_mean": safes.mean(), "auc_safe_std": safes.std(),
            "auc_leaked_mean": leakeds.mean(), "auc_leaked_std": leakeds.std(),
            "optimism_mean": (leakeds - safes).mean(),
            "n_seeds": N_SEEDS,
        })
        print(f"\n{label}: safe {safes.mean():.3f} +/- {safes.std():.3f}, "
              f"leaked {leakeds.mean():.3f} +/- {leakeds.std():.3f}, "
              f"edge dependency {np.mean(deps):.3f}  ({time.time() - t0:.0f}s)\n", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "null_control_connectome.csv", index=False)
    print("wrote results/null_control_connectome.csv")


if __name__ == "__main__":
    main()
