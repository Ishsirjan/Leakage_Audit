"""Null control: how large an AUC can leaked feature selection manufacture from noise?

Features are independent standard normal variates and labels are assigned at random,
so the true signal is exactly zero and any AUC above 0.5 is an artefact of the
protocol. Dimensions match the COBRE analysis (n = 146, p = 4,950 candidate edges).
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
RESULTS.mkdir(exist_ok=True)

N_COBRE, P_COBRE = 146, 4950
N_SEEDS = 10


def null_dataset(n, p, rng):
    X = rng.standard_normal((n, p))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n // 2, replace=False)] = 1
    return X, y


def run_cell(n, p, k, seed):
    rng = np.random.default_rng(seed)
    X, y = null_dataset(n, p, rng)
    d = run_safe_vs_leaked_topk(
        X, y, np.arange(n), demo=None, k=k,
        cfg=LAPConfig(n_splits=5, n_repeats=2, random_state=seed),
    )
    safe = float(d.loc[d.name.str.contains("SAFE"), "auc_mean"].iloc[0])
    leaked = float(d.loc[d.name.str.contains("LEAKED"), "auc_mean"].iloc[0])
    return safe, leaked


def summarise(records, **keys):
    safe = np.array([r[0] for r in records])
    leaked = np.array([r[1] for r in records])
    return {
        **keys,
        "auc_safe_mean": safe.mean(), "auc_safe_std": safe.std(),
        "auc_leaked_mean": leaked.mean(), "auc_leaked_std": leaked.std(),
        "optimism_mean": (leaked - safe).mean(), "optimism_std": (leaked - safe).std(),
        "n_seeds": len(records),
    }


def main():
    rows_k, rows_n = [], []

    for k in (5, 20, 100):
        t0 = time.time()
        recs = [run_cell(N_COBRE, P_COBRE, k, 1000 + s) for s in range(N_SEEDS)]
        row = summarise(recs, n=N_COBRE, p=P_COBRE, k=k)
        rows_k.append(row)
        print(f"k={k:3d}  safe {row['auc_safe_mean']:.3f}  "
              f"leaked {row['auc_leaked_mean']:.3f}  "
              f"optimism {row['optimism_mean']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows_k).to_csv(RESULTS / "null_control_by_k.csv", index=False)

    for n in (50, 100, 146, 300, 600):
        t0 = time.time()
        recs = [run_cell(n, P_COBRE, 20, 2000 + s) for s in range(N_SEEDS)]
        row = summarise(recs, n=n, p=P_COBRE, k=20)
        rows_n.append(row)
        print(f"n={n:4d}  safe {row['auc_safe_mean']:.3f}  "
              f"leaked {row['auc_leaked_mean']:.3f}  "
              f"optimism {row['optimism_mean']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows_n).to_csv(RESULTS / "null_control_by_n.csv", index=False)
    print("wrote results/null_control_by_k.csv and null_control_by_n.csv")


if __name__ == "__main__":
    main()
