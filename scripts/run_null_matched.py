"""Null control at the exact dimensions of each analysed cohort.

Gives the reference value a leaked pipeline reports when the data contain no signal
whatsoever, so that the leaked estimates obtained on COBRE and ABIDE can be read
against the level their own design can manufacture.
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
N_SEEDS = 10

CELLS = [
    ("COBRE-matched", 146, 4950),
    ("ABIDE-matched", 844, 6105),
]


def main():
    rows = []
    for label, n, p in CELLS:
        t0 = time.time()
        safes, leakeds = [], []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(7000 + s)
            X = rng.standard_normal((n, p)).astype(np.float32)
            y = np.zeros(n, dtype=int)
            y[rng.choice(n, size=n // 2, replace=False)] = 1
            d = run_safe_vs_leaked_topk(
                X, y, np.arange(n), demo=None, k=20,
                cfg=LAPConfig(n_splits=5, n_repeats=3, random_state=7000 + s, n_jobs=2),
            )
            safes.append(float(d.loc[d.name.str.contains("SAFE"), "auc_mean"].iloc[0]))
            leakeds.append(float(d.loc[d.name.str.contains("LEAKED"), "auc_mean"].iloc[0]))
        safes, leakeds = np.array(safes), np.array(leakeds)
        rows.append({
            "cell": label, "n": n, "p": p, "k": 20,
            "auc_safe_mean": safes.mean(), "auc_safe_std": safes.std(),
            "auc_leaked_mean": leakeds.mean(), "auc_leaked_std": leakeds.std(),
            "optimism_mean": (leakeds - safes).mean(),
            "n_seeds": N_SEEDS,
        })
        print(f"{label}: safe {safes.mean():.3f} +/- {safes.std():.3f}, "
              f"leaked {leakeds.mean():.3f} +/- {leakeds.std():.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "null_control_matched.csv", index=False)
    print("wrote results/null_control_matched.csv")


if __name__ == "__main__":
    main()
