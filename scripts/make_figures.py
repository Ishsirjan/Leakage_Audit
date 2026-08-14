"""Generate paper figures from results CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leakage_audit.config import RESULTS_DIR, WORKDIR, ensure_workdir

plt.rcParams.update({"font.size": 10, "font.family": "serif", "figure.dpi": 300})


def main():
    ensure_workdir()
    bench = RESULTS_DIR / "methods_benchmark_comparison.csv"
    if not bench.is_file():
        bench = WORKDIR / "methods_benchmark_comparison.csv"
    if not bench.is_file():
        raise FileNotFoundError("Run scripts/run_methods_benchmark.py first")

    df = pd.read_csv(bench)
    best = df.sort_values("auc_mean", ascending=False).drop_duplicates("method").sort_values("auc_mean")
    pretty = {
        "elasticnet_4950edges_demo": "Elastic net + demo",
        "elasticnet_4950edges": "Elastic net",
        "pca30_edges_demo": "PCA-30 + demo",
        "cpm10pct_demo": "CPM + demo",
        "top20_train_t_demo": "Top-20 |t| + demo",
        "fixed5_edges_demo": "Fixed 5 + demo*",
        "network28_demo": "Network-28 + demo",
    }
    labels = [pretty.get(m, m) for m in best["method"]]
    colors = ["#c44e52" if "*" in l else "#4c72b0" for l in labels]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    y = np.arange(len(best))
    ax.barh(y, best["auc_mean"], xerr=best["auc_std"], color=colors, edgecolor="black", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.5, ls="--", color="gray")
    ax.set_xlabel("Nested CV AUC")
    ax.set_title("Leakage-free methods comparison")
    plt.tight_layout()
    f1 = WORKDIR / "figure1_methods_comparison.png"
    plt.savefig(f1, bbox_inches="tight")
    plt.close()
    print(f"✓ {f1}")

    gradient = [("Leakage-free", 0.768, 0.076), ("Fixed 5", 0.855, 0.066), ("Variant C", 0.884, 0.049)]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    x = np.arange(3)
    ax.bar(x, [g[1] for g in gradient], yerr=[g[2] for g in gradient],
           color=["#4c72b0", "#dd8452", "#c44e52"], edgecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in gradient])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("Protocol-family leakage gradient")
    plt.tight_layout()
    f2 = WORKDIR / "figure2_leakage_gradient.png"
    plt.savefig(f2, bbox_inches="tight")
    plt.close()
    print(f"✓ {f2}")


if __name__ == "__main__":
    main()
